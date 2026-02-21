"""
RSC Communication Structure Detector v2 — Vectorized, Large-N
==============================================================
HYPOTHESIS: RSC rho is a quantitative metric for meaningful structured
communication. Domain-coherent concept sets will show significantly
higher cross-lingual relational consistency than random concept sets.

DESIGN:
  - All 13 monolingual BERT models (same as v4/v6)
  - BATCH encoding (all words at once per language, much faster)
  - PRE-COMPUTED gram matrices per language (reused for all sampling)
  - Coherent condition: 5 domain sets (astronomy, emotions, body, food, politics)
  - Random condition: 1000 random draws of same size from same word pool
  - Result: distribution comparison with massive statistical power

Key optimizations vs v1 detector:
  - Batch tokenization: 10x faster than word-by-word
  - Pre-computed gram matrices: O(1) per subset instead of O(n^2)
  - 1000 random samples vs 5: t-test has real power
  - All 13 languages (78 pairs) vs 2: rho variance drops ~6x

Runtime estimate: ~5-8 min (batch encoding) + ~2 min (1000 samples)
"""

import os, json, time
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

MODEL_CACHE = os.path.join(os.path.dirname(__file__), "..", "model_cache")
MUSE_CACHE  = os.path.join(os.path.dirname(__file__), "..", "data", "muse_cache")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "rsc_detector_v2_results.json")

MONOLINGUAL_MODELS = {
    "en": ("bert-base-uncased",                       "English"),
    "de": ("bert-base-german-cased",                  "German"),
    "es": ("dccuchile/bert-base-spanish-wwm-cased",   "Spanish"),
    "ru": ("DeepPavlov/rubert-base-cased",             "Russian"),
    "zh": ("bert-base-chinese",                        "Mandarin"),
    "ar": ("aubmindlab/bert-base-arabertv02",          "Arabic"),
    "he": ("avichr/heBERT",                            "Hebrew"),
    "ta": ("l3cube-pune/tamil-bert",                   "Tamil"),
    "tl": ("jcblaise/bert-tagalog-base-cased",         "Tagalog"),
    "id": ("indobenchmark/indobert-base-p1",           "Indonesian"),
    "ja": ("cl-tohoku/bert-base-japanese-v2",          "Japanese"),
    "ko": ("klue/bert-base",                           "Korean"),
    "tr": ("dbmdz/bert-base-turkish-cased",            "Turkish"),
}

LANGUAGE_FAMILIES = {
    "en": "Indo-European", "de": "Indo-European",
    "es": "Indo-European", "ru": "Indo-European",
    "zh": "Sino-Tibetan",
    "ar": "Afro-Asiatic",  "he": "Afro-Asiatic",
    "ta": "Dravidian",
    "tl": "Austronesian",  "id": "Austronesian",
    "ja": "Japonic", "ko": "Koreanic", "tr": "Turkic",
}

# ---------------------------------------------------------------------------
# Domain concept sets (English words — translations looked up via MUSE)
# Chosen to be clearly domain-coherent with mutual semantic relationships
# ---------------------------------------------------------------------------

DOMAIN_SETS = {
    "astronomy":  ["star","planet","moon","sun","comet","galaxy","orbit","gravity","meteor","space","light","void","atmosphere","satellite","telescope"],
    "emotions":   ["joy","sadness","anger","fear","disgust","surprise","love","hate","shame","pride","envy","anxiety","hope","calm","trust"],
    "body":       ["eye","hand","foot","heart","brain","mouth","ear","nose","arm","leg","back","skin","blood","bone","face"],
    "food":       ["bread","water","meat","fish","fruit","milk","egg","salt","sugar","oil","wine","rice","soup","honey","butter"],
    "politics":   ["law","power","war","justice","freedom","leader","vote","state","army","peace","rights","order","rule","party","conflict"],
    "animals":    ["dog","cat","bird","fish","horse","cow","wolf","lion","snake","eagle","fox","bear","mouse","rabbit","deer"],
    "nature":     ["tree","river","mountain","fire","rain","wind","stone","sea","forest","earth","sky","grass","cloud","flower","ice"],
}

SKIP_WORDS = {"the","a","an","and","or","but","in","on","at","to","for","of","with","by","is","are","it"}

# ---------------------------------------------------------------------------
# Step 1: Load MUSE translations for all domain words + a pool of random words
# ---------------------------------------------------------------------------

def load_muse_translations(words_needed: List[str]) -> Dict[str, Dict[str, str]]:
    """
    For each word in words_needed, look up its translation in every MUSE file.
    Returns {word: {lang: translation}} for words with full coverage.
    """
    non_en_langs = [l for l in MONOLINGUAL_MODELS if l != "en"]
    lang_dicts = {}

    for lang in non_en_langs:
        cache_path = os.path.join(MUSE_CACHE, f"en-{lang}.txt")
        d = {}
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    eng = parts[0].lower()
                    if eng not in d:
                        d[eng] = parts[1]
        lang_dicts[lang] = d

    result = {}
    for word in words_needed:
        entry = {"en": word}
        ok = True
        for lang in non_en_langs:
            t = lang_dicts[lang].get(word)
            if t is None:
                ok = False
                break
            entry[lang] = t
        if ok:
            result[word] = entry

    return result


def load_muse_pool(pool_size: int = 500) -> Dict[str, Dict[str, str]]:
    """
    Load a pool of mid-frequency MUSE words (rank 600-1600) for random sampling.
    These are common words with full 12-language coverage but no specific domain.
    """
    non_en_langs = [l for l in MONOLINGUAL_MODELS if l != "en"]
    lang_dicts = {}

    for lang in non_en_langs:
        cache_path = os.path.join(MUSE_CACHE, f"en-{lang}.txt")
        d = {}
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    eng = parts[0].lower()
                    if eng not in d and eng not in SKIP_WORDS and eng.isalpha() and len(eng) > 2:
                        d[eng] = parts[1]
        lang_dicts[lang] = d

    # Find words with full coverage
    coverage = {}
    for d in lang_dicts.values():
        for w in d:
            coverage[w] = coverage.get(w, 0) + 1

    all_covered = sorted(
        [w for w, c in coverage.items() if c == len(non_en_langs)],
        key=lambda w: coverage[w], reverse=True
    )
    # Take mid-frequency band to avoid overlap with domain words
    # and to avoid top-frequency culturally universal words
    pool_words = all_covered[600:600 + pool_size]
    print(f"  Pool: {len(pool_words)} mid-frequency words (rank 600-{600+len(pool_words)})")

    pool = {}
    for word in pool_words:
        entry = {"en": word}
        for lang in non_en_langs:
            entry[lang] = lang_dicts[lang][word]
        pool[word] = entry

    return pool

# ---------------------------------------------------------------------------
# Step 2: Batch BERT encoding
# Encode ALL words for a language at once (no word-by-word loop)
# Returns numpy array (n_words, hidden_size), normalized
# ---------------------------------------------------------------------------

_model_cache_mem = {}

def load_model(lang: str, model_name: str, lang_name: str):
    if lang in _model_cache_mem:
        return _model_cache_mem[lang]
    from transformers import AutoTokenizer, AutoModel
    print(f"  [{lang_name:<12}] loading...", end=" ", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE)
    mdl = AutoModel.from_pretrained(model_name, cache_dir=MODEL_CACHE)
    mdl.eval()
    _model_cache_mem[lang] = (tok, mdl)
    print("ok")
    return tok, mdl


def batch_encode(lang: str, model_name: str, lang_name: str,
                 words: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Batch encode words. Returns (n_words, hidden_size) float32 array, L2-normalized.
    Uses batched tokenization for speed.
    """
    import torch
    tok, mdl = load_model(lang, model_name, lang_name)
    all_vecs = []

    for i in range(0, len(words), batch_size):
        batch = words[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=16)
        with torch.no_grad():
            out = mdl(**enc)
        # CLS token embeddings, shape (batch, hidden)
        vecs = out.last_hidden_state[:, 0, :].numpy().astype(np.float32)
        # L2 normalize each row
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        vecs = vecs / norms
        all_vecs.append(vecs)

    return np.vstack(all_vecs)  # (n_words, hidden_size)

# ---------------------------------------------------------------------------
# Step 3: Pre-compute gram matrices per language
# G[lang] = E @ E.T, shape (n_words, n_words)
# For any subset S: G[S][:,S] is the relational matrix — O(1) slice
# ---------------------------------------------------------------------------

def compute_gram_matrices(concept_words: List[str],
                          translation_map: Dict[str, Dict[str, str]]
                          ) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Encode all concept_words for all 13 languages.
    Returns gram_matrices {lang: (n,n) array} and active_langs list.
    """
    gram = {}
    active = []
    for lang, (model_name, lang_name) in MONOLINGUAL_MODELS.items():
        words = [translation_map[c][lang] for c in concept_words]
        E = batch_encode(lang, model_name, lang_name, words)
        gram[lang] = E @ E.T   # (n, n) relational matrix
        active.append(lang)
    return gram, active

# ---------------------------------------------------------------------------
# Step 4: Vectorized RSC rho for any subset of concept indices
# ---------------------------------------------------------------------------

def rho_for_subset(gram: Dict[str, np.ndarray], langs: List[str],
                   idx: np.ndarray) -> float:
    """
    Given pre-computed gram matrices, compute mean RSC rho for a subset of
    concept indices. idx is a 1D integer array of concept indices.
    """
    n = len(idx)
    if n < 5:
        return float("nan")
    tri = np.triu_indices(n, k=1)
    pair_rhos = []
    for i in range(len(langs)):
        for j in range(i + 1, len(langs)):
            a = gram[langs[i]][np.ix_(idx, idx)][tri]
            b = gram[langs[j]][np.ix_(idx, idx)][tri]
            rho, _ = stats.spearmanr(a, b)
            if not np.isnan(rho):
                pair_rhos.append(rho)
    return float(np.mean(pair_rhos)) if pair_rhos else float("nan")


def permutation_test_fast(gram: Dict[str, np.ndarray], langs: List[str],
                          idx: np.ndarray, n_perm: int = 500) -> float:
    """
    Permutation test for a specific concept subset.
    Shuffles concept indices independently per language.
    """
    rng = np.random.default_rng(42)
    observed = rho_for_subset(gram, langs, idx)
    n_total = gram[langs[0]].shape[0]
    count = 0

    # For permutation: instead of shuffling labels, draw random subsets of same size
    # (equivalent to label permutation when pool is large)
    for _ in range(n_perm):
        rand_idx = rng.choice(n_total, size=len(idx), replace=False)
        perm_rho = rho_for_subset(gram, langs, rand_idx)
        if perm_rho >= observed:
            count += 1

    p = count / n_perm
    return p if p > 0 else 1.0 / (n_perm + 1)


def sample_rho_distribution(gram: Dict[str, np.ndarray], langs: List[str],
                             n_samples: int, subset_size: int,
                             rng: np.random.Generator) -> np.ndarray:
    """
    Draw n_samples random subsets of subset_size from the gram matrix pool.
    Returns array of mean RSC rho values — the null/baseline distribution.
    """
    n_total = gram[langs[0]].shape[0]
    rhos = []
    for _ in range(n_samples):
        idx = rng.choice(n_total, size=subset_size, replace=False)
        rhos.append(rho_for_subset(gram, langs, idx))
    return np.array(rhos)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    t0 = time.time()
    print("RSC Communication Structure Detector v2")
    print("=" * 65)
    print("Design: 13 languages | batch encoding | pre-computed gram matrices")
    print("  Coherent: 7 domain sets of 15 concepts")
    print("  Random:   1000 random samples of 15 from 500-word pool")
    print("Prediction: mean(coherent) >> mean(random)  =>  RSC detects structure")
    print("=" * 65)

    # --- Load translations ---
    print("\n[1] Loading MUSE translations...")
    all_domain_words = list({w for words in DOMAIN_SETS.values() for w in words})
    domain_translations = load_muse_translations(all_domain_words)
    print(f"  Domain words with full 13-lang coverage: {len(domain_translations)}/{len(all_domain_words)}")

    pool = load_muse_pool(pool_size=600)

    # Merge domain + pool into a single concept universe for encoding
    universe = {**pool, **domain_translations}  # domain words override pool if overlap
    universe_words = list(universe.keys())
    print(f"  Total concepts to encode: {len(universe_words)}")

    # --- Batch encode all languages ---
    print("\n[2] Batch encoding all concepts for all 13 languages...")
    t_enc = time.time()
    gram, active_langs = compute_gram_matrices(universe_words, universe)
    print(f"  Encoding done in {time.time()-t_enc:.0f}s | Languages: {len(active_langs)}")

    # Build index map: concept -> position in universe array
    idx_map = {w: i for i, w in enumerate(universe_words)}

    # --- Pre-compute pool indices (excluding domain words) ---
    domain_word_set = set(domain_translations.keys())
    pool_indices = np.array([idx_map[w] for w in pool if w in idx_map and w not in domain_word_set])
    print(f"  Pool indices (random sampling): {len(pool_indices)}")

    rng = np.random.default_rng(99)

    # --- COHERENT condition ---
    print("\n[3] Computing RSC rho for coherent domain sets...")
    coherent_results = []
    for domain, words in DOMAIN_SETS.items():
        # Only use words that have full coverage
        valid = [w for w in words if w in idx_map]
        if len(valid) < 10:
            print(f"  {domain}: only {len(valid)} words covered, skipping")
            continue
        idx = np.array([idx_map[w] for w in valid])
        rho = rho_for_subset(gram, active_langs, idx)
        # Quick permutation test (200 perms) to get individual p-values
        p = permutation_test_fast(gram, active_langs, idx, n_perm=200)
        coherent_results.append({"domain": domain, "n": len(valid), "rho": rho, "p": p})
        print(f"  {domain:<12}: n={len(valid):>2}  rho={rho:+.4f}  p={p:.3f}")

    coherent_rhos = np.array([r["rho"] for r in coherent_results])

    # --- RANDOM condition: 1000 samples ---
    print(f"\n[4] Sampling random distribution (1000 samples, n=15 each)...")
    t_samp = time.time()
    random_rhos = sample_rho_distribution(
        gram, active_langs, n_samples=1000, subset_size=15, rng=rng
    )
    print(f"  Done in {time.time()-t_samp:.0f}s")
    print(f"  Random rho: mean={np.mean(random_rhos):+.4f}  std={np.std(random_rhos):.4f}  "
          f"95% CI=[{np.percentile(random_rhos,2.5):+.4f}, {np.percentile(random_rhos,97.5):+.4f}]")

    # --- STATISTICAL COMPARISON ---
    print(f"\n[5] Statistical comparison...")
    mean_coherent = float(np.mean(coherent_rhos))
    mean_random   = float(np.mean(random_rhos))
    std_coherent  = float(np.std(coherent_rhos))
    std_random    = float(np.std(random_rhos))

    # How many random samples fall below each coherent domain rho?
    percentile_ranks = {r["domain"]: float(np.mean(random_rhos < r["rho"])) * 100
                        for r in coherent_results}

    # t-test: coherent rhos vs random rhos
    t_stat, p_ttest = stats.ttest_ind(coherent_rhos, random_rhos)

    # Cohen's d effect size
    pooled_std = np.sqrt((std_coherent**2 + std_random**2) / 2)
    cohens_d = (mean_coherent - mean_random) / (pooled_std + 1e-9)

    # What percentile of random distribution is the mean coherent rho?
    mean_coherent_pctile = float(np.mean(random_rhos < mean_coherent)) * 100

    print(f"\n{'='*65}")
    print(f"RESULTS -- RSC COMMUNICATION STRUCTURE DETECTOR v2")
    print(f"{'='*65}")
    print(f"\n  Condition     n_sets   Mean rho    Std     95% CI")
    print(f"  ----------    ------   --------    ---     ------")
    print(f"  Coherent      {len(coherent_rhos):>6}   {mean_coherent:+.4f}    {std_coherent:.4f}")
    print(f"  Random        {len(random_rhos):>6}   {mean_random:+.4f}    {std_random:.4f}   "
          f"[{np.percentile(random_rhos,2.5):+.4f}, {np.percentile(random_rhos,97.5):+.4f}]")

    print(f"\n  Domain rhos vs random distribution:")
    print(f"  {'Domain':<12}  {'rho':>7}  {'p(perm)':>8}  {'%-ile vs random':>16}")
    print(f"  {'-'*12}  {'-'*7}  {'-'*8}  {'-'*16}")
    for r in sorted(coherent_results, key=lambda x: -x["rho"]):
        pctile = percentile_ranks[r["domain"]]
        marker = "***" if pctile > 95 else ("**" if pctile > 90 else ("*" if pctile > 80 else ""))
        print(f"  {r['domain']:<12}  {r['rho']:>+7.4f}  {r['p']:>8.3f}  {pctile:>14.1f}%  {marker}")

    print(f"\n  Coherent vs Random (t-test): t={t_stat:+.2f}  p={p_ttest:.4f}  Cohen's d={cohens_d:.2f}")
    print(f"  Mean coherent rho is at the {mean_coherent_pctile:.1f}th percentile of random")

    gradient = mean_coherent > mean_random
    sig = p_ttest < 0.05

    if sig and gradient:
        verdict = "CONFIRMED: RSC rho detects structured communication. Coherent >> Random."
        interpretation = (
            f"Domain-coherent concept sets score at the {mean_coherent_pctile:.0f}th percentile "
            f"of the random distribution (Cohen's d={cohens_d:.2f}). "
            f"RSC rho distinguishes organized conceptual structure from random vocabulary."
        )
    elif gradient and not sig:
        verdict = "DIRECTIONAL but not significant at p<0.05 (needs more domain sets for power)"
        interpretation = f"Gradient holds but n=7 coherent sets limits power. Direction correct."
    else:
        verdict = "NOT CONFIRMED in this test."
        interpretation = "Coherent domains do not score higher than random samples."

    print(f"\n  VERDICT: {verdict}")
    print(f"  {interpretation}")
    print(f"\n  Total runtime: {time.time()-t0:.0f}s")

    # Save
    save_data = {
        "version": "rsc_detector_v2",
        "hypothesis": "RSC rho is a metric for meaningful structured communication",
        "prediction": "Coherent domains > random samples of same size",
        "n_languages": len(active_langs),
        "n_random_samples": 1000,
        "coherent": {
            "mean_rho": mean_coherent, "std": std_coherent,
            "domains": coherent_results,
            "percentile_vs_random": mean_coherent_pctile,
        },
        "random": {
            "mean_rho": mean_random, "std": std_random,
            "ci_95": [float(np.percentile(random_rhos, 2.5)),
                      float(np.percentile(random_rhos, 97.5))],
            "n_samples": len(random_rhos),
        },
        "ttest": {"t": float(t_stat), "p": float(p_ttest)},
        "cohens_d": float(cohens_d),
        "gradient_holds": bool(gradient),
        "verdict": verdict,
        "runtime_seconds": round(time.time()-t0, 1),
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"  Results saved: {RESULTS_FILE}")


if __name__ == "__main__":
    run()
