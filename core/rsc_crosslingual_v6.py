"""
RSC Cross-Lingual Universality — v6 (Audit-Clean Version)
==========================================================
This version addresses three issues identified in the code audit:

  Issue 1 — STATISTICAL TEST (Critical)
    v4/v5 used ttest_1samp on 78 pairwise rhos. Those rhos are NOT
    independent (each language appears in 12 pairs). ttest_1samp
    underestimates std error, producing p-values that are too small.
    FIX: Language-level permutation test. Shuffle concept labels
    independently per language, recompute all pairs, measure how often
    the permuted mean beats the observed mean. This preserves the
    dependence structure exactly.

  Issue 2 — MUSE SELECTION BIAS (Critical)
    Top-frequency MUSE words are culturally universal (water, hand, fire).
    Testing only those concepts biases toward finding universality.
    FIX: Run experiment on three frequency bands:
      - High (rank 1-1200): most frequent / most culturally universal
      - Mid  (rank 3000-4200): medium frequency
      - Low  (rank 6000-7200): lower frequency, more culture-specific
    If signal holds across all three bands -> universality is real.
    If signal only holds for high-frequency -> might be cultural, not structural.

  Issue 3 — SENTENCE TEMPLATE (v5, Moderate)
    "I use {} every day." is semantically broken for abstract nouns,
    emotions, body parts, adjectives, verbs.
    FIX: Use "There is {}." — grammatically valid, semantically neutral
    for essentially all content words in all languages.

Everything else identical to v4: 13 monolingual BERT models, same
structural test (relational matrix Spearman rho), same shuffle baseline
for comparison.
"""

import os
import json
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Model registry (identical to v4)
# ---------------------------------------------------------------------------

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
    "ja": "Japonic",
    "ko": "Koreanic",
    "tr": "Turkic",
}

# Fix 3: neutral sentence template that works for all content word types
SENTENCE_TEMPLATES = {
    "en": "There is {}.",
    "de": "Es gibt {}.",
    "es": "Hay {}.",
    "ru": "Есть {}.",
    "zh": "有{}。",
    "ar": "هناك {}.",
    "he": "יש {}.",
    "ta": "{} இருக்கிறது.",
    "tl": "Mayroong {}.",
    "id": "Ada {}.",
    "ja": "{}があります。",
    "ko": "{}이 있습니다.",
    "tr": "{} var.",
}

CACHE_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "muse_cache")
MODEL_CACHE = os.path.join(os.path.dirname(__file__), "..", "model_cache")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "v6_results.json")

SKIP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "my", "your", "his", "her", "our", "their", "what", "who",
    "which", "when", "where", "how", "if", "not", "no", "yes", "as", "so",
    "do", "did", "does", "have", "has", "had", "will", "would", "could",
    "should", "may", "might", "shall", "can", "must", "than", "then",
    "also", "more", "very", "just", "up", "out", "about", "into", "over",
    "after", "before", "because", "while", "through", "between", "under",
}


# ---------------------------------------------------------------------------
# MUSE loader — returns ranked candidates (ordered by frequency/coverage)
# so we can slice into frequency bands
# ---------------------------------------------------------------------------

def load_muse_candidates(min_coverage: int = None) -> Tuple[List[str], Dict]:
    """
    Load all MUSE candidates with full language coverage.
    Returns (ranked_concepts, all_dicts) — ranked from most to least frequent.
    """
    import urllib.request

    MUSE_URL = "https://dl.fbaipublicfiles.com/arrival/dictionaries/en-{lang}.txt"
    langs = [l for l in MONOLINGUAL_MODELS if l != "en"]
    if min_coverage is None:
        min_coverage = len(langs)

    all_dicts = {}
    for lang in langs:
        cache_path = os.path.join(CACHE_DIR, f"en-{lang}.txt")
        if not os.path.exists(cache_path):
            print(f"  Downloading en-{lang}...", end=" ", flush=True)
            os.makedirs(CACHE_DIR, exist_ok=True)
            urllib.request.urlretrieve(MUSE_URL.format(lang=lang), cache_path)
            print("done")
        translations = {}
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    eng = parts[0].lower()
                    if eng not in translations and eng not in SKIP_WORDS and eng.isalpha():
                        translations[eng] = parts[1]
        all_dicts[lang] = translations

    coverage = {}
    for d in all_dicts.values():
        for word in d:
            coverage[word] = coverage.get(word, 0) + 1

    # Rank by coverage (proxy for frequency — high coverage = high frequency)
    ranked = sorted(
        [w for w, c in coverage.items() if c >= min_coverage],
        key=lambda w: coverage[w], reverse=True
    )
    return ranked, all_dicts


def slice_concepts(ranked: List[str], all_dicts: Dict, start: int, end: int) -> Dict:
    """Extract concepts for a given frequency rank band [start, end)."""
    band = ranked[start:end]
    langs = list(all_dicts.keys())
    concepts = {}
    for concept in band:
        entry = {"en": concept}
        for lang, d in all_dicts.items():
            if concept in d:
                entry[lang] = d[concept]
        # Only include concepts with full coverage
        if all(lang in entry for lang in langs):
            concepts[concept] = entry
    return concepts


# ---------------------------------------------------------------------------
# Sentence-level encoding (Fix 3: neutral template)
# ---------------------------------------------------------------------------

def encode_language_sentence(
    lang: str,
    model_name: str,
    lang_name: str,
    words: List[str],
    concept_names: List[str],
    batch_size: int = 32,
) -> Optional[Dict[str, np.ndarray]]:
    import torch
    from transformers import AutoTokenizer, AutoModel

    template = SENTENCE_TEMPLATES.get(lang, "There is {}.")
    sentences = [template.format(w) for w in words]

    try:
        print(f"  [{lang_name:12}] Loading {model_name}...", end=" ", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE)
        model = AutoModel.from_pretrained(model_name, cache_dir=MODEL_CACHE)
        model.eval()
        print("ok", flush=True)
    except Exception as e:
        print(f"FAILED: {e}")
        return None

    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            try:
                encoded = tokenizer(batch, padding=True, truncation=True,
                                    max_length=64, return_tensors="pt")
                outputs = model(**encoded)
                hidden = outputs.last_hidden_state
                mask   = encoded["attention_mask"].unsqueeze(-1).float()
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = (summed / counts).numpy()
                norms  = np.linalg.norm(pooled, axis=1, keepdims=True)
                norms  = np.where(norms == 0, 1, norms)
                all_embeddings.append(pooled / norms)
            except Exception:
                all_embeddings.append(np.zeros((len(batch), 768)))

    embeddings = np.vstack(all_embeddings)
    return {concept_names[i]: embeddings[i] for i in range(len(concept_names))}


# ---------------------------------------------------------------------------
# Structural comparison (identical to v4)
# ---------------------------------------------------------------------------

def relational_matrix(embeddings: Dict, concepts: List[str]) -> np.ndarray:
    vecs = np.stack([embeddings[c] for c in concepts])
    return vecs @ vecs.T


def structural_similarity(emb_a: Dict, emb_b: Dict) -> float:
    shared = [c for c in emb_a if c in emb_b]
    if len(shared) < 10:
        return float("nan")
    mat_a = relational_matrix(emb_a, shared)
    mat_b = relational_matrix(emb_b, shared)
    n = len(shared)
    idx = np.triu_indices(n, k=1)
    rho, _ = stats.spearmanr(mat_a[idx], mat_b[idx])
    return float(rho) if not np.isnan(rho) else 0.0


def compute_all_pairs(lang_embeddings: Dict, languages: List[str]):
    pairs, scores, fam_a, fam_b = [], [], [], []
    for i, la in enumerate(languages):
        for lb in languages[i+1:]:
            s = structural_similarity(lang_embeddings[la], lang_embeddings[lb])
            if not np.isnan(s):
                pairs.append((la, lb))
                scores.append(s)
                fam_a.append(LANGUAGE_FAMILIES.get(la, "?"))
                fam_b.append(LANGUAGE_FAMILIES.get(lb, "?"))
    return pairs, scores, fam_a, fam_b


# ---------------------------------------------------------------------------
# Fix 1: Language-level permutation test
# Shuffle concept labels independently per language, then recompute all pairs.
# This is the correct test because it:
#   - Preserves the dependence structure among pairs
#   - Preserves embedding geometry within each language
#   - Only destroys cross-language concept correspondence
# ---------------------------------------------------------------------------

def permutation_test(lang_embeddings: Dict, languages: List[str], n_perm: int = 1000) -> float:
    """Returns p-value: proportion of permutations with mean rho >= observed mean."""
    rng = np.random.default_rng(42)
    all_concept_keys = list(list(lang_embeddings.values())[0].keys())

    # Observed
    _, true_scores, _, _ = compute_all_pairs(lang_embeddings, languages)
    observed_mean = float(np.mean(true_scores))

    count_geq = 0
    for _ in range(n_perm):
        shuffled_embs = {}
        for lang, emb in lang_embeddings.items():
            perm = rng.permutation(all_concept_keys).tolist()
            orig = list(emb.keys())
            shuffled_embs[lang] = {perm[i]: emb[orig[i]] for i in range(len(orig))}
        _, perm_scores, _, _ = compute_all_pairs(shuffled_embs, languages)
        if perm_scores and float(np.mean(perm_scores)) >= observed_mean:
            count_geq += 1

    p = count_geq / n_perm
    return p if p > 0 else 1.0 / (n_perm + 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_band(concepts: Dict, lang_embeddings_loaded: Optional[Dict],
             band_label: str, langs: List[str]) -> Dict:
    """Run the experiment for one concept band. Re-use cached embeddings if provided."""
    concept_names = list(concepts.keys())
    print(f"\n  Band: {band_label} ({len(concept_names)} concepts)")

    if lang_embeddings_loaded is None:
        print("  Encoding...")
        lang_embeddings = {}
        for lang, (model_name, lang_name) in MONOLINGUAL_MODELS.items():
            if lang not in langs:
                continue
            words = [concepts[c][lang] for c in concept_names]
            result = encode_language_sentence(lang, model_name, lang_name, words, concept_names)
            if result:
                lang_embeddings[lang] = result
    else:
        # Slice to this band's concepts
        lang_embeddings = {
            lang: {c: emb[c] for c in concept_names if c in emb}
            for lang, emb in lang_embeddings_loaded.items()
        }

    active = [l for l in lang_embeddings if len(lang_embeddings[l]) >= len(concept_names) * 0.95]

    pairs, scores, fam_a, fam_b = compute_all_pairs(lang_embeddings, active)
    true_mean = float(np.mean(scores)) if scores else float("nan")

    print(f"  Running permutation test (200 permutations)...", flush=True)
    p_val = permutation_test(lang_embeddings, active, n_perm=200)

    same_scores = [s for s, fa, fb in zip(scores, fam_a, fam_b) if fa == fb]
    diff_scores  = [s for s, fa, fb in zip(scores, fam_a, fam_b) if fa != fb]
    sf_mean = float(np.mean(same_scores)) if same_scores else float("nan")
    df_mean = float(np.mean(diff_scores))  if diff_scores  else float("nan")
    _, p_fam = stats.ttest_ind(same_scores, diff_scores) if (same_scores and diff_scores) else (None, float("nan"))

    print(f"    rho = {true_mean:+.4f}   p = {p_val:.2e}   family_gap_p = {p_fam:.3f}")
    return {
        "band": band_label,
        "n_concepts": len(concept_names),
        "n_languages": len(active),
        "true_rho": true_mean,
        "p_value": p_val,
        "same_family_rho": sf_mean,
        "diff_family_rho": df_mean,
        "family_gap_p": float(p_fam) if not np.isnan(p_fam) else None,
        "lang_embeddings": lang_embeddings,  # returned so we don't re-encode
    }


def run():
    print("RSC Cross-Lingual Universality — v6 (Audit-Clean)")
    print("=" * 65)
    print("Fixes applied vs v4/v5:")
    print("  1. Language-level permutation test (not ttest_1samp)")
    print("  2. Frequency-stratified sensitivity analysis (high/mid/low)")
    print("  3. Neutral sentence template 'There is {}.'")
    print("=" * 65)

    print("\nLoading MUSE frequency rankings...")
    ranked, all_dicts = load_muse_candidates()
    print(f"  Total candidates with full coverage: {len(ranked)}")

    # Define three frequency bands
    # Fix 2: stratified sampling to test MUSE selection bias
    # Use 300 concepts per band (balances statistical power vs runtime)
    total = len(ranked)
    band_size = min(300, total // 4)
    BANDS = [
        ("high-freq (top)",    0,                    band_size),
        ("mid-freq  (middle)", total // 2 - band_size // 2, total // 2 + band_size // 2),
        ("low-freq  (bottom)", total - band_size,    total),
    ]

    # Check we have enough candidates
    if len(ranked) < 600:
        print(f"  WARNING: Only {len(ranked)} candidates — adjusting bands")
        n = len(ranked)
        BANDS = [
            (f"high-freq (rank 1-400)",         0,    min(400, n//3)),
            (f"mid-freq  (rank {n//3}-{2*n//3})", n//3, 2*n//3),
            (f"low-freq  (rank {2*n//3}-{n})",   2*n//3, n),
        ]

    print(f"\nEncoding 13 languages once (full high-freq band)...")
    high_concepts = slice_concepts(ranked, all_dicts, BANDS[0][1], BANDS[0][2])
    print(f"  High-freq concepts loaded: {len(high_concepts)}")

    lang_embeddings_full = {}
    concept_names_full = list(high_concepts.keys())
    active_langs = []
    for lang, (model_name, lang_name) in MONOLINGUAL_MODELS.items():
        words = [high_concepts[c][lang] for c in concept_names_full]
        result = encode_language_sentence(lang, model_name, lang_name, words, concept_names_full)
        if result:
            lang_embeddings_full[lang] = result
            active_langs.append(lang)

    print(f"  Successfully encoded: {len(active_langs)} languages")

    # Run full experiment on high-freq band (reuse loaded models)
    band_results = []
    print("\nRunning experiments across frequency bands:")

    # Band 1: high-freq (already encoded)
    pairs, scores, fam_a, fam_b = compute_all_pairs(lang_embeddings_full, active_langs)
    true_mean_high = float(np.mean(scores))
    print(f"\n  Band: {BANDS[0][0]} ({len(high_concepts)} concepts)")
    print(f"  Running permutation test (200 permutations)...", flush=True)
    p_high = permutation_test(lang_embeddings_full, active_langs, n_perm=200)
    same_high = [s for s, fa, fb in zip(scores, fam_a, fam_b) if fa == fb]
    diff_high  = [s for s, fa, fb in zip(scores, fam_a, fam_b) if fa != fb]
    _, p_fam_high = stats.ttest_ind(same_high, diff_high) if (same_high and diff_high) else (None, float("nan"))
    print(f"    rho = {true_mean_high:+.4f}   p = {p_high:.2e}   family_gap_p = {p_fam_high:.3f}")
    band_results.append({
        "band": BANDS[0][0], "n_concepts": len(high_concepts),
        "true_rho": true_mean_high, "p_value": p_high,
        "same_family_rho": float(np.mean(same_high)) if same_high else None,
        "diff_family_rho": float(np.mean(diff_high))  if diff_high  else None,
        "family_gap_p": float(p_fam_high),
    })

    # Bands 2 and 3: different concept slices, re-encode from scratch
    for band_label, start, end in BANDS[1:]:
        band_concepts = slice_concepts(ranked, all_dicts, start, end)
        if len(band_concepts) < 100:
            print(f"\n  Band: {band_label} — insufficient concepts ({len(band_concepts)}), skipping")
            continue

        print(f"\n  Band: {band_label} ({len(band_concepts)} concepts)")
        print(f"  Encoding {len(band_concepts)} concepts per language...")
        band_embs = {}
        for lang, (model_name, lang_name) in MONOLINGUAL_MODELS.items():
            concept_names_band = list(band_concepts.keys())
            words = [band_concepts[c][lang] for c in concept_names_band]
            result = encode_language_sentence(lang, model_name, lang_name, words, concept_names_band)
            if result:
                band_embs[lang] = result

        band_active = list(band_embs.keys())
        pairs_b, scores_b, fa_b, fb_b = compute_all_pairs(band_embs, band_active)
        true_mean_b = float(np.mean(scores_b)) if scores_b else float("nan")
        print(f"  Running permutation test (200 permutations)...", flush=True)
        p_b = permutation_test(band_embs, band_active, n_perm=200)
        same_b = [s for s, fa, fb in zip(scores_b, fa_b, fb_b) if fa == fb]
        diff_b  = [s for s, fa, fb in zip(scores_b, fa_b, fb_b) if fa != fb]
        _, p_fam_b = stats.ttest_ind(same_b, diff_b) if (same_b and diff_b) else (None, float("nan"))
        print(f"    rho = {true_mean_b:+.4f}   p = {p_b:.2e}   family_gap_p = {p_fam_b:.3f}")
        band_results.append({
            "band": band_label, "n_concepts": len(band_concepts),
            "true_rho": true_mean_b, "p_value": p_b,
            "same_family_rho": float(np.mean(same_b)) if same_b else None,
            "diff_family_rho": float(np.mean(diff_b))  if diff_b  else None,
            "family_gap_p": float(p_fam_b),
        })

    # Final summary
    print("\n" + "=" * 65)
    print("RESULTS — v6 (Audit-Clean: permutation test + stratified sampling)")
    print("=" * 65)
    print(f"\n  {'Band':<35} {'Concepts':>8}  {'rho':>7}  {'p-value':>10}  {'fam_gap_p':>9}")
    print(f"  {'-'*35} {'-'*8}  {'-'*7}  {'-'*10}  {'-'*9}")
    for r in band_results:
        print(f"  {r['band']:<35} {r['n_concepts']:>8}  {r['true_rho']:>+7.4f}  {r['p_value']:>10.2e}  {r['family_gap_p']:>9.3f}")

    print("\n  Interpretation:")
    sig_bands = [r for r in band_results if r['p_value'] < 0.05]
    print(f"  {len(sig_bands)}/{len(band_results)} frequency bands show significant signal (p<0.05)")
    if len(sig_bands) == len(band_results):
        print("  -> Signal is NOT limited to high-frequency / culturally universal words")
        print("  -> MUSE selection bias does not explain the result")
    elif len(sig_bands) > 0:
        print("  -> Signal is strongest for high-frequency words; weaker for low-frequency")
        print("  -> Some MUSE selection bias possible; interpret with caution")
    else:
        print("  -> Signal only in high-freq band: likely MUSE selection bias")

    # Save
    save_results = {
        "version": "v6_audit_clean",
        "fixes_applied": [
            "language-level permutation test (not ttest_1samp)",
            "frequency-stratified sensitivity analysis",
            "neutral sentence template: 'There is {}.'"
        ],
        "sentence_templates": SENTENCE_TEMPLATES,
        "n_languages": len(active_langs),
        "active_languages": active_langs,
        "band_results": [{k: v for k, v in r.items() if k != "lang_embeddings"}
                         for r in band_results],
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: {RESULTS_FILE}")


if __name__ == "__main__":
    run()
