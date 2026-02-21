"""
RSC Cross-Lingual Universality — v3 (1000+ concepts, automated)
================================================================
Automatically downloads MUSE bilingual dictionaries (Facebook Research)
for all 13 languages, finds 1000+ concepts with cross-language coverage,
encodes with LaBSE, and runs the full structural convergence test.

No manual data entry. No hardcoded translations.
Runs fully automated from a clean machine.

Data source: MUSE (Multilingual Unsupervised and Supervised Embeddings)
  - Lample et al., 2018: https://arxiv.org/abs/1710.04087
  - Facebook Research, open source (CC BY-NC 4.0)
  - Bilingual word dictionaries for 45 languages
  - URL: https://dl.fbaipublicfiles.com/arrival/dictionaries/en-{lang}.txt

Language families (8 genuinely unrelated):
  Indo-European: English, German, Spanish, Russian
  Sino-Tibetan:  Mandarin Chinese
  Afro-Asiatic:  Arabic, Hebrew
  Dravidian:     Tamil
  Austronesian:  Tagalog, Indonesian
  Japonic:       Japanese
  Koreanic:      Korean
  Turkic:        Turkish

Circularity controls:
  - Structural test (relational matrices), not raw embedding similarity
  - Shuffle baseline destroys concept correspondence
  - LaBSE partial circularity acknowledged; controlled via shuffle
"""

import os
import json
import time
import urllib.request
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "muse_cache")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "v3_results.json")

LANGUAGE_CODES = {
    "de": "German",      "es": "Spanish",    "ru": "Russian",
    "zh": "Mandarin",    "ar": "Arabic",     "he": "Hebrew",
    "ta": "Tamil",       "tl": "Tagalog",    "id": "Indonesian",
    "ja": "Japanese",    "ko": "Korean",     "tr": "Turkish",
}

LANGUAGE_FAMILIES = {
    "en": "Indo-European", "de": "Indo-European", "es": "Indo-European", "ru": "Indo-European",
    "zh": "Sino-Tibetan",
    "ar": "Afro-Asiatic",  "he": "Afro-Asiatic",
    "ta": "Dravidian",
    "tl": "Austronesian",  "id": "Austronesian",
    "ja": "Japonic",
    "ko": "Koreanic",
    "tr": "Turkic",
}

MUSE_URL = "https://dl.fbaipublicfiles.com/arrival/dictionaries/en-{lang}.txt"

# Words to skip — function words, articles, pronouns (not concept-bearing)
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
# Step 1: Download / load MUSE dictionaries
# ---------------------------------------------------------------------------

def load_muse_dict(lang: str, max_entries: int = 50000) -> Dict[str, str]:
    """
    Download (or load cached) MUSE en-{lang} bilingual dictionary.
    Returns {english_word: foreign_translation} for unique English words.
    First translation per English word is used (highest-frequency match).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"en-{lang}.txt")

    if not os.path.exists(cache_path):
        url = MUSE_URL.format(lang=lang)
        print(f"  Downloading en-{lang} from MUSE...", end=" ", flush=True)
        urllib.request.urlretrieve(url, cache_path)
        print("done")
    else:
        print(f"  en-{lang}: loaded from cache")

    translations = {}
    with open(cache_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_entries:
                break
            parts = line.strip().split()
            if len(parts) >= 2:
                eng = parts[0].lower()
                foreign = parts[1]
                # Only take first translation per English word
                if eng not in translations and eng not in SKIP_WORDS and eng.isalpha():
                    translations[eng] = foreign

    return translations


# ---------------------------------------------------------------------------
# Step 2: Find concepts with broad language coverage
# ---------------------------------------------------------------------------

def find_covered_concepts(
    all_dicts: Dict[str, Dict[str, str]],
    min_coverage: int = 10,
    target_count: int = 1200,
) -> Dict[str, Dict[str, str]]:
    """
    Find English concepts with translations in at least min_coverage languages.
    Returns {concept: {lang: word}} for the top target_count concepts.

    Concepts are ordered by frequency (MUSE dictionaries are frequency-sorted),
    so highest-frequency concepts come first — these are the most common,
    well-established words in each language.
    """
    # Count coverage per English word
    coverage: Dict[str, int] = {}
    for lang, d in all_dicts.items():
        for word in d:
            coverage[word] = coverage.get(word, 0) + 1

    # Filter to min_coverage and sort by coverage (most covered first)
    candidates = [w for w, c in coverage.items() if c >= min_coverage]
    candidates.sort(key=lambda w: coverage[w], reverse=True)

    print(f"\n  Concepts with coverage >= {min_coverage}/12 languages: {len(candidates)}")

    # Build concept table
    result = {}
    for concept in candidates[:target_count]:
        entry = {"en": concept}
        for lang, d in all_dicts.items():
            if concept in d:
                entry[lang] = d[concept]
        result[concept] = entry

    print(f"  Using top {len(result)} concepts")
    return result


# ---------------------------------------------------------------------------
# Step 3: Encode with LaBSE
# ---------------------------------------------------------------------------

def encode_concepts_labse(
    concepts: Dict[str, Dict[str, str]],
    languages: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    For each language, encode all concepts using LaBSE.
    Returns {lang: {concept: embedding_vector}}
    """
    from sentence_transformers import SentenceTransformer

    print("\nLoading LaBSE (cached)...", end=" ", flush=True)
    model = SentenceTransformer("LaBSE", cache_folder="./model_cache")
    print("ok")

    lang_embeddings: Dict[str, Dict[str, np.ndarray]] = {}

    for lang in languages:
        lang_name = LANGUAGE_CODES.get(lang, lang) if lang != "en" else "English"
        words = []
        concept_list = []
        for concept, translations in concepts.items():
            if lang in translations:
                words.append(translations[lang])
                concept_list.append(concept)

        if not words:
            continue

        print(f"  Encoding {lang_name}: {len(words)} words...", end=" ", flush=True)
        embeddings = model.encode(words, show_progress_bar=False, normalize_embeddings=True,
                                  batch_size=256)
        lang_embeddings[lang] = {concept_list[i]: embeddings[i] for i in range(len(concept_list))}
        print("done")

    return lang_embeddings


# ---------------------------------------------------------------------------
# Step 4: Structural comparison
# ---------------------------------------------------------------------------

def relational_matrix(embeddings: Dict[str, np.ndarray], concepts: List[str]) -> np.ndarray:
    """Pairwise cosine similarity matrix within one language (normalized embeddings -> dot product)."""
    vecs = np.stack([embeddings[c] for c in concepts])
    return vecs @ vecs.T


def structural_similarity(emb_a: Dict, emb_b: Dict) -> float:
    """Spearman rho on upper triangle of relational matrices."""
    shared = [c for c in emb_a if c in emb_b]
    if len(shared) < 10:
        return float("nan")
    mat_a = relational_matrix(emb_a, shared)
    mat_b = relational_matrix(emb_b, shared)
    n = len(shared)
    idx = np.triu_indices(n, k=1)
    rho, _ = stats.spearmanr(mat_a[idx], mat_b[idx])
    return float(rho) if not np.isnan(rho) else 0.0


def shuffle_baseline(lang_embeddings: Dict, languages: List[str], n: int = 300) -> float:
    """Shuffle concept labels, measure residual correlation."""
    rng = np.random.default_rng(42)
    all_concepts = list(list(lang_embeddings.values())[0].keys())
    scores = []
    for _ in range(n):
        la, lb = rng.choice(languages, 2, replace=False)
        shuffled = rng.permutation(all_concepts).tolist()
        emb_shuffled = {shuffled[i]: lang_embeddings[la][all_concepts[i]]
                        for i in range(len(all_concepts))
                        if all_concepts[i] in lang_embeddings[la]}
        scores.append(structural_similarity(emb_shuffled, lang_embeddings[lb]))
    return float(np.nanmean(scores))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run():
    print("=" * 65)
    print("RSC Cross-Lingual Universality — v3 (1000+ concepts, MUSE)")
    print("=" * 65)

    # Step 1: Download MUSE dicts
    print("\nStep 1: Loading MUSE bilingual dictionaries...")
    all_dicts = {}
    for lang in LANGUAGE_CODES:
        all_dicts[lang] = load_muse_dict(lang)
        time.sleep(0.1)  # polite delay

    # Step 2: Find covered concepts
    print("\nStep 2: Finding concepts with broad language coverage...")
    concepts = find_covered_concepts(all_dicts, min_coverage=10, target_count=1200)

    n_concepts = len(concepts)
    lang_coverage = {lang: sum(1 for c in concepts.values() if lang in c)
                     for lang in list(LANGUAGE_CODES.keys()) + ["en"]}
    print("\n  Coverage per language:")
    for lang, count in sorted(lang_coverage.items(), key=lambda x: -x[1]):
        name = LANGUAGE_CODES.get(lang, "English")
        bar = "#" * (count // 30)
        print(f"    {name:12} ({lang}): {count:4d}  {bar}")

    # Step 3: Encode
    all_languages = ["en"] + list(LANGUAGE_CODES.keys())
    print("\nStep 3: Encoding with LaBSE...")
    lang_embeddings = encode_concepts_labse(concepts, all_languages)

    active_languages = list(lang_embeddings.keys())
    print(f"\n  Active languages: {len(active_languages)}")

    # Step 4: Pairwise structural similarity
    print("\nStep 4: Computing pairwise structural convergence...")
    all_pairs = []
    same_family_scores = []
    diff_family_scores = []

    for i, lang_a in enumerate(active_languages):
        for lang_b in active_languages[i+1:]:
            rho = structural_similarity(lang_embeddings[lang_a], lang_embeddings[lang_b])
            if np.isnan(rho):
                continue
            same = LANGUAGE_FAMILIES.get(lang_a) == LANGUAGE_FAMILIES.get(lang_b)
            all_pairs.append((lang_a, lang_b, rho, same))
            (same_family_scores if same else diff_family_scores).append(rho)

    # Step 5: Shuffle baseline
    print("\nStep 5: Computing shuffle baseline (300 shuffles)...")
    baseline_mean = shuffle_baseline(lang_embeddings, active_languages, n=300)

    # Stats
    true_mean = float(np.mean([s for _, _, s, _ in all_pairs]))
    t_base, p_base = stats.ttest_1samp([s for _, _, s, _ in all_pairs], baseline_mean)
    sf_mean = float(np.mean(same_family_scores)) if same_family_scores else 0.0
    df_mean = float(np.mean(diff_family_scores)) if diff_family_scores else 0.0
    t_fam, p_fam = stats.ttest_ind(same_family_scores, diff_family_scores)

    # Results
    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"\n  Concepts tested:           {n_concepts}")
    print(f"  Languages:                 {len(active_languages)}")
    print(f"  Language families:         {len(set(LANGUAGE_FAMILIES[l] for l in active_languages if l in LANGUAGE_FAMILIES))}")
    print(f"\n  True structural rho:       {true_mean:+.4f}")
    print(f"  Shuffle baseline rho:      {baseline_mean:+.4f}")
    print(f"  Lift:                      {true_mean - baseline_mean:+.4f}")
    print(f"  p vs baseline:             {p_base:.2e}")
    print(f"\n  Same-family mean rho:      {sf_mean:+.4f}  (n={len(same_family_scores)})")
    print(f"  Diff-family mean rho:      {df_mean:+.4f}  (n={len(diff_family_scores)})")
    print(f"  Family gap p-value:        {p_fam:.4f}")

    if p_fam > 0.05:
        print("  -> Family gap NOT significant: signal is universal, not ancestry")
    else:
        print("  -> Family gap significant: ancestry partially explains signal")

    print("\n  Top 20 cross-family pairs:")
    cross_family = [(a, b, r) for a, b, r, same in all_pairs if not same]
    cross_family.sort(key=lambda x: x[2], reverse=True)
    for la, lb, rho in cross_family[:20]:
        na = LANGUAGE_CODES.get(la, "English")
        nb = LANGUAGE_CODES.get(lb, "English")
        fa = LANGUAGE_FAMILIES.get(la, "?")
        fb = LANGUAGE_FAMILIES.get(lb, "?")
        print(f"    {na:12} <-> {nb:12}  rho={rho:+.3f}  [{fa} <-> {fb}]")

    # Save results
    results = {
        "n_concepts": n_concepts,
        "n_languages": len(active_languages),
        "true_mean_rho": true_mean,
        "baseline_rho": baseline_mean,
        "lift": true_mean - baseline_mean,
        "p_vs_baseline": float(p_base),
        "same_family_rho": sf_mean,
        "diff_family_rho": df_mean,
        "p_family_gap": float(p_fam),
        "signal_detected": bool(true_mean > baseline_mean and p_base < 0.05),
        "universality_confirmed": bool(p_fam > 0.05),
        "top_cross_family": [(LANGUAGE_CODES.get(a, a), LANGUAGE_CODES.get(b, b), round(r, 4))
                             for a, b, r in cross_family[:20]],
    }
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {RESULTS_FILE}")
    print("=" * 65)

    return results


if __name__ == "__main__":
    run()
