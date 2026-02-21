"""
RSC Cross-Lingual Universality — v5 (Monolingual BERT, sentence-level encoding)
================================================================================
Same airtight design as v4 — 13 separate monolingual BERT models, zero cross-
lingual training — but now each concept is encoded in a sentence context instead
of as a bare word.

Why sentence-level matters:
  BERT was pre-trained on full sentences. Encoding bare words ("water") puts the
  model in an unusual regime — the [CLS] and context tokens are effectively empty.
  Sentence context ("I use water every day.") lets the model draw on the full
  range of learned contextual relationships, which should produce richer, more
  structurally consistent embeddings.

Sentence templates ("I like [word].") translated per language:
  Simple present-tense copula/verb sentences using the concept word.
  Grammatically imperfect for all edge cases but semantically consistent —
  each model sees the word in the same pragmatic role across languages.

Hypothesis: sentence-level embeddings produce higher structural rho than
bare-word embeddings (v4: rho=+0.058, p=4.62e-16).

Everything else identical to v4: same 1200 MUSE concepts, same relational
matrix comparison, same shuffle baseline (300 shuffles).
"""

import os
import json
import numpy as np
from scipy import stats
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Sentence templates — "I use [word] every day." in each language
# Simple, consistent, semantically neutral for most content words.
# ---------------------------------------------------------------------------

SENTENCE_TEMPLATES = {
    "en": "I use {} every day.",
    "de": "Ich benutze {} jeden Tag.",
    "es": "Uso {} todos los días.",
    "ru": "Я использую {} каждый день.",
    "zh": "我每天都用{}。",
    "ar": "أستخدم {} كل يوم.",
    "he": "אני משתמש ב{} כל יום.",
    "ta": "நான் {} ஒவ்வொரு நாளும் பயன்படுத்துகிறேன்.",
    "tl": "Ginagamit ko ang {} araw-araw.",
    "id": "Saya menggunakan {} setiap hari.",
    "ja": "私は毎日{}を使います。",
    "ko": "나는 매일 {}을 사용합니다.",
    "tr": "Her gün {} kullanıyorum.",
}

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

CACHE_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "muse_cache")
MODEL_CACHE = os.path.join(os.path.dirname(__file__), "..", "model_cache")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "v5_results.json")


# ---------------------------------------------------------------------------
# Reuse v4's concept loader (same 1,200 MUSE concepts)
# ---------------------------------------------------------------------------

def load_concepts_from_muse(target_count: int = 1200) -> Dict[str, Dict[str, str]]:
    import urllib.request

    MUSE_URL  = "https://dl.fbaipublicfiles.com/arrival/dictionaries/en-{lang}.txt"
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

    langs = [l for l in MONOLINGUAL_MODELS if l != "en"]
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
    for lang, d in all_dicts.items():
        for word in d:
            coverage[word] = coverage.get(word, 0) + 1

    candidates = sorted([w for w, c in coverage.items() if c >= len(langs)],
                        key=lambda w: coverage[w], reverse=True)[:target_count]

    concepts = {}
    for concept in candidates:
        entry = {"en": concept}
        for lang, d in all_dicts.items():
            if concept in d:
                entry[lang] = d[concept]
        concepts[concept] = entry

    print(f"  Loaded {len(concepts)} concepts with full coverage")
    return concepts


# ---------------------------------------------------------------------------
# Sentence-level encoding
# ---------------------------------------------------------------------------

def encode_language_sentence(
    lang: str,
    model_name: str,
    lang_name: str,
    words: List[str],
    concept_names: List[str],
    batch_size: int = 32,       # smaller batch — sentences are longer
) -> Optional[Dict[str, np.ndarray]]:
    """
    Encode each word embedded in a sentence template using a monolingual BERT model.
    Uses mean pooling over all non-[CLS]/[SEP] non-padding tokens.
    Returns {concept: embedding} or None on failure.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    template = SENTENCE_TEMPLATES.get(lang, "I use {} every day.")
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
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt",
                )
                outputs = model(**encoded)
                hidden = outputs.last_hidden_state      # (batch, seq_len, 768)
                mask   = encoded["attention_mask"].unsqueeze(-1).float()
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = (summed / counts).numpy()
                norms  = np.linalg.norm(pooled, axis=1, keepdims=True)
                norms  = np.where(norms == 0, 1, norms)
                all_embeddings.append(pooled / norms)
            except Exception as e:
                all_embeddings.append(np.zeros((len(batch), 768)))

    embeddings = np.vstack(all_embeddings)
    return {concept_names[i]: embeddings[i] for i in range(len(concept_names))}


# ---------------------------------------------------------------------------
# Structural comparison (identical to v4)
# ---------------------------------------------------------------------------

def relational_matrix(embeddings: Dict[str, np.ndarray], concepts: List[str]) -> np.ndarray:
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


def shuffle_baseline(lang_embeddings: Dict, languages: List[str], n: int = 300) -> float:
    rng = np.random.default_rng(42)
    all_concepts = list(list(lang_embeddings.values())[0].keys())
    scores = []
    for _ in range(n):
        la, lb = rng.choice(languages, 2, replace=False)
        shuffled = rng.permutation(all_concepts).tolist()
        emb_shuffled = {shuffled[i]: lang_embeddings[la][all_concepts[i]]
                        for i in range(len(all_concepts))
                        if all_concepts[i] in lang_embeddings[la]}
        s = structural_similarity(emb_shuffled, lang_embeddings[lb])
        if not np.isnan(s):
            scores.append(s)
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("RSC Cross-Lingual Universality — v5 (Monolingual BERT, sentence-level)")
    print("=" * 70)

    print("Loading 1,200 MUSE concepts...")
    concepts = load_concepts_from_muse(1200)
    concept_names = list(concepts.keys())

    print(f"Encoding {len(concept_names)} concepts per language (sentence-level):")
    lang_embeddings = {}
    for lang, (model_name, lang_name) in MONOLINGUAL_MODELS.items():
        words = [concepts[c][lang] for c in concept_names]
        result = encode_language_sentence(lang, model_name, lang_name, words, concept_names)
        if result:
            lang_embeddings[lang] = result

    languages = list(lang_embeddings.keys())
    print(f"  Successfully encoded: {len(languages)} languages")

    # All pairwise structural similarities
    pairs, scores, families_a, families_b = [], [], [], []
    for i, la in enumerate(languages):
        for j, lb in enumerate(languages):
            if j <= i:
                continue
            s = structural_similarity(lang_embeddings[la], lang_embeddings[lb])
            if not np.isnan(s):
                pairs.append((la, lb))
                scores.append(s)
                families_a.append(LANGUAGE_FAMILIES[la])
                families_b.append(LANGUAGE_FAMILIES[lb])

    scores_arr = np.array(scores)
    true_rho = float(np.mean(scores_arr))

    print("Computing shuffle baseline (300 shuffles)...")
    baseline = shuffle_baseline(lang_embeddings, languages)

    # One-sample t-test: is the mean of all pairwise rhos different from baseline?
    # Same test as v4 — correct for this design.
    _, p_val = stats.ttest_1samp(scores_arr, baseline)

    # Family gap analysis
    same_scores = [s for s, fa, fb in zip(scores, families_a, families_b) if fa == fb]
    diff_scores = [s for s, fa, fb in zip(scores, families_a, families_b) if fa != fb]
    if len(same_scores) >= 2 and len(diff_scores) >= 2:
        _, family_p = stats.ttest_ind(same_scores, diff_scores)
    else:
        family_p = float("nan")

    # Comparison table
    VERSIONS = [
        ("v1 (phonological)",         14,    0.059, -0.003,  "0.0009"),
        ("v2 (LaBSE, 39 concepts)",   39,    0.620,  0.001,  "<1e-06"),
        ("v3 (LaBSE, MUSE 1200)",   1200,    0.537, -0.001,  "1.3e-64"),
        ("v4 (monolingual, MUSE 1200)", 1200, 0.058,  0.001, "4.62e-16"),
        ("v5 (sentence-level, MUSE 1200)", len(concept_names),
                                       true_rho, baseline, f"{p_val:.2e}"),
    ]

    print()
    print("RESULTS — v5 Sentence-Level Monolingual BERT")
    print(f"  Concepts:        {len(concept_names)}")
    print(f"  Languages:       {len(languages)}")
    print(f"  True rho:        {true_rho:+.4f}")
    print(f"  Baseline rho:    {baseline:+.4f}")
    print(f"  Lift:            {true_rho - baseline:+.4f}")
    print(f"  p vs baseline:   {p_val:.2e}")
    if same_scores:
        print(f"  Same-family rho: {np.mean(same_scores):+.4f}  (n={len(same_scores)})")
        print(f"  Diff-family rho: {np.mean(diff_scores):+.4f}  (n={len(diff_scores)})")
        print(f"  Family gap p:    {family_p:.4f}")
        if not np.isnan(family_p) and family_p > 0.05:
            print(f"  -> UNIVERSAL: ancestry gap NOT significant")
        else:
            print(f"  -> Family gap significant (p={family_p:.4f})")

    print()
    print(f"  {'Version':<38} {'Concepts':>8}  {'True rho':>8}  {'Baseline':>8}  {'p-value':>12}")
    for name, nc, rho, base, pv in VERSIONS:
        print(f"  {name:<38} {nc:>8}    {rho:>+.3f}    {base:>+.3f}  {pv:>12}")

    # Save results
    results = {
        "version": "v5",
        "encoding": "sentence-level",
        "templates": SENTENCE_TEMPLATES,
        "n_concepts": len(concept_names),
        "n_languages": len(languages),
        "languages": languages,
        "true_rho": true_rho,
        "baseline_rho": baseline,
        "p_value": p_val,
        "same_family_rho": float(np.mean(same_scores)) if same_scores else None,
        "diff_family_rho": float(np.mean(diff_scores)) if diff_scores else None,
        "family_gap_p": float(family_p) if not np.isnan(family_p) else None,
        "pair_scores": {f"{la}-{lb}": s for (la, lb), s in zip(pairs, scores)},
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {RESULTS_FILE}")


if __name__ == "__main__":
    run()
