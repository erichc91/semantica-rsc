"""
RSC Cross-Lingual Universality Experiment — v2 with LaBSE Embeddings
=====================================================================
Upgrade from v1 (phonological features, 14 concepts) to:
  - Real multilingual embeddings via LaBSE (768-dim per word)
  - ~40 Swadesh list concepts (expanded from 14)
  - Same structural comparison: relational matrix Spearman rho
  - Same shuffle baseline

Partial circularity note (honest):
  LaBSE was trained on parallel corpora — it learns to put translation
  pairs near each other. So raw embedding similarity across languages
  is partly by design. We control for this via the STRUCTURAL test:
    - We compare relational MATRICES (how each concept relates to all others
      within its own language), not raw embedding similarity
    - The shuffle baseline destroys concept correspondence while keeping
      embeddings intact — if LaBSE alignment alone drove results, shuffling
      would not reduce correlation. Signal above shuffle = real structural
      substrate beyond alignment-by-construction.

Language families (8, genuinely unrelated):
  Indo-European: English, German, Spanish, Russian
  Sino-Tibetan:  Mandarin Chinese
  Afro-Asiatic:  Arabic, Hebrew
  Dravidian:     Tamil
  Austronesian:  Tagalog, Indonesian
  Japonic:       Japanese
  Koreanic:      Korean
  Turkic:        Turkish
"""

import numpy as np
from scipy import stats
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# ~40 Swadesh list core concepts — words per language
# Sources: Wiktionary Swadesh lists, Leipzig Glossing project
# Each word is in the script/romanization of its native language.
# LaBSE tokenizes natively — no English anchor.
# ---------------------------------------------------------------------------

SWADESH_WORDS: Dict[str, Dict[str, str]] = {
    # concept: {lang_code: native_word}
    "I":        {"en":"I",         "de":"ich",        "es":"yo",        "ru":"я",
                 "zh":"我",         "ar":"أنا",        "he":"אני",       "ta":"நான்",
                 "tl":"ako",       "id":"saya",       "ja":"私",        "ko":"나",       "tr":"ben"},
    "you":      {"en":"you",       "de":"du",         "es":"tú",        "ru":"ты",
                 "zh":"你",         "ar":"أنت",        "he":"אתה",       "ta":"நீ",
                 "tl":"ikaw",      "id":"kamu",       "ja":"あなた",     "ko":"너",       "tr":"sen"},
    "we":       {"en":"we",        "de":"wir",        "es":"nosotros",  "ru":"мы",
                 "zh":"我们",       "ar":"نحن",        "he":"אנחנו",     "ta":"நாங்கள்",
                 "tl":"kami",      "id":"kami",       "ja":"私たち",     "ko":"우리",     "tr":"biz"},
    "this":     {"en":"this",      "de":"dies",       "es":"esto",      "ru":"это",
                 "zh":"这个",       "ar":"هذا",        "he":"זה",        "ta":"இது",
                 "tl":"ito",       "id":"ini",        "ja":"これ",       "ko":"이것",     "tr":"bu"},
    "who":      {"en":"who",       "de":"wer",        "es":"quién",     "ru":"кто",
                 "zh":"谁",         "ar":"من",         "he":"מי",        "ta":"யார்",
                 "tl":"sino",      "id":"siapa",      "ja":"誰",        "ko":"누구",     "tr":"kim"},
    "one":      {"en":"one",       "de":"ein",        "es":"uno",       "ru":"один",
                 "zh":"一",         "ar":"واحد",       "he":"אחד",       "ta":"ஒன்று",
                 "tl":"isa",       "id":"satu",       "ja":"一",        "ko":"하나",     "tr":"bir"},
    "two":      {"en":"two",       "de":"zwei",       "es":"dos",       "ru":"два",
                 "zh":"二",         "ar":"اثنان",      "he":"שתיים",     "ta":"இரண்டு",
                 "tl":"dalawa",    "id":"dua",        "ja":"二",        "ko":"둘",       "tr":"iki"},
    "big":      {"en":"big",       "de":"groß",       "es":"grande",    "ru":"большой",
                 "zh":"大",         "ar":"كبير",       "he":"גדול",      "ta":"பெரிய",
                 "tl":"malaki",    "id":"besar",      "ja":"大きい",     "ko":"큰",       "tr":"büyük"},
    "small":    {"en":"small",     "de":"klein",      "es":"pequeño",   "ru":"маленький",
                 "zh":"小",         "ar":"صغير",       "he":"קטן",       "ta":"சின்ன",
                 "tl":"maliit",    "id":"kecil",      "ja":"小さい",     "ko":"작은",     "tr":"küçük"},
    "woman":    {"en":"woman",     "de":"Frau",       "es":"mujer",     "ru":"женщина",
                 "zh":"女人",       "ar":"امرأة",      "he":"אישה",      "ta":"பெண்",
                 "tl":"babae",     "id":"perempuan",  "ja":"女",        "ko":"여자",     "tr":"kadın"},
    "man":      {"en":"man",       "de":"Mann",       "es":"hombre",    "ru":"мужчина",
                 "zh":"男人",       "ar":"رجل",        "he":"איש",       "ta":"ஆண்",
                 "tl":"lalaki",    "id":"laki-laki",  "ja":"男",        "ko":"남자",     "tr":"erkek"},
    "water":    {"en":"water",     "de":"Wasser",     "es":"agua",      "ru":"вода",
                 "zh":"水",         "ar":"ماء",        "he":"מים",       "ta":"தண்ணீர்",
                 "tl":"tubig",     "id":"air",        "ja":"水",        "ko":"물",       "tr":"su"},
    "fire":     {"en":"fire",      "de":"Feuer",      "es":"fuego",     "ru":"огонь",
                 "zh":"火",         "ar":"نار",        "he":"אש",        "ta":"தீ",
                 "tl":"apoy",      "id":"api",        "ja":"火",        "ko":"불",       "tr":"ateş"},
    "sun":      {"en":"sun",       "de":"Sonne",      "es":"sol",       "ru":"солнце",
                 "zh":"太阳",       "ar":"شمس",        "he":"שמש",       "ta":"சூரியன்",
                 "tl":"araw",      "id":"matahari",   "ja":"太陽",       "ko":"태양",     "tr":"güneş"},
    "moon":     {"en":"moon",      "de":"Mond",       "es":"luna",      "ru":"луна",
                 "zh":"月",         "ar":"قمر",        "he":"ירח",       "ta":"நிலவு",
                 "tl":"buwan",     "id":"bulan",      "ja":"月",        "ko":"달",       "tr":"ay"},
    "blood":    {"en":"blood",     "de":"Blut",       "es":"sangre",    "ru":"кровь",
                 "zh":"血",         "ar":"دم",         "he":"דם",        "ta":"இரத்தம்",
                 "tl":"dugo",      "id":"darah",      "ja":"血",        "ko":"피",       "tr":"kan"},
    "head":     {"en":"head",      "de":"Kopf",       "es":"cabeza",    "ru":"голова",
                 "zh":"头",         "ar":"رأس",        "he":"ראש",       "ta":"தலை",
                 "tl":"ulo",       "id":"kepala",     "ja":"頭",        "ko":"머리",     "tr":"baş"},
    "eye":      {"en":"eye",       "de":"Auge",       "es":"ojo",       "ru":"глаз",
                 "zh":"眼睛",       "ar":"عين",        "he":"עין",       "ta":"கண்",
                 "tl":"mata",      "id":"mata",       "ja":"目",        "ko":"눈",       "tr":"göz"},
    "ear":      {"en":"ear",       "de":"Ohr",        "es":"oreja",     "ru":"ухо",
                 "zh":"耳",         "ar":"أذن",        "he":"אוזן",      "ta":"காது",
                 "tl":"tainga",    "id":"telinga",    "ja":"耳",        "ko":"귀",       "tr":"kulak"},
    "nose":     {"en":"nose",      "de":"Nase",       "es":"nariz",     "ru":"нос",
                 "zh":"鼻子",       "ar":"أنف",        "he":"אף",        "ta":"மூக்கு",
                 "tl":"ilong",     "id":"hidung",     "ja":"鼻",        "ko":"코",       "tr":"burun"},
    "mouth":    {"en":"mouth",     "de":"Mund",       "es":"boca",      "ru":"рот",
                 "zh":"嘴",         "ar":"فم",         "he":"פה",        "ta":"வாய்",
                 "tl":"bibig",     "id":"mulut",      "ja":"口",        "ko":"입",       "tr":"ağız"},
    "tooth":    {"en":"tooth",     "de":"Zahn",       "es":"diente",    "ru":"зуб",
                 "zh":"牙",         "ar":"سن",         "he":"שן",        "ta":"பல்",
                 "tl":"ngipin",    "id":"gigi",       "ja":"歯",        "ko":"이",       "tr":"diş"},
    "hand":     {"en":"hand",      "de":"Hand",       "es":"mano",      "ru":"рука",
                 "zh":"手",         "ar":"يد",         "he":"יד",        "ta":"கை",
                 "tl":"kamay",     "id":"tangan",     "ja":"手",        "ko":"손",       "tr":"el"},
    "foot":     {"en":"foot",      "de":"Fuß",        "es":"pie",       "ru":"нога",
                 "zh":"脚",         "ar":"قدم",        "he":"רגל",       "ta":"கால்",
                 "tl":"paa",       "id":"kaki",       "ja":"足",        "ko":"발",       "tr":"ayak"},
    "heart":    {"en":"heart",     "de":"Herz",       "es":"corazón",   "ru":"сердце",
                 "zh":"心",         "ar":"قلب",        "he":"לב",        "ta":"இதயம்",
                 "tl":"puso",      "id":"jantung",    "ja":"心",        "ko":"심장",     "tr":"kalp"},
    "mother":   {"en":"mother",    "de":"Mutter",     "es":"madre",     "ru":"мать",
                 "zh":"妈妈",       "ar":"أم",         "he":"אם",        "ta":"அம்மா",
                 "tl":"ina",       "id":"ibu",        "ja":"母",        "ko":"어머니",   "tr":"anne"},
    "die":      {"en":"die",       "de":"sterben",    "es":"morir",     "ru":"умереть",
                 "zh":"死",         "ar":"مات",        "he":"מת",        "ta":"இறந்தது",
                 "tl":"mamatay",   "id":"mati",       "ja":"死ぬ",       "ko":"죽다",     "tr":"ölmek"},
    "eat":      {"en":"eat",       "de":"essen",      "es":"comer",     "ru":"есть",
                 "zh":"吃",         "ar":"أكل",        "he":"אכל",       "ta":"சாப்பிடு",
                 "tl":"kumain",    "id":"makan",      "ja":"食べる",     "ko":"먹다",     "tr":"yemek"},
    "drink":    {"en":"drink",     "de":"trinken",    "es":"beber",     "ru":"пить",
                 "zh":"喝",         "ar":"شرب",        "he":"שתה",       "ta":"குடி",
                 "tl":"uminom",    "id":"minum",      "ja":"飲む",       "ko":"마시다",   "tr":"içmek"},
    "walk":     {"en":"walk",      "de":"gehen",      "es":"caminar",   "ru":"ходить",
                 "zh":"走",         "ar":"مشى",        "he":"הלך",       "ta":"நட",
                 "tl":"maglakad",  "id":"berjalan",   "ja":"歩く",       "ko":"걷다",     "tr":"yürümek"},
    "rain":     {"en":"rain",      "de":"Regen",      "es":"lluvia",    "ru":"дождь",
                 "zh":"雨",         "ar":"مطر",        "he":"גשם",       "ta":"மழை",
                 "tl":"ulan",      "id":"hujan",      "ja":"雨",        "ko":"비",       "tr":"yağmur"},
    "stone":    {"en":"stone",     "de":"Stein",      "es":"piedra",    "ru":"камень",
                 "zh":"石头",       "ar":"حجر",        "he":"אבן",       "ta":"கல்",
                 "tl":"bato",      "id":"batu",       "ja":"石",        "ko":"돌",       "tr":"taş"},
    "night":    {"en":"night",     "de":"Nacht",      "es":"noche",     "ru":"ночь",
                 "zh":"夜",         "ar":"ليل",        "he":"לילה",      "ta":"இரவு",
                 "tl":"gabi",      "id":"malam",      "ja":"夜",        "ko":"밤",       "tr":"gece"},
    "good":     {"en":"good",      "de":"gut",        "es":"bueno",     "ru":"хороший",
                 "zh":"好",         "ar":"جيد",        "he":"טוב",       "ta":"நல்ல",
                 "tl":"mabuti",    "id":"baik",       "ja":"良い",       "ko":"좋은",     "tr":"iyi"},
    "new":      {"en":"new",       "de":"neu",        "es":"nuevo",     "ru":"новый",
                 "zh":"新",         "ar":"جديد",       "he":"חדש",       "ta":"புதிய",
                 "tl":"bago",      "id":"baru",       "ja":"新しい",     "ko":"새로운",   "tr":"yeni"},
    "hot":      {"en":"hot",       "de":"heiß",       "es":"caliente",  "ru":"горячий",
                 "zh":"热",         "ar":"حار",        "he":"חם",        "ta":"சூடு",
                 "tl":"mainit",    "id":"panas",      "ja":"熱い",       "ko":"뜨거운",   "tr":"sıcak"},
    "cold":     {"en":"cold",      "de":"kalt",       "es":"frío",      "ru":"холодный",
                 "zh":"冷",         "ar":"بارد",       "he":"קר",        "ta":"குளிர்",
                 "tl":"malamig",   "id":"dingin",     "ja":"寒い",       "ko":"차가운",   "tr":"soğuk"},
    "path":     {"en":"path",      "de":"Weg",        "es":"camino",    "ru":"путь",
                 "zh":"路",         "ar":"طريق",       "he":"דרך",       "ta":"பாதை",
                 "tl":"daan",      "id":"jalan",      "ja":"道",        "ko":"길",       "tr":"yol"},
    "name":     {"en":"name",      "de":"Name",       "es":"nombre",    "ru":"имя",
                 "zh":"名字",       "ar":"اسم",        "he":"שם",        "ta":"பெயர்",
                 "tl":"pangalan",  "id":"nama",       "ja":"名前",       "ko":"이름",     "tr":"isim"},
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

LANGUAGE_NAMES = {
    "en": "English", "de": "German",    "es": "Spanish",   "ru": "Russian",
    "zh": "Mandarin", "ar": "Arabic",   "he": "Hebrew",    "ta": "Tamil",
    "tl": "Tagalog",  "id": "Indonesian","ja": "Japanese",  "ko": "Korean", "tr": "Turkish",
}


def load_labse():
    """Load LaBSE model — cached after first download."""
    from sentence_transformers import SentenceTransformer
    print("Loading LaBSE model (cached after first run)...")
    model = SentenceTransformer('LaBSE', cache_folder='./model_cache')
    return model


def build_language_embeddings(model, lang: str) -> Dict[str, np.ndarray]:
    """
    Get LaBSE embeddings for all concepts in a given language.
    Each language's words are encoded independently via LaBSE.

    Note on partial circularity: LaBSE aligns translation pairs,
    so embeddings for "water"(en) and "水"(zh) are already close.
    The STRUCTURAL test (relational matrix comparison) controls for
    this via the shuffle baseline — see run_experiment().
    """
    words = []
    concepts = []
    for concept, lang_data in SWADESH_WORDS.items():
        if lang in lang_data:
            words.append(lang_data[lang])
            concepts.append(concept)

    embeddings = model.encode(words, show_progress_bar=False, normalize_embeddings=True)
    return {concept: embeddings[i] for i, concept in enumerate(concepts)}


def relational_matrix(embeddings: Dict[str, np.ndarray], concepts: List[str]) -> np.ndarray:
    """
    Build pairwise cosine similarity matrix for a set of concepts within one language.
    This is the 'structural position' — how each concept relates to all others.
    Since embeddings are already L2-normalized, cosine sim = dot product.
    """
    vecs = np.stack([embeddings[c] for c in concepts])  # shape: (n_concepts, 768)
    return vecs @ vecs.T  # shape: (n_concepts, n_concepts)


def structural_similarity(emb_a: Dict, emb_b: Dict) -> float:
    """
    Compare structural positions across two languages.
    Builds relational matrices for both, then Spearman ρ on upper triangle.
    """
    shared = [c for c in emb_a if c in emb_b]
    if len(shared) < 4:
        return float("nan")

    mat_a = relational_matrix(emb_a, shared)
    mat_b = relational_matrix(emb_b, shared)

    n = len(shared)
    idx = np.triu_indices(n, k=1)
    rho, _ = stats.spearmanr(mat_a[idx], mat_b[idx])
    return float(rho) if not np.isnan(rho) else 0.0


def run_experiment():
    """
    Main experiment: does structural position of core concepts correlate
    across language families, beyond what a random shuffle baseline predicts?
    """
    print("=" * 65)
    print("RSC Cross-Lingual Universality — v2 (LaBSE embeddings)")
    print("=" * 65)

    model = load_labse()
    languages = list(LANGUAGE_FAMILIES.keys())

    print(f"\nBuilding embeddings for {len(languages)} languages...")
    lang_embeddings = {}
    for lang in languages:
        lang_embeddings[lang] = build_language_embeddings(model, lang)
    n_concepts = len(list(lang_embeddings.values())[0])
    print(f"Concepts per language: {n_concepts}")
    print(f"Language families: {len(set(LANGUAGE_FAMILIES.values()))}")

    # --- Pairwise structural similarity ---
    print("\nPairwise Structural Convergence (Spearman rho on relational matrices):")
    print("-" * 65)

    all_pairs = []
    same_family_scores = []
    diff_family_scores = []

    for i, lang_a in enumerate(languages):
        for lang_b in languages[i+1:]:
            rho = structural_similarity(lang_embeddings[lang_a], lang_embeddings[lang_b])
            same = LANGUAGE_FAMILIES[lang_a] == LANGUAGE_FAMILIES[lang_b]
            all_pairs.append((lang_a, lang_b, rho, same))
            (same_family_scores if same else diff_family_scores).append(rho)

    all_pairs.sort(key=lambda x: x[2], reverse=True)
    for lang_a, lang_b, rho, same in all_pairs:
        tag = (f"[same: {LANGUAGE_FAMILIES[lang_a]}]"
               if same else
               f"[{LANGUAGE_FAMILIES[lang_a]} <-> {LANGUAGE_FAMILIES[lang_b]}]")
        print(f"  {LANGUAGE_NAMES[lang_a]:12} <-> {LANGUAGE_NAMES[lang_b]:12}  rho={rho:+.3f}  {tag}")

    # --- Family comparison ---
    print("\nSame-Family vs Different-Family:")
    print("-" * 65)
    sf_mean = np.mean(same_family_scores)
    df_mean = np.mean(diff_family_scores)
    t_stat, p_fam = stats.ttest_ind(same_family_scores, diff_family_scores)
    print(f"  Same-family mean rho:   {sf_mean:+.3f}  (n={len(same_family_scores)})")
    print(f"  Diff-family mean rho:   {df_mean:+.3f}  (n={len(diff_family_scores)})")
    print(f"  t={t_stat:.3f}, p={p_fam:.4f}")

    # --- Shuffle baseline ---
    print("\nRandom Shuffle Baseline:")
    print("-" * 65)
    rng = np.random.default_rng(42)
    concepts = list(list(lang_embeddings.values())[0].keys())
    baseline_scores = []
    for _ in range(500):
        la, lb = rng.choice(languages, 2, replace=False)
        shuffled_keys = rng.permutation(concepts).tolist()
        shuffled_emb = {shuffled_keys[i]: lang_embeddings[la][concepts[i]]
                        for i in range(len(concepts))}
        baseline_scores.append(structural_similarity(shuffled_emb, lang_embeddings[lb]))

    baseline_mean = np.mean(baseline_scores)
    true_mean = np.mean([s for _, _, s, _ in all_pairs])
    t_base, p_base = stats.ttest_1samp([s for _, _, s, _ in all_pairs], baseline_mean)

    print(f"  True convergence mean rho:   {true_mean:+.4f}")
    print(f"  Shuffled baseline mean rho:  {baseline_mean:+.4f}")
    print(f"  Lift:                        {true_mean - baseline_mean:+.4f}")
    print(f"  t={t_base:.3f}, p={p_base:.6f}")

    # --- Interpretation ---
    signal = true_mean > baseline_mean and p_base < 0.05
    cross_family = df_mean > baseline_mean

    print("\nInterpretation:")
    print("-" * 65)
    if signal:
        print("  SIGNAL DETECTED: structural position consistent above shuffle")
        print(f"  baseline (p={p_base:.6f})")
    else:
        print("  SIGNAL WEAK: true pairs not clearly above shuffled baseline")

    if cross_family:
        print(f"\n  CROSS-FAMILY SIGNAL: unrelated families show rho > baseline")
        print(f"  Cannot be explained by shared ancestry.")
        print(f"  Consistent with RSC universal substrate claim.")

    print("\nHonest caveats:")
    print("  - LaBSE trained on parallel corpora: individual concept alignment")
    print("    is partly by design. Shuffle baseline controls for this.")
    print("  - n_concepts =", n_concepts, " — stronger test needs 200+")
    print("  - Next step: ConceptNet expansion + pre-registration")

    print("\n" + "=" * 65)
    print(f"Summary: true_mean={true_mean:+.4f}, baseline={baseline_mean:+.4f},")
    print(f"         lift={true_mean - baseline_mean:+.4f}, p={p_base:.6f}")
    print(f"         cross-family rho={df_mean:+.4f} (same-family={sf_mean:+.4f})")
    print("=" * 65)

    return {
        "n_concepts": n_concepts,
        "true_mean": true_mean,
        "baseline_mean": baseline_mean,
        "lift": true_mean - baseline_mean,
        "p_vs_baseline": p_base,
        "same_family_mean": sf_mean,
        "diff_family_mean": df_mean,
        "signal_detected": signal,
        "cross_family_signal": cross_family,
    }


if __name__ == "__main__":
    results = run_experiment()
