"""
RSC Targeted Experiments -- High-Signal Concept Sets
=====================================================
Three experiments testing RSC universality on curated, theoretically-motivated
concept sets. All use the same v4/v6 design: 13 monolingual BERT models, bare-word
[CLS] embeddings, language-level permutation test.

  Exp 1: EMOTION UNIVERSALITY (15 concepts)
    Prediction: rho > 0, likely strongest signal of all experiments.
    Motivation: Ekman basic emotions are theorized as pan-cultural and hardwired.
    If RSC holds anywhere, it should hold for emotions.

  Exp 2: ANTONYM STRUCTURAL CONSISTENCY (20 words = 10 pairs)
    Prediction: rho > 0 AND antonym pairs show lower cross-language distance
    variance than random pairs.
    Standard RSC test + special antonym-distance consistency analysis.

  Exp 3: CHILDREN'S FIRST WORDS (20 concepts)
    Prediction: STRONGEST rho of all experiments.
    Motivation: Cross-linguistic early vocabulary (WordBank, Tardif et al. 2008)
    represents pre-cultural concept priming -- closest to innate structure.

Runtime estimate: ~5 minutes total (1000 perms x 15-20 concepts = trivially fast).
"""

import os
import json
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Model registry (identical to v4/v6)
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

MODEL_CACHE = os.path.join(os.path.dirname(__file__), "..", "model_cache")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "targeted_experiments_results.json")

# ---------------------------------------------------------------------------
# Experiment 1: Emotion universality
# Ekman's 6 basic emotions + 9 secondary emotions = 15 total
# ---------------------------------------------------------------------------

EMOTION_CONCEPTS = {
    "joy":       {"en":"joy","de":"Freude","es":"alegria","ru":"радость","zh":"喜悦","ar":"فرح","he":"שמחה","ta":"மகிழ்ச்சி","tl":"galak","id":"kegembiraan","ja":"喜び","ko":"기쁨","tr":"sevinc"},
    "sadness":   {"en":"sadness","de":"Trauer","es":"tristeza","ru":"грусть","zh":"悲伤","ar":"حزن","he":"עצב","ta":"சோகம்","tl":"kalungkutan","id":"kesedihan","ja":"悲しみ","ko":"슬픔","tr":"uzuntu"},
    "anger":     {"en":"anger","de":"Wut","es":"ira","ru":"гнев","zh":"愤怒","ar":"غضب","he":"כעס","ta":"கோபம்","tl":"galit","id":"kemarahan","ja":"怒り","ko":"분노","tr":"ofke"},
    "fear":      {"en":"fear","de":"Angst","es":"miedo","ru":"страх","zh":"恐惧","ar":"خوف","he":"פחד","ta":"பயம்","tl":"takot","id":"ketakutan","ja":"恐れ","ko":"두려움","tr":"korku"},
    "disgust":   {"en":"disgust","de":"Ekel","es":"asco","ru":"отвращение","zh":"厌恶","ar":"اشمئزاز","he":"גועל","ta":"அருவருப்பு","tl":"kadiri","id":"jijik","ja":"嫌悪","ko":"혐오","tr":"igrenti"},
    "surprise":  {"en":"surprise","de":"Ueberraschung","es":"sorpresa","ru":"удивление","zh":"惊讶","ar":"مفاجأة","he":"הפתעה","ta":"ஆச்சரியம்","tl":"gulat","id":"kejutan","ja":"驚き","ko":"놀라움","tr":"surpriz"},
    "love":      {"en":"love","de":"Liebe","es":"amor","ru":"любовь","zh":"爱","ar":"حب","he":"אהבה","ta":"அன்பு","tl":"pag-ibig","id":"cinta","ja":"愛","ko":"사랑","tr":"ask"},
    "hate":      {"en":"hate","de":"Hass","es":"odio","ru":"ненависть","zh":"恨","ar":"كراهية","he":"שנאה","ta":"வெறுப்பு","tl":"galit","id":"kebencian","ja":"憎しみ","ko":"증오","tr":"nefret"},
    "shame":     {"en":"shame","de":"Scham","es":"verguenza","ru":"стыд","zh":"羞耻","ar":"خجل","he":"בושה","ta":"வெட்கம்","tl":"hiya","id":"malu","ja":"恥","ko":"수치심","tr":"utanc"},
    "pride":     {"en":"pride","de":"Stolz","es":"orgullo","ru":"гордость","zh":"骄傲","ar":"فخر","he":"גאווה","ta":"பெருமை","tl":"pagmamalaki","id":"kebanggaan","ja":"誇り","ko":"자부심","tr":"gurur"},
    "envy":      {"en":"envy","de":"Neid","es":"envidia","ru":"зависть","zh":"嫉妒","ar":"حسد","he":"קנאה","ta":"பொறாமை","tl":"inggit","id":"iri","ja":"嫉妬","ko":"질투","tr":"kiskanclık"},
    "anxiety":   {"en":"anxiety","de":"Sorge","es":"ansiedad","ru":"тревога","zh":"焦虑","ar":"قلق","he":"חרדה","ta":"கவலை","tl":"pagkabalisa","id":"kecemasan","ja":"不安","ko":"불안","tr":"kaygi"},
    "hope":      {"en":"hope","de":"Hoffnung","es":"esperanza","ru":"надежда","zh":"希望","ar":"أمل","he":"תקווה","ta":"நம்பிக்கை","tl":"pag-asa","id":"harapan","ja":"希望","ko":"희망","tr":"umut"},
    "calm":      {"en":"calm","de":"Ruhe","es":"calma","ru":"спокойствие","zh":"平静","ar":"هدوء","he":"שלווה","ta":"அமைதி","tl":"katahimikan","id":"tenang","ja":"平静","ko":"평온","tr":"sakinlik"},
    "trust":     {"en":"trust","de":"Vertrauen","es":"confianza","ru":"доверие","zh":"信任","ar":"ثقة","he":"אמון","ta":"நம்பகம்","tl":"tiwala","id":"kepercayaan","ja":"信頼","ko":"신뢰","tr":"guven"},
}

# ---------------------------------------------------------------------------
# Experiment 2: Antonym structural consistency (10 pairs = 20 words)
# ---------------------------------------------------------------------------

ANTONYM_PAIRS = [
    ("hot",    "cold"),
    ("big",    "small"),
    ("fast",   "slow"),
    ("light",  "dark"),
    ("young",  "old"),
    ("rich",   "poor"),
    ("happy",  "sad"),
    ("strong", "weak"),
    ("good",   "bad"),
    ("love",   "hate"),
]

ANTONYM_CONCEPTS = {
    "hot":    {"en":"hot","de":"heiss","es":"caliente","ru":"горячий","zh":"热","ar":"حار","he":"חם","ta":"சூடான","tl":"mainit","id":"panas","ja":"熱い","ko":"뜨거운","tr":"sicak"},
    "cold":   {"en":"cold","de":"kalt","es":"frio","ru":"холодный","zh":"冷","ar":"بارد","he":"קר","ta":"குளிர்ந்த","tl":"malamig","id":"dingin","ja":"冷たい","ko":"차가운","tr":"soguk"},
    "big":    {"en":"big","de":"gross","es":"grande","ru":"большой","zh":"大","ar":"كبير","he":"גדול","ta":"பெரிய","tl":"malaki","id":"besar","ja":"大きい","ko":"큰","tr":"buyuk"},
    "small":  {"en":"small","de":"klein","es":"pequeno","ru":"маленький","zh":"小","ar":"صغير","he":"קטן","ta":"சிறிய","tl":"maliit","id":"kecil","ja":"小さい","ko":"작은","tr":"kucuk"},
    "fast":   {"en":"fast","de":"schnell","es":"rapido","ru":"быстрый","zh":"快","ar":"سريع","he":"מהיר","ta":"வேகமான","tl":"mabilis","id":"cepat","ja":"速い","ko":"빠른","tr":"hizli"},
    "slow":   {"en":"slow","de":"langsam","es":"lento","ru":"медленный","zh":"慢","ar":"بطيء","he":"איטי","ta":"மெதுவான","tl":"mabagal","id":"lambat","ja":"遅い","ko":"느린","tr":"yavas"},
    "light":  {"en":"light","de":"hell","es":"luz","ru":"светлый","zh":"亮","ar":"مضيء","he":"בהיר","ta":"ஒளிமயமான","tl":"maliwanag","id":"terang","ja":"明るい","ko":"밝은","tr":"aydinlik"},
    "dark":   {"en":"dark","de":"dunkel","es":"oscuro","ru":"тёмный","zh":"暗","ar":"مظلم","he":"כהה","ta":"இருண்ட","tl":"madilim","id":"gelap","ja":"暗い","ko":"어두운","tr":"karanlik"},
    "young":  {"en":"young","de":"jung","es":"joven","ru":"молодой","zh":"年轻","ar":"شاب","he":"צעיר","ta":"இளைய","tl":"bata","id":"muda","ja":"若い","ko":"젊은","tr":"genc"},
    "old":    {"en":"old","de":"alt","es":"viejo","ru":"старый","zh":"老","ar":"عجوز","he":"זקן","ta":"வயதான","tl":"matanda","id":"tua","ja":"老いた","ko":"나이든","tr":"yasli"},
    "rich":   {"en":"rich","de":"reich","es":"rico","ru":"богатый","zh":"富","ar":"غني","he":"עשיר","ta":"பணக்கார","tl":"mayaman","id":"kaya","ja":"豊か","ko":"부유한","tr":"zengin"},
    "poor":   {"en":"poor","de":"arm","es":"pobre","ru":"бедный","zh":"穷","ar":"فقير","he":"עני","ta":"ஏழை","tl":"mahirap","id":"miskin","ja":"貧しい","ko":"가난한","tr":"fakir"},
    "happy":  {"en":"happy","de":"glucklich","es":"feliz","ru":"счастливый","zh":"快乐","ar":"سعيد","he":"שמח","ta":"மகிழ்ச்சியான","tl":"masaya","id":"bahagia","ja":"幸せ","ko":"행복한","tr":"mutlu"},
    "sad":    {"en":"sad","de":"traurig","es":"triste","ru":"грустный","zh":"悲伤","ar":"حزين","he":"עצוב","ta":"சோகமான","tl":"malungkot","id":"sedih","ja":"悲しい","ko":"슬픈","tr":"uzgun"},
    "strong": {"en":"strong","de":"stark","es":"fuerte","ru":"сильный","zh":"强","ar":"قوي","he":"חזק","ta":"வலிமையான","tl":"malakas","id":"kuat","ja":"強い","ko":"강한","tr":"guclu"},
    "weak":   {"en":"weak","de":"schwach","es":"debil","ru":"слабый","zh":"弱","ar":"ضعيف","he":"חלש","ta":"பலவீனமான","tl":"mahina","id":"lemah","ja":"弱い","ko":"약한","tr":"zayif"},
    "good":   {"en":"good","de":"gut","es":"bueno","ru":"хороший","zh":"好","ar":"جيد","he":"טוב","ta":"நல்ல","tl":"mabuti","id":"baik","ja":"良い","ko":"좋은","tr":"iyi"},
    "bad":    {"en":"bad","de":"schlecht","es":"malo","ru":"плохой","zh":"坏","ar":"سيئ","he":"רע","ta":"கெட்ட","tl":"masama","id":"buruk","ja":"悪い","ko":"나쁜","tr":"kotu"},
    "love":   {"en":"love","de":"Liebe","es":"amor","ru":"любовь","zh":"爱","ar":"حب","he":"אהבה","ta":"அன்பு","tl":"pag-ibig","id":"cinta","ja":"愛","ko":"사랑","tr":"ask"},
    "hate":   {"en":"hate","de":"Hass","es":"odio","ru":"ненависть","zh":"恨","ar":"كراهية","he":"שנאה","ta":"வெறுப்பு","tl":"pagkapoot","id":"kebencian","ja":"憎しみ","ko":"증오","tr":"nefret"},
}

# ---------------------------------------------------------------------------
# Experiment 3: Children's first words
# Based on WordBank cross-linguistic acquisition data (Frank et al. 2021)
# and Tardif et al. 2008 (first words across 5 languages)
# ---------------------------------------------------------------------------

CHILD_CONCEPTS = {
    "mama":   {"en":"mama","de":"Mama","es":"mama","ru":"мама","zh":"妈妈","ar":"ماما","he":"מאמא","ta":"அம்மா","tl":"mama","id":"mama","ja":"ママ","ko":"엄마","tr":"anne"},
    "papa":   {"en":"papa","de":"Papa","es":"papa","ru":"папа","zh":"爸爸","ar":"بابا","he":"פאפא","ta":"அப்பா","tl":"papa","id":"papa","ja":"パパ","ko":"아빠","tr":"baba"},
    "no":     {"en":"no","de":"nein","es":"no","ru":"нет","zh":"不","ar":"لا","he":"לא","ta":"இல்லை","tl":"hindi","id":"tidak","ja":"いや","ko":"아니요","tr":"hayir"},
    "more":   {"en":"more","de":"mehr","es":"mas","ru":"ещё","zh":"更多","ar":"أكثر","he":"עוד","ta":"இன்னும்","tl":"higit","id":"lagi","ja":"もっと","ko":"더","tr":"daha"},
    "hot":    {"en":"hot","de":"heiss","es":"caliente","ru":"горячий","zh":"热","ar":"حار","he":"חם","ta":"சூடான","tl":"mainit","id":"panas","ja":"あつい","ko":"뜨거운","tr":"sicak"},
    "ball":   {"en":"ball","de":"Ball","es":"pelota","ru":"мяч","zh":"球","ar":"كرة","he":"כדור","ta":"பந்து","tl":"bola","id":"bola","ja":"ボール","ko":"공","tr":"top"},
    "dog":    {"en":"dog","de":"Hund","es":"perro","ru":"собака","zh":"狗","ar":"كلب","he":"כלב","ta":"நாய்","tl":"aso","id":"anjing","ja":"いぬ","ko":"개","tr":"kopek"},
    "cat":    {"en":"cat","de":"Katze","es":"gato","ru":"кошка","zh":"猫","ar":"قطة","he":"חתול","ta":"பூனை","tl":"pusa","id":"kucing","ja":"ねこ","ko":"고양이","tr":"kedi"},
    "water":  {"en":"water","de":"Wasser","es":"agua","ru":"вода","zh":"水","ar":"ماء","he":"מים","ta":"தண்ணீர்","tl":"tubig","id":"air","ja":"みず","ko":"물","tr":"su"},
    "milk":   {"en":"milk","de":"Milch","es":"leche","ru":"молоко","zh":"牛奶","ar":"حليب","he":"חלב","ta":"பால்","tl":"gatas","id":"susu","ja":"ミルク","ko":"우유","tr":"sut"},
    "eat":    {"en":"eat","de":"essen","es":"comer","ru":"есть","zh":"吃","ar":"أكل","he":"לאכול","ta":"சாப்பிடு","tl":"kumain","id":"makan","ja":"たべる","ko":"먹다","tr":"yemek"},
    "baby":   {"en":"baby","de":"Baby","es":"bebe","ru":"малыш","zh":"宝宝","ar":"طفل","he":"תינוק","ta":"குழந்தை","tl":"sanggol","id":"bayi","ja":"あかちゃん","ko":"아기","tr":"bebek"},
    "yes":    {"en":"yes","de":"ja","es":"si","ru":"да","zh":"是","ar":"نعم","he":"כן","ta":"ஆம்","tl":"oo","id":"ya","ja":"うん","ko":"응","tr":"evet"},
    "up":     {"en":"up","de":"hoch","es":"arriba","ru":"вверх","zh":"上","ar":"فوق","he":"למעלה","ta":"மேலே","tl":"taas","id":"atas","ja":"うえ","ko":"위","tr":"yukari"},
    "done":   {"en":"done","de":"fertig","es":"listo","ru":"готово","zh":"完了","ar":"انتهى","he":"גמור","ta":"முடிந்தது","tl":"tapos","id":"selesai","ja":"おわり","ko":"끝","tr":"bitti"},
    "come":   {"en":"come","de":"kommen","es":"ven","ru":"иди","zh":"来","ar":"تعال","he":"בוא","ta":"வா","tl":"halika","id":"datang","ja":"くる","ko":"와","tr":"gel"},
    "look":   {"en":"look","de":"schauen","es":"mira","ru":"смотри","zh":"看","ar":"انظر","he":"תסתכל","ta":"பார்","tl":"tingnan","id":"lihat","ja":"みる","ko":"봐","tr":"bak"},
    "hello":  {"en":"hello","de":"hallo","es":"hola","ru":"привет","zh":"你好","ar":"مرحبا","he":"שלום","ta":"வணக்கம்","tl":"kumusta","id":"halo","ja":"こんにちは","ko":"안녕","tr":"merhaba"},
    "bye":    {"en":"bye","de":"tschuss","es":"adios","ru":"пока","zh":"再见","ar":"وداعا","he":"להתראות","ta":"போய்வா","tl":"paalam","id":"sampai jumpa","ko":"바이바이","ja":"バイバイ","tr":"hosca kal"},
    "go":     {"en":"go","de":"gehen","es":"ir","ru":"идти","zh":"走","ar":"اذهب","he":"לך","ta":"போ","tl":"pumunta","id":"pergi","ja":"いく","ko":"가","tr":"git"},
}

# ---------------------------------------------------------------------------
# BERT encoding (bare-word [CLS] token — identical to v4)
# ---------------------------------------------------------------------------

_model_cache = {}

def load_model(lang: str, model_name: str, lang_name: str):
    if lang in _model_cache:
        return _model_cache[lang]
    import torch
    from transformers import AutoTokenizer, AutoModel
    cache = MODEL_CACHE
    print(f"  [{lang_name:<12}] Loading {model_name}...", end=" ", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache)
    mdl = AutoModel.from_pretrained(model_name, cache_dir=cache)
    mdl.eval()
    _model_cache[lang] = (tok, mdl)
    print("ok")
    return tok, mdl


def encode_words(lang: str, model_name: str, lang_name: str, words: List[str], concepts: List[str]) -> Dict:
    import torch
    tok, mdl = load_model(lang, model_name, lang_name)
    embeddings = {}
    with torch.no_grad():
        for concept, word in zip(concepts, words):
            inputs = tok(word, return_tensors="pt", truncation=True, max_length=32)
            outputs = mdl(**inputs)
            vec = outputs.last_hidden_state[0, 0, :].numpy()
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            embeddings[concept] = vec
    return embeddings


def encode_concept_set(concept_dict: Dict, lang: str) -> Dict:
    """Encode all concepts for one language using its native word."""
    model_name, lang_name = MONOLINGUAL_MODELS[lang]
    concepts = list(concept_dict.keys())
    words = [concept_dict[c][lang] for c in concepts]
    return encode_words(lang, model_name, lang_name, words, concepts)

# ---------------------------------------------------------------------------
# RSC core (identical to v4/v6)
# ---------------------------------------------------------------------------

def relational_matrix(embeddings: Dict, concepts: List[str]) -> np.ndarray:
    vecs = np.stack([embeddings[c] for c in concepts])
    return vecs @ vecs.T


def structural_similarity(emb_a: Dict, emb_b: Dict) -> float:
    shared = [c for c in emb_a if c in emb_b]
    if len(shared) < 5:
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


def permutation_test(lang_embeddings: Dict, languages: List[str], n_perm: int = 1000) -> float:
    rng = np.random.default_rng(42)
    all_keys = list(list(lang_embeddings.values())[0].keys())
    _, true_scores, _, _ = compute_all_pairs(lang_embeddings, languages)
    observed_mean = float(np.mean(true_scores))
    count_geq = 0
    for _ in range(n_perm):
        shuffled = {}
        for lang, emb in lang_embeddings.items():
            perm = rng.permutation(all_keys).tolist()
            orig = list(emb.keys())
            shuffled[lang] = {perm[i]: emb[orig[i]] for i in range(len(orig))}
        _, perm_scores, _, _ = compute_all_pairs(shuffled, languages)
        if perm_scores and float(np.mean(perm_scores)) >= observed_mean:
            count_geq += 1
    p = count_geq / n_perm
    return p if p > 0 else 1.0 / (n_perm + 1)

# ---------------------------------------------------------------------------
# Special analysis: antonym distance consistency
# Tests whether antonym pairs occupy CONSISTENTLY DISTANT positions
# in the relational matrix, across all languages.
# RSC prediction: antonym-pair distances should be MORE consistent
# (lower std dev across languages) than random pairs.
# ---------------------------------------------------------------------------

def antonym_distance_analysis(lang_embeddings: Dict, languages: List[str]) -> Dict:
    """
    For each antonym pair and each language, compute cosine similarity
    between the two words. Measure consistency (std dev) across languages.
    Compare to random pair consistency.
    """
    results = {}
    rng = np.random.default_rng(42)
    concept_keys = list(list(lang_embeddings.values())[0].keys())

    # Antonym pair distances per language
    pair_distances = {f"{a}_{b}": [] for a, b in ANTONYM_PAIRS}
    for lang in languages:
        emb = lang_embeddings[lang]
        for a, b in ANTONYM_PAIRS:
            if a in emb and b in emb:
                sim = float(np.dot(emb[a], emb[b]))  # cosine (vecs already normalized)
                pair_distances[f"{a}_{b}"].append(sim)

    # Antonym consistency: std dev across languages for each pair
    antonym_stds = []
    antonym_means = []
    for pair_key, sims in pair_distances.items():
        if len(sims) >= 5:
            antonym_stds.append(float(np.std(sims)))
            antonym_means.append(float(np.mean(sims)))

    # Random pair consistency: sample 100 random pairs, measure their std dev
    random_stds = []
    for _ in range(100):
        a_key, b_key = rng.choice(concept_keys, 2, replace=False)
        random_sims = []
        for lang in languages:
            emb = lang_embeddings[lang]
            if a_key in emb and b_key in emb:
                sim = float(np.dot(emb[a_key], emb[b_key]))
                random_sims.append(sim)
        if len(random_sims) >= 5:
            random_stds.append(float(np.std(random_sims)))

    avg_antonym_std = float(np.mean(antonym_stds)) if antonym_stds else float("nan")
    avg_random_std  = float(np.mean(random_stds))  if random_stds  else float("nan")
    avg_antonym_sim = float(np.mean(antonym_means)) if antonym_means else float("nan")

    # t-test: are antonym stds significantly lower than random stds?
    t, p = stats.ttest_ind(antonym_stds, random_stds) if (antonym_stds and random_stds) else (float("nan"), float("nan"))

    return {
        "pair_distances": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in pair_distances.items() if v},
        "avg_antonym_std": avg_antonym_std,
        "avg_random_std":  avg_random_std,
        "avg_antonym_similarity": avg_antonym_sim,
        "antonym_std_vs_random_t": float(t),
        "antonym_std_vs_random_p": float(p),
        "interpretation": (
            "Antonym pairs MORE structurally consistent than random (RSC holds for antonyms)"
            if avg_antonym_std < avg_random_std else
            "Antonym pairs NOT more consistent than random"
        ),
    }

# ---------------------------------------------------------------------------
# Run one experiment
# ---------------------------------------------------------------------------

def run_experiment(name: str, concept_dict: Dict, n_perm: int = 1000,
                   run_antonym_analysis: bool = False) -> Dict:
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"  Concepts: {len(concept_dict)} | Permutations: {n_perm}")
    print(f"{'='*60}")

    # Encode all languages
    lang_embeddings = {}
    active_langs = []
    for lang in MONOLINGUAL_MODELS:
        if all(lang in concept_dict[c] for c in concept_dict):
            emb = encode_concept_set(concept_dict, lang)
            if emb:
                lang_embeddings[lang] = emb
                active_langs.append(lang)

    print(f"  Encoded: {len(active_langs)} languages")

    # RSC test
    pairs, scores, fam_a, fam_b = compute_all_pairs(lang_embeddings, active_langs)
    observed_rho = float(np.mean(scores)) if scores else float("nan")

    print(f"  Observed mean rho: {observed_rho:+.4f}")
    print(f"  Running permutation test ({n_perm} perms)...", flush=True)
    p_val = permutation_test(lang_embeddings, active_langs, n_perm=n_perm)
    print(f"  p = {p_val:.2e}")

    # Family gap
    same = [s for s, fa, fb in zip(scores, fam_a, fam_b) if fa == fb]
    diff = [s for s, fa, fb in zip(scores, fam_a, fam_b) if fa != fb]
    _, p_fam = stats.ttest_ind(same, diff) if (same and diff) else (None, float("nan"))

    print(f"  family_gap_p: {p_fam:.3f} ({'sig - some family effect' if p_fam < 0.05 else 'n.s. - truly universal'})")

    result = {
        "name": name,
        "n_concepts": len(concept_dict),
        "n_languages": len(active_langs),
        "active_languages": active_langs,
        "observed_rho": observed_rho,
        "p_value": p_val,
        "n_permutations": n_perm,
        "same_family_rho": float(np.mean(same)) if same else None,
        "diff_family_rho": float(np.mean(diff)) if diff else None,
        "family_gap_p": float(p_fam) if not np.isnan(p_fam) else None,
    }

    if run_antonym_analysis:
        print(f"  Running antonym-distance consistency analysis...")
        antonym_result = antonym_distance_analysis(lang_embeddings, active_langs)
        result["antonym_analysis"] = antonym_result
        print(f"  Avg antonym consistency std: {antonym_result['avg_antonym_std']:.4f}")
        print(f"  Avg random pair std:         {antonym_result['avg_random_std']:.4f}")
        print(f"  -> {antonym_result['interpretation']}")
        print(f"  antonym vs random p: {antonym_result['antonym_std_vs_random_p']:.3f}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("RSC Targeted Experiments -- High-Signal Concept Sets")
    print("="*60)
    print("Three theoretically-motivated experiments:")
    print("  1. Emotion universality  (15 concepts)")
    print("  2. Antonym consistency   (20 words, 10 pairs)")
    print("  3. Children's first words (20 concepts)")
    print("  Using 1000 permutations -- est. < 5 min total")
    print("="*60)

    import time
    all_results = []
    t0 = time.time()

    # Experiment 1: Emotions
    t1 = time.time()
    r1 = run_experiment("Emotion Universality (Ekman + secondary)", EMOTION_CONCEPTS, n_perm=1000)
    r1["runtime_seconds"] = round(time.time() - t1, 1)
    all_results.append(r1)

    # Experiment 2: Antonyms
    t2 = time.time()
    r2 = run_experiment("Antonym Structural Consistency (10 pairs)", ANTONYM_CONCEPTS, n_perm=1000,
                        run_antonym_analysis=True)
    r2["runtime_seconds"] = round(time.time() - t2, 1)
    all_results.append(r2)

    # Experiment 3: Children's first words
    t3 = time.time()
    r3 = run_experiment("Children's First Words (WordBank cross-linguistic)", CHILD_CONCEPTS, n_perm=1000)
    r3["runtime_seconds"] = round(time.time() - t3, 1)
    all_results.append(r3)

    total = round(time.time() - t0, 1)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY -- All Targeted Experiments")
    print(f"{'='*60}")
    print(f"  {'Experiment':<45} {'n':>4}  {'rho':>7}  {'p-value':>10}  {'fam_gap':>7}  {'time':>6}")
    print(f"  {'-'*45} {'-'*4}  {'-'*7}  {'-'*10}  {'-'*7}  {'-'*6}")
    for r in all_results:
        sig = "*" if r["p_value"] < 0.05 else " "
        print(f"  {r['name'][:45]:<45} {r['n_concepts']:>4}  {r['observed_rho']:>+7.4f}  {r['p_value']:>10.2e}{sig} {r['family_gap_p']:>7.3f}  {r['runtime_seconds']:>5.0f}s")
    print(f"\n  Total runtime: {total:.0f}s | * = p<0.05")

    # Compare to v6 baseline
    v6_rho = 0.073
    print(f"\n  v6 baseline (high-freq, 300 concepts): rho={v6_rho:+.4f}")
    for r in all_results:
        diff = r["observed_rho"] - v6_rho
        print(f"  {r['name'][:40]:<40}: {'+' if diff >= 0 else ''}{diff:.4f} vs baseline")

    # Save
    save_data = {
        "version": "targeted_experiments_v1",
        "description": "Three curated concept-set experiments: emotions, antonyms, children's first words",
        "total_runtime_seconds": total,
        "experiments": all_results,
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: {RESULTS_FILE}")


if __name__ == "__main__":
    run()
