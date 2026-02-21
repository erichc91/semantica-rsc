"""
RSC Communication Structure Detector — Falsification Test
==========================================================
HYPOTHESIS: RSC rho (cross-lingual relational consistency) is a
quantitative metric for meaningful structured communication.

PREDICTION: Domain-coherent concept sets will show significantly
higher RSC rho than randomly mixed concept sets drawn from the
same vocabulary pool.

Three conditions:
  A) COHERENT   — 5 domain-specific sets (astronomy, emotion, body, food, politics)
                  Prediction: rho ~ 0.10-0.15 (strong structural relationships)
  B) MIXED      — 5 randomly mixed sets from the same 75 words
                  Prediction: rho ~ 0.06-0.08 (our v4 baseline, no domain structure)
  C) NOISE      — 5 sets of unrelated low-frequency/obscure words
                  Prediction: rho ~ 0.02-0.04 (approaching zero)

FALSIFICATION: If coherent == mixed, RSC does NOT detect structured communication.
If coherent > mixed > noise, RSC IS a communication structure detector.

Only 2 languages (English + German) — clean bilateral comparison.
Models already cached. Estimated runtime: ~90 seconds.
"""

import os, json, time
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

MODEL_CACHE = os.path.join(os.path.dirname(__file__), "..", "model_cache")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "rsc_detector_results.json")

# ---------------------------------------------------------------------------
# Concept sets — 5 domains x 15 concepts, with English + German translations
# ---------------------------------------------------------------------------

DOMAINS = {
    "astronomy": {
        "en": ["star","planet","orbit","galaxy","light","gravity","moon","comet","sun","space","meteor","telescope","nebula","universe","void"],
        "de": ["Stern","Planet","Orbit","Galaxie","Licht","Schwerkraft","Mond","Komet","Sonne","Raum","Meteor","Teleskop","Nebel","Universum","Leere"],
    },
    "emotions": {
        "en": ["joy","sadness","anger","fear","disgust","surprise","love","hate","shame","pride","envy","anxiety","hope","calm","trust"],
        "de": ["Freude","Trauer","Wut","Angst","Ekel","Ueberraschung","Liebe","Hass","Scham","Stolz","Neid","Sorge","Hoffnung","Ruhe","Vertrauen"],
    },
    "body": {
        "en": ["eye","hand","foot","heart","brain","mouth","ear","nose","arm","leg","back","skin","blood","bone","face"],
        "de": ["Auge","Hand","Fuss","Herz","Gehirn","Mund","Ohr","Nase","Arm","Bein","Ruecken","Haut","Blut","Knochen","Gesicht"],
    },
    "food": {
        "en": ["bread","water","meat","fish","fruit","milk","egg","salt","sugar","oil","wine","rice","soup","cake","vegetable"],
        "de": ["Brot","Wasser","Fleisch","Fisch","Frucht","Milch","Ei","Salz","Zucker","Oel","Wein","Reis","Suppe","Kuchen","Gemuese"],
    },
    "politics": {
        "en": ["law","power","war","justice","freedom","leader","vote","state","army","peace","rights","order","rule","party","conflict"],
        "de": ["Gesetz","Macht","Krieg","Gerechtigkeit","Freiheit","Fuehrer","Wahl","Staat","Armee","Frieden","Rechte","Ordnung","Regel","Partei","Konflikt"],
    },
}

# NOISE condition — obscure, unrelated, low-frequency words with no coherent structure
# Intentionally cross-domain and semantically unrelated within each set
NOISE_SETS = {
    "noise_1": {
        "en": ["hinge","decimal","marsh","clergy","axle","velvet","census","tundra","memoir","capsule","forge","anthem","parish","ledger","spore"],
        "de": ["Scharnier","Dezimal","Sumpf","Klerus","Achse","Samt","Volkszaehlung","Tundra","Memoiren","Kapsel","Schmiede","Hymne","Gemeinde","Hauptbuch","Spore"],
    },
    "noise_2": {
        "en": ["cobalt","riddle","funnel","brace","plaque","herald","notch","crest","anvil","dagger","flask","pewter","casket","tuber","silt"],
        "de": ["Kobalt","Raetsel","Trichter","Strebe","Plakette","Herold","Kerbe","Wappen","Amboss","Dolch","Flasche","Zinn","Schatulle","Knolle","Schlick"],
    },
    "noise_3": {
        "en": ["cornet","thatch","glaze","sinew","pestle","tallow","grovel","crevice","wraith","quill","buoy","mallet","scythe","tendril","mortar"],
        "de": ["Kornett","Stroh","Glasur","Sehne","Stoesser","Talg","Kriechen","Spalte","Geist","Feder","Boje","Schlegel","Sense","Ranke","Moertel"],
    },
}

# ---------------------------------------------------------------------------
# Build MIXED condition from domain vocabulary
# Take 3 words from each domain per mixed set -> completely cross-domain
# ---------------------------------------------------------------------------

def build_mixed_sets():
    domain_names = list(DOMAINS.keys())
    en_words_by_domain = {d: DOMAINS[d]["en"] for d in domain_names}
    de_words_by_domain = {d: DOMAINS[d]["de"] for d in domain_names}

    # 5 domains x 15 words = 75 words total
    # Split into 5 groups of 15 by taking every 5th word across all domains
    # so each mixed set has 3 words from each of the 5 domains
    mixed_sets = {}
    for i in range(5):
        en_mix, de_mix = [], []
        for d in domain_names:
            en_mix += en_words_by_domain[d][i*3:(i+1)*3]
            de_mix += de_words_by_domain[d][i*3:(i+1)*3]
        mixed_sets[f"mixed_{i+1}"] = {"en": en_mix, "de": de_mix}
    return mixed_sets

# ---------------------------------------------------------------------------
# BERT encoding (bare-word, CLS token — same as v4)
# ---------------------------------------------------------------------------

_model_cache_mem = {}

def load_bert(lang_code, model_name, display):
    if lang_code in _model_cache_mem:
        return _model_cache_mem[lang_code]
    import torch
    from transformers import AutoTokenizer, AutoModel
    print(f"  Loading {display} ({model_name})...", end=" ", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE)
    mdl = AutoModel.from_pretrained(model_name, cache_dir=MODEL_CACHE)
    mdl.eval()
    _model_cache_mem[lang_code] = (tok, mdl)
    print("ok")
    return tok, mdl


def encode_words(tok, mdl, words, labels):
    import torch
    emb = {}
    with torch.no_grad():
        for label, word in zip(labels, words):
            inp = tok(word, return_tensors="pt", truncation=True, max_length=16)
            out = mdl(**inp)
            vec = out.last_hidden_state[0, 0, :].numpy().copy()
            vec /= (np.linalg.norm(vec) + 1e-9)
            emb[label] = vec
    return emb

# ---------------------------------------------------------------------------
# RSC rho between two embedding dicts
# ---------------------------------------------------------------------------

def rsc_rho(emb_a: dict, emb_b: dict) -> float:
    shared = [k for k in emb_a if k in emb_b]
    if len(shared) < 5:
        return float("nan")
    vecs_a = np.stack([emb_a[k] for k in shared])
    vecs_b = np.stack([emb_b[k] for k in shared])
    mat_a = vecs_a @ vecs_a.T
    mat_b = vecs_b @ vecs_b.T
    n = len(shared)
    idx = np.triu_indices(n, k=1)
    rho, _ = stats.spearmanr(mat_a[idx], mat_b[idx])
    return float(rho) if not np.isnan(rho) else 0.0


def permutation_test_bilateral(emb_en: dict, emb_de: dict, n_perm=500) -> float:
    """One-pair permutation test: shuffle labels in EN, measure rho vs DE."""
    rng = np.random.default_rng(42)
    keys = list(emb_en.keys())
    observed = rsc_rho(emb_en, emb_de)
    count = 0
    for _ in range(n_perm):
        perm_keys = rng.permutation(keys).tolist()
        shuffled = {perm_keys[i]: emb_en[keys[i]] for i in range(len(keys))}
        if rsc_rho(shuffled, emb_de) >= observed:
            count += 1
    p = count / n_perm
    return p if p > 0 else 1.0 / (n_perm + 1)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    t0 = time.time()
    print("RSC Communication Structure Detector -- Falsification Test")
    print("=" * 65)
    print("PREDICTION: Coherent domain sets > Mixed sets > Noise sets")
    print("If confirmed: RSC rho = metric for meaningful structured communication")
    print("=" * 65)

    # Load models
    print("\nLoading models (using cache)...")
    en_tok, en_mdl = load_bert("en", "bert-base-uncased", "English BERT")
    de_tok, de_mdl = load_bert("de", "bert-base-german-cased", "German BERT")

    mixed_sets = build_mixed_sets()

    all_results = {"coherent": [], "mixed": [], "noise": []}

    # --- COHERENT condition ---
    print("\n--- CONDITION A: COHERENT (domain-specific sets) ---")
    for domain, trans in DOMAINS.items():
        labels = [f"{domain}_{i}" for i in range(len(trans["en"]))]
        emb_en = encode_words(en_tok, en_mdl, trans["en"], labels)
        emb_de = encode_words(de_tok, de_mdl, trans["de"], labels)
        rho = rsc_rho(emb_en, emb_de)
        p = permutation_test_bilateral(emb_en, emb_de, n_perm=500)
        all_results["coherent"].append({"name": domain, "rho": rho, "p": p})
        print(f"  {domain:<12}: rho={rho:+.4f}  p={p:.3f}")

    # --- MIXED condition ---
    print("\n--- CONDITION B: MIXED (cross-domain shuffled sets) ---")
    for set_name, trans in mixed_sets.items():
        labels = [f"mix_{i}" for i in range(len(trans["en"]))]
        emb_en = encode_words(en_tok, en_mdl, trans["en"], labels)
        emb_de = encode_words(de_tok, de_mdl, trans["de"], labels)
        rho = rsc_rho(emb_en, emb_de)
        p = permutation_test_bilateral(emb_en, emb_de, n_perm=500)
        all_results["mixed"].append({"name": set_name, "rho": rho, "p": p})
        print(f"  {set_name:<12}: rho={rho:+.4f}  p={p:.3f}")

    # --- NOISE condition ---
    print("\n--- CONDITION C: NOISE (semantically incoherent words) ---")
    for set_name, trans in NOISE_SETS.items():
        labels = [f"noise_{i}" for i in range(len(trans["en"]))]
        emb_en = encode_words(en_tok, en_mdl, trans["en"], labels)
        emb_de = encode_words(de_tok, de_mdl, trans["de"], labels)
        rho = rsc_rho(emb_en, emb_de)
        p = permutation_test_bilateral(emb_en, emb_de, n_perm=500)
        all_results["noise"].append({"name": set_name, "rho": rho, "p": p})
        print(f"  {set_name:<12}: rho={rho:+.4f}  p={p:.3f}")

    # --- Statistical comparison ---
    coherent_rhos = [r["rho"] for r in all_results["coherent"]]
    mixed_rhos    = [r["rho"] for r in all_results["mixed"]]
    noise_rhos    = [r["rho"] for r in all_results["noise"]]

    mean_c = float(np.mean(coherent_rhos))
    mean_m = float(np.mean(mixed_rhos))
    mean_n = float(np.mean(noise_rhos))

    t_cm, p_cm = stats.ttest_ind(coherent_rhos, mixed_rhos)
    t_mn, p_mn = stats.ttest_ind(mixed_rhos, noise_rhos)
    t_cn, p_cn = stats.ttest_ind(coherent_rhos, noise_rhos)

    print(f"\n{'='*65}")
    print(f"RESULTS -- RSC COMMUNICATION STRUCTURE DETECTOR")
    print(f"{'='*65}")
    print(f"  Condition       Mean rho    Std")
    print(f"  ----------      --------    ---")
    print(f"  Coherent:       {mean_c:+.4f}    {np.std(coherent_rhos):.4f}  <- domain-structured")
    print(f"  Mixed:          {mean_m:+.4f}    {np.std(mixed_rhos):.4f}  <- cross-domain scrambled")
    print(f"  Noise:          {mean_n:+.4f}    {np.std(noise_rhos):.4f}  <- semantically incoherent")
    print(f"")
    print(f"  Comparisons (t-test, n=5 each):")
    print(f"  Coherent vs Mixed:  t={t_cm:+.2f}  p={p_cm:.3f}  {'SIGNIFICANT' if p_cm < 0.05 else 'n.s.'}")
    print(f"  Mixed vs Noise:     t={t_mn:+.2f}  p={p_mn:.3f}  {'SIGNIFICANT' if p_mn < 0.05 else 'n.s.'}")
    print(f"  Coherent vs Noise:  t={t_cn:+.2f}  p={p_cn:.3f}  {'SIGNIFICANT' if p_cn < 0.05 else 'n.s.'}")
    print(f"")

    gradient_holds = mean_c > mean_m > mean_n
    prediction_confirmed = gradient_holds and p_cm < 0.1

    if prediction_confirmed:
        verdict = "CONFIRMED: RSC detects communication structure. Gradient holds as predicted."
    elif mean_c > mean_n and p_cn < 0.05:
        verdict = "PARTIALLY CONFIRMED: Coherent > Noise is significant but Mixed is ambiguous."
    else:
        verdict = "NOT CONFIRMED: RSC rho does not distinguish communication structure in this test."

    print(f"  Gradient (C>M>N): {'YES' if gradient_holds else 'NO'}")
    print(f"  VERDICT: {verdict}")
    print(f"\n  Total runtime: {time.time()-t0:.0f}s")

    # Save
    save_data = {
        "hypothesis": "RSC rho is a metric for meaningful structured communication",
        "prediction": "Coherent > Mixed > Noise",
        "results": {
            "coherent": {"mean_rho": mean_c, "std": float(np.std(coherent_rhos)), "sets": all_results["coherent"]},
            "mixed":    {"mean_rho": mean_m, "std": float(np.std(mixed_rhos)),    "sets": all_results["mixed"]},
            "noise":    {"mean_rho": mean_n, "std": float(np.std(noise_rhos)),    "sets": all_results["noise"]},
        },
        "comparisons": {
            "coherent_vs_mixed": {"t": float(t_cm), "p": float(p_cm)},
            "mixed_vs_noise":    {"t": float(t_mn), "p": float(p_mn)},
            "coherent_vs_noise": {"t": float(t_cn), "p": float(p_cn)},
        },
        "gradient_holds": gradient_holds,
        "verdict": verdict,
        "runtime_seconds": round(time.time()-t0, 1),
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"  Results saved: {RESULTS_FILE}")


if __name__ == "__main__":
    run()
