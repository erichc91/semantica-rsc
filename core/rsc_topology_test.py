"""
RSC Structural Topology Test — Hierarchical vs Parallel Domains
================================================================
REVISED HYPOTHESIS based on detector v2 finding:

RSC rho does NOT simply detect "domain membership."
RSC rho detects RELATIONAL TOPOLOGY — specifically, whether concepts
have rich structured RELATIONSHIPS between them (not just belong to a category).

PREDICTION (stronger, more specific, more falsifiable):
  Hierarchical/relational domains: rho >> random
    (astronomy: gravity->mass->orbit->planet->star — each concept relates to others)
    (body systems: heart->blood->vessel->organ — functional hierarchy)
    (causation: cause->effect->result->consequence — semantic chain)

  Parallel/instance domains: rho ~ random
    (animals: dog, cat, wolf, eagle — siblings in a category, no inter-relationships)
    (colors: red, blue, green — parallel instances with no mutual relationships)
    (countries: France, Japan, Brazil — category members, not relationally structured)

If confirmed: RSC measures RELATIONAL DENSITY, not category membership.
This is a STRONGER and more theoretically interesting claim than the original.
A "communication structure detector" that specifically detects relational organization.

Uses same infrastructure as v2 (models already loaded, gram matrices cached).
"""

import os, json, time
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

MODEL_CACHE = os.path.join(os.path.dirname(__file__), "..", "model_cache")
MUSE_CACHE  = os.path.join(os.path.dirname(__file__), "..", "data", "muse_cache")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "rsc_topology_results.json")

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

SKIP_WORDS = {"the","a","an","and","or","but","in","on","at","to","for","of","with","by","is","are","it"}

# ---------------------------------------------------------------------------
# HIERARCHICAL/RELATIONAL concept sets — concepts relate TO each other
# Each word in the set has semantic relationships with multiple other words
# ---------------------------------------------------------------------------

HIERARCHICAL = {
    # Causal chain: words relate through causation/mechanism
    "causation":    ["cause","effect","result","reason","consequence","force","action","reaction","trigger","response","origin","change","outcome","purpose","process"],
    # Scale hierarchy: words relate through magnitude ordering
    "scale":        ["small","large","tiny","huge","atom","cell","organ","body","city","planet","galaxy","universe","micro","macro","infinite"],
    # Time structure: words relate through temporal ordering/structure
    "time":         ["moment","second","minute","hour","day","week","month","year","decade","century","past","present","future","begin","end"],
    # Cognitive hierarchy: words relate through cognitive processes
    "cognition":    ["sense","perceive","notice","think","reason","understand","know","learn","remember","forget","imagine","believe","decide","act","result"],
    # Emotional valence spectrum: words are ordered on valence/arousal dimensions
    "valence":      ["joy","happiness","calm","content","neutral","boredom","worry","anxiety","fear","anger","rage","grief","despair","pain","agony"],
    # Social hierarchy: words relate through social structure
    "social":       ["individual","family","group","community","village","city","nation","culture","civilization","leader","follower","law","power","conflict","peace"],
}

# ---------------------------------------------------------------------------
# PARALLEL/INSTANCE concept sets — words are siblings in a category
# Minimal semantic relationships BETWEEN members (just "all are X")
# ---------------------------------------------------------------------------

PARALLEL = {
    # Animals: all are animals, but dog doesn't relate to eagle meaningfully
    "animals":      ["dog","cat","horse","cow","wolf","lion","eagle","snake","fish","bear","rabbit","deer","fox","owl","whale"],
    # Colors: parallel perceptual instances
    "colors":       ["red","blue","green","yellow","black","white","brown","orange","purple","pink","grey","gold","silver","dark","bright"],
    # Tools: parallel functional instances
    "tools":        ["hammer","knife","saw","drill","rope","needle","axe","shovel","key","lock","wheel","nail","chain","hook","lever"],
    # Body parts (isolated): listed as parallel instances, not functional hierarchy
    "body_parts":   ["eye","ear","nose","mouth","hand","foot","arm","leg","knee","elbow","finger","toe","shoulder","ankle","wrist"],
    # Weather: parallel atmospheric states
    "weather":      ["rain","wind","snow","sun","cloud","storm","fog","ice","thunder","lightning","frost","hail","flood","drought","mist"],
    # Minerals/materials: parallel substance instances
    "materials":    ["gold","iron","stone","wood","clay","sand","glass","salt","coal","copper","silver","lead","bone","ash","mud"],
}

# ---------------------------------------------------------------------------
# Reuse encoding infrastructure from v2
# ---------------------------------------------------------------------------

_model_cache_mem = {}

def load_model(lang, model_name, lang_name):
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


def batch_encode(lang, model_name, lang_name, words, batch_size=128):
    import torch
    tok, mdl = load_model(lang, model_name, lang_name)
    all_vecs = []
    for i in range(0, len(words), batch_size):
        batch = words[i:i+batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=16)
        with torch.no_grad():
            out = mdl(**enc)
        vecs = out.last_hidden_state[:, 0, :].numpy().astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        all_vecs.append(vecs / norms)
    return np.vstack(all_vecs)


def load_muse_translations(words):
    non_en = [l for l in MONOLINGUAL_MODELS if l != "en"]
    lang_dicts = {}
    for lang in non_en:
        d = {}
        with open(os.path.join(MUSE_CACHE, f"en-{lang}.txt"), encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].lower() not in d:
                    d[parts[0].lower()] = parts[1]
        lang_dicts[lang] = d

    result = {}
    for w in words:
        entry = {"en": w}
        ok = True
        for lang in non_en:
            t = lang_dicts[lang].get(w)
            if not t:
                ok = False; break
            entry[lang] = t
        if ok:
            result[w] = entry
    return result


def load_pool(pool_size=600):
    non_en = [l for l in MONOLINGUAL_MODELS if l != "en"]
    lang_dicts = {}
    for lang in non_en:
        d = {}
        with open(os.path.join(MUSE_CACHE, f"en-{lang}.txt"), encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    eng = parts[0].lower()
                    if eng not in d and eng not in SKIP_WORDS and eng.isalpha() and len(eng) > 2:
                        d[eng] = parts[1]
        lang_dicts[lang] = d

    coverage = {}
    for d in lang_dicts.values():
        for w in d:
            coverage[w] = coverage.get(w, 0) + 1

    ranked = sorted([w for w, c in coverage.items() if c == len(non_en)],
                    key=lambda w: coverage[w], reverse=True)
    pool_words = ranked[600:600+pool_size]

    pool = {}
    for w in pool_words:
        entry = {"en": w}
        for lang in non_en:
            entry[lang] = lang_dicts[lang][w]
        pool[w] = entry
    return pool


def compute_gram_matrices(words, trans_map):
    grams = {}
    active = []
    for lang, (mname, lname) in MONOLINGUAL_MODELS.items():
        word_list = [trans_map[w][lang] for w in words]
        E = batch_encode(lang, mname, lname, word_list)
        grams[lang] = E @ E.T
        active.append(lang)
    return grams, active


def rho_for_subset(grams, langs, idx):
    n = len(idx)
    if n < 5:
        return float("nan")
    tri = np.triu_indices(n, k=1)
    rhos = []
    for i in range(len(langs)):
        for j in range(i+1, len(langs)):
            a = grams[langs[i]][np.ix_(idx, idx)][tri]
            b = grams[langs[j]][np.ix_(idx, idx)][tri]
            rho, _ = stats.spearmanr(a, b)
            if not np.isnan(rho):
                rhos.append(rho)
    return float(np.mean(rhos)) if rhos else float("nan")


def sample_distribution(grams, langs, n_samples, subset_size, rng, exclude_idx=None):
    n_total = grams[langs[0]].shape[0]
    pool = np.arange(n_total)
    if exclude_idx is not None:
        mask = np.ones(n_total, dtype=bool)
        mask[exclude_idx] = False
        pool = pool[mask]

    rhos = []
    for _ in range(n_samples):
        idx = rng.choice(pool, size=subset_size, replace=False)
        rhos.append(rho_for_subset(grams, langs, idx))
    return np.array(rhos)


def run():
    t0 = time.time()
    print("RSC Structural Topology Test")
    print("=" * 65)
    print("PREDICTION:")
    print("  Hierarchical domains (relational topology) >> Random")
    print("  Parallel domains (category instances)      ~= Random")
    print("If confirmed: RSC measures relational density, not category membership")
    print("=" * 65)

    # Load all concept words
    all_hier = {w for words in HIERARCHICAL.values() for w in words}
    all_para = {w for words in PARALLEL.values() for w in words}
    all_words = list(all_hier | all_para)

    print(f"\n[1] Loading MUSE translations...")
    translations = load_muse_translations(all_words)
    pool = load_pool(600)
    print(f"  Hierarchical words covered: {sum(1 for w in all_hier if w in translations)}/{len(all_hier)}")
    print(f"  Parallel words covered:     {sum(1 for w in all_para if w in translations)}/{len(all_para)}")

    universe = {**pool, **translations}
    universe_words = list(universe.keys())
    idx_map = {w: i for i, w in enumerate(universe_words)}

    print(f"\n[2] Batch encoding {len(universe_words)} concepts, 13 languages...")
    t_enc = time.time()
    grams, active_langs = compute_gram_matrices(universe_words, universe)
    print(f"  Done in {time.time()-t_enc:.0f}s")

    all_concept_idx = np.array([idx_map[w] for w in translations if w in idx_map])
    pool_only_idx = np.array([idx_map[w] for w in pool if w in idx_map and w not in translations])

    rng = np.random.default_rng(42)

    # Compute rho for each domain set
    print(f"\n[3] Computing rho for hierarchical domains...")
    hier_results = []
    for domain, words in HIERARCHICAL.items():
        valid = [w for w in words if w in idx_map]
        if len(valid) < 8: continue
        idx = np.array([idx_map[w] for w in valid])
        rho = rho_for_subset(grams, active_langs, idx)
        hier_results.append({"domain": domain, "type": "hierarchical", "n": len(valid), "rho": rho})
        print(f"  {domain:<12}: n={len(valid):>2}  rho={rho:+.4f}")

    print(f"\n[4] Computing rho for parallel domains...")
    para_results = []
    for domain, words in PARALLEL.items():
        valid = [w for w in words if w in idx_map]
        if len(valid) < 8: continue
        idx = np.array([idx_map[w] for w in valid])
        rho = rho_for_subset(grams, active_langs, idx)
        para_results.append({"domain": domain, "type": "parallel", "n": len(valid), "rho": rho})
        print(f"  {domain:<12}: n={len(valid):>2}  rho={rho:+.4f}")

    # Random distribution: 2000 samples from pool
    print(f"\n[5] Sampling random distribution (2000 samples)...")
    t_s = time.time()
    n_sample = 15
    random_rhos = sample_distribution(grams, active_langs, 2000, n_sample, rng)
    print(f"  Done in {time.time()-t_s:.0f}s")

    hier_rhos = np.array([r["rho"] for r in hier_results])
    para_rhos = np.array([r["rho"] for r in para_results])
    rand_mean = float(np.mean(random_rhos))
    rand_std  = float(np.std(random_rhos))
    rand_p95  = float(np.percentile(random_rhos, 95))

    # Statistical tests
    t_hr, p_hr = stats.ttest_ind(hier_rhos, random_rhos)
    t_pr, p_pr = stats.ttest_ind(para_rhos, random_rhos)
    t_hp, p_hp = stats.ttest_ind(hier_rhos, para_rhos)
    d_hr = (np.mean(hier_rhos) - rand_mean) / (np.std(random_rhos) + 1e-9)
    d_pr = (np.mean(para_rhos) - rand_mean) / (np.std(random_rhos) + 1e-9)

    print(f"\n{'='*65}")
    print(f"RESULTS — RSC STRUCTURAL TOPOLOGY TEST")
    print(f"{'='*65}")
    print(f"\n  {'Condition':<14} {'n_sets':>6}  {'mean rho':>9}  {'std':>6}  {'vs random':>12}")
    print(f"  {'-'*14} {'-'*6}  {'-'*9}  {'-'*6}  {'-'*12}")
    print(f"  {'Hierarchical':<14} {len(hier_rhos):>6}  {np.mean(hier_rhos):>+9.4f}  {np.std(hier_rhos):>6.4f}  d={d_hr:>+6.2f}")
    print(f"  {'Parallel':<14} {len(para_rhos):>6}  {np.mean(para_rhos):>+9.4f}  {np.std(para_rhos):>6.4f}  d={d_pr:>+6.2f}")
    print(f"  {'Random(2000)':<14} {len(random_rhos):>6}  {rand_mean:>+9.4f}  {rand_std:>6.4f}  (baseline)")

    print(f"\n  Individual domain rhos:")
    print(f"  {'Domain':<14} {'Type':<14} {'n':>3}  {'rho':>7}  {'%-ile vs rand':>14}")
    print(f"  {'-'*14} {'-'*14} {'-'*3}  {'-'*7}  {'-'*14}")
    all_domains = sorted(hier_results + para_results, key=lambda x: -x["rho"])
    for r in all_domains:
        pctile = float(np.mean(random_rhos < r["rho"])) * 100
        bar = "#" * int(pctile / 10)
        print(f"  {r['domain']:<14} {r['type']:<14} {r['n']:>3}  {r['rho']:>+7.4f}  {pctile:>12.1f}%  {bar}")

    print(f"\n  Statistical tests:")
    print(f"  Hierarchical vs Random: t={t_hr:+.2f}  p={p_hr:.4f}  d={d_hr:.2f}  {'SIG' if p_hr < 0.05 else 'n.s.'}")
    print(f"  Parallel vs Random:     t={t_pr:+.2f}  p={p_pr:.4f}  d={d_pr:.2f}  {'SIG' if p_pr < 0.05 else 'n.s.'}")
    print(f"  Hierarchical vs Parallel: t={t_hp:+.2f}  p={p_hp:.4f}  {'SIG' if p_hp < 0.05 else 'n.s.'}")

    h_above_rand = np.mean(hier_rhos) > rand_mean
    p_near_rand  = abs(np.mean(para_rhos) - rand_mean) < rand_std
    gradient     = np.mean(hier_rhos) > np.mean(para_rhos) > rand_mean

    if p_hr < 0.05 and p_near_rand:
        verdict = "CONFIRMED: Hierarchical > Random (sig), Parallel ~ Random. RSC = relational topology detector."
    elif gradient:
        verdict = "DIRECTIONAL: gradient holds (H > P > R) but limited by n_domains. More domain sets needed."
    elif h_above_rand and not p_near_rand:
        verdict = "PARTIAL: Hierarchical above random but parallel is also elevated. Topology signal mixed."
    else:
        verdict = "NOT CONFIRMED in this test."

    print(f"\n  Gradient (H > P > R): {'YES' if gradient else 'NO'}")
    print(f"  VERDICT: {verdict}")
    print(f"\n  Total runtime: {time.time()-t0:.0f}s")

    save_data = {
        "version": "rsc_topology_v1",
        "hypothesis": "RSC detects relational topology (not just category membership)",
        "prediction": "Hierarchical domains >> Parallel domains ~ Random",
        "n_languages": len(active_langs),
        "n_random_samples": 2000,
        "hierarchical": {"mean_rho": float(np.mean(hier_rhos)), "std": float(np.std(hier_rhos)),
                         "cohens_d_vs_random": float(d_hr), "p_vs_random": float(p_hr),
                         "domains": hier_results},
        "parallel":     {"mean_rho": float(np.mean(para_rhos)), "std": float(np.std(para_rhos)),
                         "cohens_d_vs_random": float(d_pr), "p_vs_random": float(p_pr),
                         "domains": para_results},
        "random":       {"mean_rho": rand_mean, "std": rand_std, "p95": rand_p95, "n": 2000},
        "tests": {"hier_vs_random": {"t": float(t_hr), "p": float(p_hr)},
                  "para_vs_random": {"t": float(t_pr), "p": float(p_pr)},
                  "hier_vs_para":   {"t": float(t_hp), "p": float(p_hp)}},
        "gradient_holds": bool(gradient),
        "verdict": verdict,
        "runtime_seconds": round(time.time()-t0, 1),
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"  Results saved: {RESULTS_FILE}")


if __name__ == "__main__":
    run()
