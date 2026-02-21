# Semantica RSC — Relational Semantic Convergence

**Theory by Erich Curtis | 2025–2026**

> *Meaning doesn't require identical internal experiences.  
> It emerges from structural relationships between concepts.*

---

## The One-Line Result

Thirteen AI language models — each trained only on its own language, never shown any other —
independently learned the same structural relationships between 1,200 concepts.
**p = 1.46e-15. Family gap p = 0.659. Zero circularity.**

---

## What Is RSC?

**Relational Semantic Convergence** is a theory of meaning: any two systems that share only
*time* and *binary truth* can bootstrap to shared semantic understanding through consistent
structural relationships alone — without shared labels, shared experiences, or shared architecture.

**Core claim:** meaning is structurally inevitable, not biologically special.

Different knowledge systems, governed by the same contradiction-resolution constraints, will
independently evolve to structurally equivalent forms. This is an isomorphism claim — and it is measurable.

---

## Experimental Results

| Version | Method | Concepts | ρ (true) | ρ (baseline) | p-value | Family gap p |
|---------|--------|----------|----------|--------------|---------|--------------|
| v1 | Phonological features (independent) | 14 | +0.059 | -0.003 | 0.0009 | — |
| v2 | LaBSE multilingual embeddings | 39 | +0.620 | +0.001 | <1e-6 | 0.21 |
| v3 | LaBSE + MUSE auto (1,200 concepts) | 1,200 | +0.537 | -0.001 | 1.3e-64 | 0.029 |
| **v4** | **Monolingual BERT, bare word** | **1,200** | **+0.058** | **+0.001** | **4.62e-16** | **0.956** |
| **v5** | **Monolingual BERT, sentence-level** | **1,200** | **+0.062** | **+0.000** | **1.46e-15** | **0.659** |

**v4/v5 are the scientific gold standard.** Each language used a completely separate BERT model
trained only on that language's text. No parallel corpora. No cross-lingual objectives. No shared signal.

---

## Language Coverage (13 languages, 8 families)

| Family | Languages |
|--------|-----------|
| Indo-European | English, German, Spanish, Russian |
| Sino-Tibetan | Mandarin |
| Afro-Asiatic | Arabic, Hebrew |
| Dravidian | Tamil |
| Austronesian | Tagalog, Indonesian |
| Japonic | Japanese |
| Koreanic | Korean |
| Turkic | Turkish |

These families share **no common ancestor within ~15,000 years**. That is the point.

---

## Practical Application: Schema Reconciliation

The most immediate data science application of RSC is **schema matching** — finding which
columns in two different datasets refer to the same underlying concept, using structural position alone.

```bash
python examples/rsc_match.py data/rms_export.csv data/snowflake_schema.csv --top 3 --validate
```

**United Rentals demo (legacy RMS vs Snowflake DW, 18 columns each):**
- **73% accuracy** on known ground truth pairs
- **48% lift** over random baseline

---

## Quick Start

```bash
# Cross-lingual universality experiment (sentence-level, airtight)
python core/rsc_crosslingual_v5.py

# Schema reconciliation CLI
python examples/rsc_match.py <source.csv> <target.csv> [--top N] [--validate]

# United Rentals demo
python examples/demo_united_rentals.py
```

**Requirements:** Python 3.9+, `transformers`, `torch`, `scipy`, `sklearn`, `numpy`, `fugashi`, `unidic-lite`

---

## Project Structure

```
semantica-rsc/
├── core/
│   ├── rsc_engine.py            Schema reconciliation engine
│   ├── rsc_crosslingual.py      v1: phonological (non-circular baseline)
│   ├── rsc_crosslingual_v2.py   v2: LaBSE 39 concepts
│   ├── rsc_crosslingual_v3.py   v3: LaBSE + MUSE auto 1200 concepts
│   ├── rsc_crosslingual_v4.py   v4: monolingual BERT bare-word (airtight)
│   └── rsc_crosslingual_v5.py   v5: monolingual BERT sentence-level
├── data/
│   ├── muse_cache/              Cached MUSE bilingual dictionaries (Facebook Research)
│   ├── v3_results.json          v3 raw results
│   └── v4_results.json          v4 raw results
├── docs/
│   ├── theory.md                Full theoretical statement with math
│   ├── EXECUTIVE_SUMMARY.md     Plain-English summary with all results
│   └── WORKFLOW.md              ASCII workflow diagrams
├── examples/
│   ├── demo_united_rentals.py   United Rentals schema reconciliation demo
│   └── rsc_match.py             CLI schema reconciliation tool
└── README.md
```

---

## Mathematical Formulation

```
F: S × R -> M     where M := Γ(F(S, R)) ⊆ ℝⁿ

S = symbolic input space (concepts, words, schema columns)
R = relational tensor (edge types and weights between concepts)
M = latent meaning manifold
Γ = convergence gradient operator (attracts toward stable states)

Recursive self-inference:
  Ψᵏ(M) ≈ lim_{k→∞} Ψ(Ψ(Ψ(...Ψ(M)...)))

Convergence when:
  ∂M/∂t → 0   (semantic stabilization)
  H(nᵢ) < ε   (node entropy drops below threshold)
```

Full theory: [`docs/theory.md`](docs/theory.md)

---

## Citation

```bibtex
@theory{curtis2025rsc,
  author = {Curtis, Erich},
  title  = {Relational Semantic Convergence: A Structural Approach to Meaning},
  year   = {2025},
  note   = {erichcurtis0@gmail.com}
}
```

