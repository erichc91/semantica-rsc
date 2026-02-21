# RSC — Workflow Diagrams

**Relational Semantic Convergence**
Erich Curtis | 2025–2026

---

## 1. The Core Theory

```
Any two systems that share:
  [Time] + [Binary Truth]
        |
        v
  Can derive basic logic
        |
        v
  Can build relational concept graphs
        |
        v
  Relational structure is constrained
  by the physical/logical world
        |
        v
  Independent systems converge
  to structurally equivalent forms
        |
        v
  [Meaning is structurally inevitable]
```

---

## 2. The Experimental Design (v4/v5 — Airtight)

```
  MUSE Bilingual Dictionaries (Facebook Research, open source)
  en-de.txt  en-es.txt  en-ru.txt  en-zh.txt  ... (12 files)
       |
       v
  Find 1,200 English concepts with
  full translations in all 12 languages
       |
       +--------------------------------------------------+
       |                                                  |
       v                                                  v
  For each of 13 languages:                        SHUFFLE BASELINE
  load its own BERT model                          (same embeddings,
  (trained ONLY on that language)                   random concept labels)
       |
       v
  Encode each concept in a sentence:
  "I use [word] every day."
  (in each language's own script)
       |
       v
  Build 1200x1200 pairwise cosine
  similarity matrix (per language)
       |
       v
  Compare matrices ACROSS languages:
  Spearman rho on upper triangle
  (78 language pairs)
       |
       +---------------------------+
       v                           v
  True mean rho              Shuffled mean rho
  +0.062                     +0.000
       |                           |
       +---------------------------+
                   |
                   v
          t-test: p = 1.46e-15
          Family gap: p = 0.659
                   |
                   v
    [Universal structural signal confirmed]
    [Zero circularity — models never shared training]
    [Ancestry explains nothing]
```

---

## 3. Language Coverage

```
  8 Language Families:

  Indo-European  +--> English, German, Spanish, Russian
  Sino-Tibetan   +--> Mandarin
  Afro-Asiatic   +--> Arabic, Hebrew
  Dravidian      +--> Tamil
  Austronesian   +--> Tagalog, Indonesian
  Japonic        +--> Japanese
  Koreanic       +--> Korean
  Turkic         +--> Turkish

  These families share NO common ancestor
  within the last ~15,000 years.
  That is the point.
```

---

## 4. Experiment Version Progression

```
  v1: Phonological features only
      (syllable count, tone, consonant clusters)
      14 concepts, 13 languages
      rho=+0.059  p=0.0009
      CIRCULAR? No. Fully independent features.
           |
           v
  v2: LaBSE embeddings (multilingual model)
      39 concepts, 13 languages
      rho=+0.620  p<1e-6
      CIRCULAR? Slight — LaBSE trained on parallel text.
      Shuffle baseline controls for it but concern remains.
           |
           v
  v3: LaBSE + MUSE auto-download (1200 concepts)
      1,200 concepts, 13 languages
      rho=+0.537  p=1.3e-64
      CIRCULAR? Same as v2 — LaBSE concern.
           |
           v
  v4: Separate monolingual BERT per language (bare word)
      1,200 concepts, 13 languages, 8 families
      rho=+0.058  p=4.62e-16   family gap p=0.956
      CIRCULAR? NO. Zero shared training.
           |
           v
  v5: Separate monolingual BERT per language (sentence)
      1,200 concepts, 13 languages, 8 families
      rho=+0.062  p=1.46e-15   family gap p=0.659
      CIRCULAR? NO. Same airtight design + better encoding.
           |
           v
      [DONE — universal structural signal confirmed]
```

---

## 5. Schema Reconciliation Pipeline

```
  Source Dataset                Target Dataset
  (RMS export)                  (Snowflake DW)
       |                              |
       v                              v
  For each column:              For each column:
  - Value distribution          - Value distribution
  - Entropy                     - Entropy
  - Cardinality                 - Cardinality
  - Null rate                   - Null rate
  - Correlation to others       - Correlation to others
  - PageRank centrality         - PageRank centrality
       |                              |
       +----------+  +----------------+
                  |  |
                  v  v
           RSC Engine
           Structural similarity
           between all column pairs
                  |
                  v
           Ranked match table:
           col_a          col_b            score  confidence
           day_rate  <->  daily_rate_amt   0.87   high
           rental_days <> days_on_rent     0.81   high
           open_date  <-> start_date       0.74   medium
                  |
                  v
           Shuffle validation:
           73% accuracy vs known truth
           +48% lift over random
```

---

## 6. What Comes Next

```
  Current state:
  p = 1.46e-15  rho = +0.062  n_concepts = 1200

  To reach pre-registration level:
       |
       v
  Expand to 5,000+ concepts
  (MUSE has coverage for this)
       |
       v
  Pre-register hypothesis on OSF
  (lock the design before final run)
       |
       v
  Final airtight run
       |
       v
  Write-up: technical blog post
  or arXiv preprint
```
