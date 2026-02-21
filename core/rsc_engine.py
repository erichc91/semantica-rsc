"""
RSC Engine — Relational Semantic Convergence
============================================
Core implementation of RSC theory for practical data science use.

Theory (Curtis, 2025):
    Meaning emerges from structural relationships between concepts —
    not from shared labels, shared experiences, or shared internal architecture.
    Two systems that independently organize knowledge under the same structural
    constraints will converge toward equivalent semantic forms.

Practical application here:
    Schema reconciliation — finding which columns/concepts in two different
    datasets refer to the same underlying thing, using structure alone.
    No lookup tables. No name matching. Pure relational convergence.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Structural Profile
# ---------------------------------------------------------------------------

def build_structural_profile(df: pd.DataFrame) -> Dict[str, dict]:
    """
    Build a structural profile for each column in a DataFrame.

    RSC principle: each concept's meaning is defined by its relationships
    to other concepts — not its label. This profile captures:
      - Distribution shape (not values, shape)
      - Correlation neighborhood (which other columns move with it)
      - Centrality (how connected it is to the rest of the schema)
      - Entropy (information density)

    Args:
        df: Input DataFrame

    Returns:
        Dict mapping column name -> structural profile dict
    """
    profiles = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()

    # Build pairwise correlation matrix for numeric cols
    corr_matrix = {}
    if len(numeric_cols) >= 2:
        corr_df = df[numeric_cols].corr(method="spearman").fillna(0)
        corr_matrix = corr_df.to_dict()

    for col in all_cols:
        series = df[col].dropna()
        profile = {}

        # --- Distribution shape ---
        if col in numeric_cols:
            profile["dtype_class"] = "numeric"
            profile["mean_normalized"] = float(series.mean()) if len(series) > 0 else 0
            profile["std_normalized"] = float(series.std()) if len(series) > 0 else 0
            profile["skew"] = float(series.skew()) if len(series) > 0 else 0
            profile["kurtosis"] = float(series.kurt()) if len(series) > 0 else 0
            # Normalize to 0-1 range for comparability
            min_v, max_v = series.min(), series.max()
            if max_v != min_v:
                normalized = (series - min_v) / (max_v - min_v)
                profile["percentiles"] = list(
                    np.percentile(normalized, [10, 25, 50, 75, 90])
                )
            else:
                profile["percentiles"] = [0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            profile["dtype_class"] = "categorical"
            profile["skew"] = 0.0
            profile["kurtosis"] = 0.0
            profile["percentiles"] = [0.0, 0.0, 0.0, 0.0, 0.0]

        # --- Null density ---
        profile["null_rate"] = float(df[col].isna().mean())

        # --- Cardinality ratio (uniqueness) ---
        profile["cardinality_ratio"] = float(df[col].nunique() / max(len(df), 1))

        # --- Entropy (information density) ---
        value_counts = series.value_counts(normalize=True)
        entropy = float(-np.sum(value_counts * np.log2(value_counts + 1e-10)))
        profile["entropy"] = entropy

        # --- Correlation neighborhood ---
        # Which columns is this column most related to? (structural position)
        if col in numeric_cols and corr_matrix:
            neighbors = {}
            for other_col in numeric_cols:
                if other_col != col and other_col in corr_matrix.get(col, {}):
                    neighbors[other_col] = abs(corr_matrix[col][other_col])
            # Store the sorted correlation strengths (not names — position matters, not label)
            neighbor_strengths = sorted(neighbors.values(), reverse=True)
            profile["neighbor_strengths"] = neighbor_strengths[:5]  # top 5
            profile["max_correlation"] = neighbor_strengths[0] if neighbor_strengths else 0.0
            profile["mean_correlation"] = float(np.mean(neighbor_strengths)) if neighbor_strengths else 0.0
        else:
            profile["neighbor_strengths"] = []
            profile["max_correlation"] = 0.0
            profile["mean_correlation"] = 0.0

        # --- Centrality (how many strong correlations this column has) ---
        strong_correlations = sum(
            1 for s in profile["neighbor_strengths"] if s > 0.3
        )
        profile["centrality"] = strong_correlations

        profiles[col] = profile

    return profiles


# ---------------------------------------------------------------------------
# Structural Similarity
# ---------------------------------------------------------------------------

def structural_similarity(profile_a: dict, profile_b: dict) -> float:
    """
    Compute structural similarity between two column profiles.

    RSC principle: two concepts are semantically equivalent if their
    structural positions are equivalent — same relationships to neighbors,
    same information shape, same role in the schema.

    Does NOT compare names. Does NOT require same domain.
    Compares shape, entropy, correlation neighborhood, cardinality.

    Returns:
        float in [0, 1], where 1.0 = structurally identical
    """
    score = 0.0
    weight_total = 0.0

    # --- dtype class match (must both be numeric or both categorical) ---
    w = 2.0
    if profile_a.get("dtype_class") == profile_b.get("dtype_class"):
        score += w
    weight_total += w

    # --- Null rate similarity ---
    w = 1.0
    null_diff = abs(profile_a.get("null_rate", 0) - profile_b.get("null_rate", 0))
    score += w * (1.0 - min(null_diff, 1.0))
    weight_total += w

    # --- Cardinality ratio similarity ---
    w = 1.5
    card_a = profile_a.get("cardinality_ratio", 0)
    card_b = profile_b.get("cardinality_ratio", 0)
    card_diff = abs(card_a - card_b)
    score += w * (1.0 - min(card_diff, 1.0))
    weight_total += w

    # --- Entropy similarity ---
    w = 2.0
    ent_a = profile_a.get("entropy", 0)
    ent_b = profile_b.get("entropy", 0)
    max_ent = max(ent_a, ent_b, 1.0)
    ent_sim = 1.0 - abs(ent_a - ent_b) / max_ent
    score += w * ent_sim
    weight_total += w

    # --- Distribution shape (percentiles) ---
    w = 3.0
    perc_a = profile_a.get("percentiles", [])
    perc_b = profile_b.get("percentiles", [])
    if perc_a and perc_b and len(perc_a) == len(perc_b):
        perc_diff = np.mean([abs(a - b) for a, b in zip(perc_a, perc_b)])
        score += w * (1.0 - min(perc_diff, 1.0))
        weight_total += w

    # --- Skew similarity ---
    w = 1.0
    skew_a = profile_a.get("skew", 0)
    skew_b = profile_b.get("skew", 0)
    # Both right-skewed, both left-skewed, etc.
    if skew_a * skew_b >= 0:  # same sign
        skew_diff = abs(skew_a - skew_b) / (max(abs(skew_a), abs(skew_b), 1.0))
        score += w * (1.0 - min(skew_diff, 1.0))
    weight_total += w

    # --- Correlation neighborhood similarity ---
    w = 3.0
    neigh_a = sorted(profile_a.get("neighbor_strengths", []), reverse=True)[:3]
    neigh_b = sorted(profile_b.get("neighbor_strengths", []), reverse=True)[:3]
    if neigh_a and neigh_b:
        # Pad to same length
        max_len = max(len(neigh_a), len(neigh_b))
        neigh_a = neigh_a + [0.0] * (max_len - len(neigh_a))
        neigh_b = neigh_b + [0.0] * (max_len - len(neigh_b))
        neigh_diff = np.mean([abs(a - b) for a, b in zip(neigh_a, neigh_b)])
        score += w * (1.0 - min(neigh_diff, 1.0))
        weight_total += w

    # --- Centrality similarity ---
    w = 1.5
    cent_a = profile_a.get("centrality", 0)
    cent_b = profile_b.get("centrality", 0)
    max_cent = max(cent_a, cent_b, 1)
    score += w * (1.0 - abs(cent_a - cent_b) / max_cent)
    weight_total += w

    return score / weight_total if weight_total > 0 else 0.0


# ---------------------------------------------------------------------------
# RSC Convergence — Main Matching Engine
# ---------------------------------------------------------------------------

def rsc_match_schemas(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    name_a: str = "Dataset A",
    name_b: str = "Dataset B",
    threshold: float = 0.6,
    top_k: int = 3,
) -> pd.DataFrame:
    """
    RSC schema reconciliation: find which columns in df_a correspond to
    which columns in df_b, using structural relationships alone.

    RSC principle applied: two columns that play the same structural role
    in their respective datasets — same entropy, same correlation neighborhood,
    same distribution shape, same centrality — are likely the same concept,
    regardless of what they're called.

    Args:
        df_a:      First DataFrame
        df_b:      Second DataFrame
        name_a:    Label for first dataset (display only)
        name_b:    Label for second dataset (display only)
        threshold: Minimum similarity score to include in results (0-1)
        top_k:     Max candidate matches per column to return

    Returns:
        DataFrame with columns:
          col_a, col_b, similarity, confidence, note
    """
    print(f"\nRSC Schema Reconciliation")
    print(f"  {name_a}: {len(df_a.columns)} columns, {len(df_a):,} rows")
    print(f"  {name_b}: {len(df_b.columns)} columns, {len(df_b):,} rows")
    print(f"  Building structural profiles...")

    profiles_a = build_structural_profile(df_a)
    profiles_b = build_structural_profile(df_b)

    print(f"  Computing structural similarities...")

    results = []
    for col_a, prof_a in profiles_a.items():
        scores = []
        for col_b, prof_b in profiles_b.items():
            sim = structural_similarity(prof_a, prof_b)
            scores.append((col_b, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_matches = scores[:top_k]

        for rank, (col_b, sim) in enumerate(top_matches):
            if sim >= threshold:
                # Confidence: how much better is this match vs the next?
                next_sim = scores[rank + 1][1] if rank + 1 < len(scores) else 0.0
                gap = sim - next_sim
                confidence = "HIGH" if gap > 0.15 else "MEDIUM" if gap > 0.05 else "LOW"

                # Note any label hints (not used for matching, just FYI)
                label_hint = ""
                a_clean = col_a.lower().replace("_", "").replace(" ", "")
                b_clean = col_b.lower().replace("_", "").replace(" ", "")
                if a_clean == b_clean:
                    label_hint = "exact name match"
                elif a_clean in b_clean or b_clean in a_clean:
                    label_hint = "partial name match"

                results.append({
                    "col_a": col_a,
                    "col_b": col_b,
                    "similarity": round(sim, 3),
                    "confidence": confidence,
                    "rank": rank + 1,
                    "label_hint": label_hint,
                })

    result_df = pd.DataFrame(results)
    if result_df.empty:
        print(f"  No matches found above threshold {threshold}")
        return result_df

    result_df = result_df.sort_values(
        ["col_a", "similarity"], ascending=[True, False]
    ).reset_index(drop=True)

    # Summary
    high = (result_df["confidence"] == "HIGH").sum()
    med = (result_df["confidence"] == "MEDIUM").sum()
    print(f"  Done. {high} HIGH confidence, {med} MEDIUM confidence matches found.\n")

    return result_df


# ---------------------------------------------------------------------------
# Baseline: random shuffle (to validate RSC is doing real work)
# ---------------------------------------------------------------------------

def rsc_baseline_shuffle(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    n_shuffles: int = 100,
    seed: int = 42,
) -> float:
    """
    Compute expected similarity score under random column assignment.

    If RSC's structural matching scores are significantly higher than
    this baseline, the signal is real — not just noise.

    Returns:
        Mean similarity score for random column pairings
    """
    rng = np.random.default_rng(seed)
    profiles_a = build_structural_profile(df_a)
    profiles_b = build_structural_profile(df_b)

    cols_a = list(profiles_a.keys())
    cols_b = list(profiles_b.keys())

    random_scores = []
    for _ in range(n_shuffles):
        col_a = rng.choice(cols_a)
        col_b = rng.choice(cols_b)
        sim = structural_similarity(profiles_a[col_a], profiles_b[col_b])
        random_scores.append(sim)

    return float(np.mean(random_scores))


def rsc_validate_against_baseline(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    known_matches: List[Tuple[str, str]],
    n_shuffles: int = 200,
) -> dict:
    """
    Validate RSC matching quality against a random baseline.

    Args:
        df_a:           First DataFrame
        df_b:           Second DataFrame
        known_matches:  List of (col_a, col_b) pairs known to be the same concept
        n_shuffles:     Number of random pairs for baseline

    Returns:
        Dict with: rsc_score, baseline_score, lift, signal_detected
    """
    profiles_a = build_structural_profile(df_a)
    profiles_b = build_structural_profile(df_b)

    # RSC score on known correct pairs
    rsc_scores = []
    for col_a, col_b in known_matches:
        if col_a in profiles_a and col_b in profiles_b:
            sim = structural_similarity(profiles_a[col_a], profiles_b[col_b])
            rsc_scores.append(sim)

    rsc_mean = float(np.mean(rsc_scores)) if rsc_scores else 0.0
    baseline_mean = rsc_baseline_shuffle(df_a, df_b, n_shuffles)
    lift = (rsc_mean - baseline_mean) / max(baseline_mean, 0.001)

    return {
        "rsc_score": round(rsc_mean, 3),
        "baseline_score": round(baseline_mean, 3),
        "lift": round(lift, 3),
        "signal_detected": rsc_mean > baseline_mean * 1.1,
        "n_known_matches": len(rsc_scores),
    }
