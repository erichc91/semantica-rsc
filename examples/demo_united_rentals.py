"""
RSC Schema Reconciliation — United Rentals Demo
================================================
Real scenario: United Rentals runs multiple systems that track the same
equipment rental data with different naming conventions.

Classic problem: Legacy rental management system (RMS) vs modern Snowflake
data warehouse. Two schemas, same underlying business events — the data
engineering team needs to know which columns map to which.

RSC approach: structural relationships alone. No name matching. No crosswalk.

Run:
    python examples/demo_united_rentals.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from core.rsc_engine import rsc_match_schemas, rsc_validate_against_baseline

np.random.seed(42)
N = 600


# ---------------------------------------------------------------------------
# Dataset A: Legacy RMS export (old system naming)
# ---------------------------------------------------------------------------

def make_rms_export(n: int) -> pd.DataFrame:
    """Legacy rental management system — older naming conventions."""
    day_rate = np.random.lognormal(4.2, 0.7, n).clip(20, 2000)
    rental_days = np.random.choice([1, 3, 7, 14, 28, 60, 90, 180], n,
                                    p=[0.15, 0.15, 0.25, 0.2, 0.12, 0.07, 0.04, 0.02])
    revenue = day_rate * rental_days * np.random.uniform(0.85, 1.0, n)

    return pd.DataFrame({
        "order_number":       np.arange(100000, 100000 + n),
        "open_date":          pd.date_range("2022-01-01", periods=n, freq="14H"),
        "close_date":         pd.date_range("2022-01-03", periods=n, freq="14H"),
        "cust_acct_number":   np.random.randint(10000, 99999, n),
        "cust_name":          [f"Customer_{i % 200}" for i in range(n)],
        "branch_number":      np.random.choice([1150, 1151, 1152, 1153, 1154], n),
        "district_number":    np.random.choice([11, 12, 13], n),
        "region_number":      np.random.choice([1, 2, 3], n),
        "equip_cat_code":     np.random.choice(["AE", "EL", "FO", "CO", "CR"], n),
        "equip_class_code":   np.random.choice(["AE01", "EL03", "FO02", "CO01", "CR04"], n),
        "serial_number":      [f"SN{100000 + i}" for i in range(n)],
        "day_rate":           day_rate.round(2),
        "rental_days":        rental_days,
        "total_revenue":      revenue.round(2),
        "on_rent_flag":       np.random.choice([0, 1], n, p=[0.7, 0.3]),
        "damage_waiver_flag": np.random.choice([0, 1], n, p=[0.4, 0.6]),
        "job_site_state":     np.random.choice(["TX", "CA", "FL", "NY", "OH"], n),
        "po_number":          [f"PO{200000 + i}" if np.random.rand() > 0.3 else "" for i in range(n)],
    })


# ---------------------------------------------------------------------------
# Dataset B: Snowflake DW table (modern naming conventions)
# ---------------------------------------------------------------------------

def make_snowflake_table(n: int) -> pd.DataFrame:
    """Modern Snowflake data warehouse — current naming standards."""
    rate = np.random.lognormal(4.2, 0.7, n).clip(20, 2000)
    days = np.random.choice([1, 3, 7, 14, 28, 60, 90, 180], n,
                             p=[0.15, 0.15, 0.25, 0.2, 0.12, 0.07, 0.04, 0.02])
    rev = rate * days * np.random.uniform(0.85, 1.0, n)

    return pd.DataFrame({
        "rental_contract_id": np.arange(900000, 900000 + n),
        "rental_out_dt":      pd.date_range("2022-01-01", periods=n, freq="14H"),
        "rental_return_dt":   pd.date_range("2022-01-03", periods=n, freq="14H"),
        "customer_id":        np.random.randint(10000, 99999, n),
        "customer_name":      [f"Customer_{i % 200}" for i in range(n)],
        "branch_id":          np.random.choice([1150, 1151, 1152, 1153, 1154], n),
        "district_id":        np.random.choice([11, 12, 13], n),
        "region_id":          np.random.choice([1, 2, 3], n),
        "product_category":   np.random.choice(["AE", "EL", "FO", "CO", "CR"], n),
        "product_class":      np.random.choice(["AE01", "EL03", "FO02", "CO01", "CR04"], n),
        "asset_serial_num":   [f"SN{100000 + i}" for i in range(n)],
        "daily_rate_amt":     rate.round(2),
        "days_on_rent":       days,
        "rental_revenue_amt": rev.round(2),
        "is_active_rental":   np.random.choice([0, 1], n, p=[0.7, 0.3]),
        "has_rdp":            np.random.choice([0, 1], n, p=[0.4, 0.6]),
        "job_state_cd":       np.random.choice(["TX", "CA", "FL", "NY", "OH"], n),
        "purchase_order_num": [f"PO{200000 + i}" if np.random.rand() > 0.3 else "" for i in range(n)],
    })


# ---------------------------------------------------------------------------
# Ground truth crosswalk (what a human data engineer would manually build)
# ---------------------------------------------------------------------------

KNOWN_MATCHES = [
    ("order_number",       "rental_contract_id"),
    ("open_date",          "rental_out_dt"),
    ("close_date",         "rental_return_dt"),
    ("cust_acct_number",   "customer_id"),
    ("branch_number",      "branch_id"),
    ("district_number",    "district_id"),
    ("region_number",      "region_id"),
    ("equip_cat_code",     "product_category"),
    ("equip_class_code",   "product_class"),
    ("day_rate",           "daily_rate_amt"),
    ("rental_days",        "days_on_rent"),
    ("total_revenue",      "rental_revenue_amt"),
    ("on_rent_flag",       "is_active_rental"),
    ("damage_waiver_flag", "has_rdp"),
    ("job_site_state",     "job_state_cd"),
]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("RSC Theory — Schema Reconciliation Demo")
    print("United Rentals: Legacy RMS Export vs Snowflake Data Warehouse")
    print("=" * 65)

    df_rms = make_rms_export(N)
    df_snow = make_snowflake_table(N)

    matches = rsc_match_schemas(
        df_rms, df_snow,
        name_a="Legacy RMS Export",
        name_b="Snowflake DW Table",
        threshold=0.55,
        top_k=2,
    )

    if matches.empty:
        print("No matches found.")
        return

    best = matches[matches["rank"] == 1].copy()

    print("RSC Structural Matches (no name comparison used):")
    print("-" * 65)
    for _, row in best.iterrows():
        hint = f"  [{row['label_hint']}]" if row["label_hint"] else ""
        print(
            f"  {row['col_a']:25} -> {row['col_b']:25} "
            f"sim={row['similarity']:.3f} [{row['confidence']}]{hint}"
        )

    print("\nValidation Against Known Correct Pairs:")
    print("-" * 65)
    correct = 0
    for col_a, col_b_true in KNOWN_MATCHES:
        predicted_row = best[best["col_a"] == col_a]
        if not predicted_row.empty:
            predicted_b = predicted_row.iloc[0]["col_b"]
            hit = predicted_b == col_b_true
            correct += int(hit)
            status = "CORRECT" if hit else f"WRONG (got {predicted_b})"
            sim = predicted_row.iloc[0]["similarity"]
            print(f"  {col_a:25} -> expected {col_b_true:22} | {status} (sim={sim:.3f})")
        else:
            print(f"  {col_a:25} -> expected {col_b_true:22} | NOT MATCHED")

    accuracy = correct / len(KNOWN_MATCHES)
    print(f"\n  Match accuracy: {correct}/{len(KNOWN_MATCHES)} = {accuracy:.0%}")

    print("\nBaseline Validation (RSC vs Random Pairing):")
    print("-" * 65)
    validation = rsc_validate_against_baseline(df_rms, df_snow, KNOWN_MATCHES, n_shuffles=300)
    print(f"  RSC score on known matches: {validation['rsc_score']}")
    print(f"  Random baseline score:      {validation['baseline_score']}")
    print(f"  Lift over baseline:         {validation['lift']:.1%}")
    print(f"  Signal detected:            {validation['signal_detected']}")

    if validation["signal_detected"]:
        print("\n  RSC is detecting real structural signal above random baseline.")
        print("  Structural position identifies concepts without name matching.")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
