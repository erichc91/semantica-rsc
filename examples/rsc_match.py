"""
RSC Schema Match CLI
====================
Command-line tool for RSC schema reconciliation.

Usage:
    python examples/rsc_match.py source.csv target.csv
    python examples/rsc_match.py source.csv target.csv --top 5
    python examples/rsc_match.py source.csv target.csv --out mapping.csv

Practical use case (United Rentals example):
    python examples/rsc_match.py rms_export.csv snowflake_schema.csv

Output: A mapping table showing which source columns match which target columns,
with confidence scores. No column names used in matching — pure structural RSC.
"""

import argparse
import sys
import os

import pandas as pd
import numpy as np

# Allow running from project root or examples/ dir
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.rsc_engine import rsc_match_schemas, rsc_validate_against_baseline


def main():
    parser = argparse.ArgumentParser(
        description="RSC Schema Reconciliation — match columns across two CSVs "
                    "using structural similarity (no name matching).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/rsc_match.py data/rms.csv data/snowflake.csv
  python examples/rsc_match.py data/rms.csv data/snowflake.csv --top 3
  python examples/rsc_match.py data/rms.csv data/snowflake.csv --out mapping.csv --min-score 0.4
""",
    )
    parser.add_argument("source", help="Source CSV file (e.g., legacy export)")
    parser.add_argument("target", help="Target CSV file (e.g., new schema)")
    parser.add_argument("--top", type=int, default=3,
                        help="Top N candidate matches per source column (default: 3)")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Minimum confidence score to include (0.0-1.0, default: 0.0)")
    parser.add_argument("--out", type=str, default=None,
                        help="Save mapping table to CSV file")
    parser.add_argument("--validate", action="store_true",
                        help="Run shuffle baseline validation (adds ~5 sec)")
    parser.add_argument("--no-header", action="store_true",
                        help="Suppress header block in output")
    args = parser.parse_args()

    # --- Load data ---
    try:
        source_df = pd.read_csv(args.source)
    except Exception as e:
        print(f"ERROR reading source file '{args.source}': {e}")
        sys.exit(1)

    try:
        target_df = pd.read_csv(args.target)
    except Exception as e:
        print(f"ERROR reading target file '{args.target}': {e}")
        sys.exit(1)

    if not args.no_header:
        print()
        print("=" * 64)
        print("  RSC Schema Reconciliation")
        print("  Relational Semantic Convergence — structural matching only")
        print("=" * 64)
        print(f"  Source: {args.source}  ({len(source_df):,} rows x {len(source_df.columns)} cols)")
        print(f"  Target: {args.target}  ({len(target_df):,} rows x {len(target_df.columns)} cols)")
        print()

    # --- Run RSC matching ---
    matches = rsc_match_schemas(source_df, target_df, top_k=args.top)

    if matches.empty:
        print(f"No matches found above minimum score {args.min_score:.2f}")
        sys.exit(0)

    if args.min_score > 0:
        matches = matches[matches["similarity"] >= args.min_score].copy()

    # --- Display ---
    _print_mapping_table(matches)

    # --- Validation (optional) ---
    if args.validate:
        # Use top-1 match per source column as "known" matches for baseline comparison
        top1 = matches[matches["rank"] == 1]
        known = list(zip(top1["col_a"], top1["col_b"]))
        result = rsc_validate_against_baseline(source_df, target_df,
                                               known_matches=known,
                                               n_shuffles=200)
        print()
        print("Shuffle Baseline Validation:")
        print("-" * 48)
        lift = result.get("lift", 0)
        signal = result.get("signal_detected", False)
        rsc_score = result.get("rsc_score", 0)
        baseline = result.get("baseline_score", 0)
        print(f"  RSC top-1 match score:      {rsc_score:.4f}")
        print(f"  Random baseline score:      {baseline:.4f}")
        print(f"  Lift:                       {lift:+.4f}")
        sig_str = "SIGNIFICANT" if signal else "not significant"
        print(f"  Signal: {sig_str}")
        print()
        if signal:
            print("  RSC is finding real structural signal in this schema pair.")
        else:
            print("  Signal is weak — schemas may be too similar or too dissimilar")
            print("  structurally. More data rows may improve results.")

    # --- Save (optional) ---
    if args.out:
        matches.to_csv(args.out, index=False)
        print(f"\n  Saved to: {args.out}")


def _score_label(score: float) -> str:
    # not used — kept for potential future use
    if score >= 0.75:
        return "HIGH"
    elif score >= 0.5:
        return "MEDIUM"
    elif score >= 0.3:
        return "LOW"
    else:
        return "WEAK"


def _print_mapping_table(result_df: pd.DataFrame):
    """Print a clean mapping table grouped by source column."""
    print(f"{'SOURCE COLUMN':<30}  {'TARGET MATCH':<30}  {'SCORE':>7}  CONFIDENCE  LABEL HINT")
    print("-" * 90)

    current_src = None
    for _, row in result_df.iterrows():
        src = row["col_a"]
        if src != current_src:
            if current_src is not None:
                print()
            current_src = src

        score = row["similarity"]
        score_bar = "#" * int(score * 10)
        hint = f"  [{row['label_hint']}]" if row.get("label_hint") else ""

        print(f"  {src:<28}  {row['col_b']:<30}  {score:>6.3f}  "
              f"{row['confidence']:<8}{hint}")

    print()
    print(f"  {len(result_df)} matches | "
          f"{result_df['col_a'].nunique()} source cols | "
          f"{result_df['col_b'].nunique()} unique target cols matched")


if __name__ == "__main__":
    main()
