"""Validate a hmm-m6a calls.tsv against the synthetic truth.bed.

Run this after executing the Quick Start command:

    hmm-m6a run --bam example/synthetic.bam --fasta example/synthetic.fa \
                --bed example/synthetic.bed -o calls.tsv \
                --out-bed calls.bed --save-params fitted_params.json

then:

    python example/validate.py calls.tsv

Reports precision, recall, F1, and which sites were missed or wrongly called.
Exits with code 0 if performance is within expected bounds (precision >= 0.9,
recall >= 0.7), 1 otherwise -- so this can be wired into CI.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_TRUTH = HERE / "truth.bed"

# Performance bounds for "the install is working".
# These are loose; the actual numbers should be precision=1.000, recall>=0.80.
MIN_PRECISION = 0.90
MIN_RECALL    = 0.70


def load_truth(path: Path) -> set[tuple[str, int]]:
    truth = pd.read_csv(
        path, sep="\t", header=None,
        names=["contig", "start", "end", "name", "score", "strand"],
    )
    return set(zip(truth["contig"], truth["start"].astype(int)))


def load_calls(path: Path) -> tuple[set[tuple[str, int]], pd.DataFrame]:
    calls = pd.read_csv(path, sep="\t")
    if "call" not in calls.columns:
        raise SystemExit(f"{path} has no 'call' column; was it produced by `hmm-m6a run`?")
    called = set(zip(
        calls.loc[calls["call"], "contig"],
        calls.loc[calls["call"], "pos0"].astype(int),
    ))
    return called, calls


def main() -> int:
    p = argparse.ArgumentParser(description="Validate hmm-m6a calls against truth.")
    p.add_argument("calls_tsv", help="path to calls.tsv from `hmm-m6a run`")
    p.add_argument("--truth", default=str(DEFAULT_TRUTH),
                   help=f"path to truth BED (default: {DEFAULT_TRUTH})")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero unless TP=32 FP=0 FN=7 exactly "
                        "(only meaningful with the bundled seed)")
    args = p.parse_args()

    true_sites = load_truth(Path(args.truth))
    called_sites, calls_df = load_calls(Path(args.calls_tsv))

    tp = len(true_sites & called_sites)
    fp = len(called_sites - true_sites)
    fn = len(true_sites - called_sites)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)

    print("Synthetic-example validation")
    print("-" * 36)
    print(f"truth file:        {args.truth}")
    print(f"calls file:        {args.calls_tsv}")
    print(f"true m6A sites:    {len(true_sites)}")
    print(f"HMM-called sites:  {len(called_sites)}")
    print()
    print(f"TP = {tp}")
    print(f"FP = {fp}")
    print(f"FN = {fn}")
    print(f"precision = {prec:.3f}")
    print(f"recall    = {rec:.3f}")
    print(f"F1        = {f1:.3f}")

    if fp:
        print()
        print("False positives (called sites that are not true m6A):")
        for c, p_ in sorted(called_sites - true_sites):
            row = calls_df[(calls_df["contig"] == c) & (calls_df["pos0"] == p_)].iloc[0]
            print(f"  {c}:{p_}  pM={row['pM']:.3f}  pB={row['pB']:.3f}  "
                  f"n={int(row['n'])}  rate={row['rate']:.2f}  "
                  f"drach={'yes' if row['drach'] else 'no'}")
    if fn:
        print()
        print("False negatives (true m6A sites the model missed):")
        for c, p_ in sorted(true_sites - called_sites):
            sub = calls_df[(calls_df["contig"] == c) & (calls_df["pos0"] == p_)]
            if sub.empty:
                print(f"  {c}:{p_}  not in calls.tsv (filtered out before scoring)")
                continue
            row = sub.iloc[0]
            print(f"  {c}:{p_}  pM={row['pM']:.3f}  pB={row['pB']:.3f}  "
                  f"n={int(row['n'])}  rate={row['rate']:.2f}  "
                  f"drach={'yes' if row['drach'] else 'no'}")

    print()
    if args.strict:
        ok = (tp == 32 and fp == 0 and fn == 7)
        if ok:
            print("OK (strict): exactly matches the reference run.")
            return 0
        print("FAIL (strict): differs from the reference run "
              "(expected TP=32 FP=0 FN=7).")
        return 1

    ok = (prec >= MIN_PRECISION and rec >= MIN_RECALL)
    if ok:
        print(f"OK: precision >= {MIN_PRECISION:.2f} and recall >= {MIN_RECALL:.2f}.")
        return 0
    print(f"FAIL: precision < {MIN_PRECISION:.2f} or recall < {MIN_RECALL:.2f}.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
