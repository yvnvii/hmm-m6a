"""End-to-end test: run the full CLI pipeline on the synthetic example
and validate the call output against the bundled truth BED.

This is the integration test users can run to confirm the package works
correctly on their machine. The unit tests in test_hmm.py cover the math;
this file covers BAM I/O, the CLI plumbing, and the calling rule together.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "example"


pytestmark = pytest.mark.skipif(
    not (EXAMPLE / "synthetic.bam").exists(),
    reason="example/synthetic.bam missing",
)


def _read_truth() -> set[tuple[str, int]]:
    truth = pd.read_csv(
        EXAMPLE / "truth.bed", sep="\t", header=None,
        names=["contig", "start", "end", "name", "score", "strand"],
    )
    return set(zip(truth["contig"], truth["start"].astype(int)))


def _read_calls(p: Path) -> tuple[set[tuple[str, int]], pd.DataFrame]:
    df = pd.read_csv(p, sep="\t")
    called = set(zip(df.loc[df["call"], "contig"],
                     df.loc[df["call"], "pos0"].astype(int)))
    return called, df


def test_cli_run_on_synthetic_example(tmp_path: Path):
    """Smoke test: hmm-m6a run on the bundled example produces sensible output."""
    out_tsv = tmp_path / "calls.tsv"
    out_bed = tmp_path / "calls.bed"
    params  = tmp_path / "fitted_params.json"

    cmd = [
        sys.executable, "-m", "hmm_m6a.cli", "-q", "run",
        "--bam",          str(EXAMPLE / "synthetic.bam"),
        "--fasta",        str(EXAMPLE / "synthetic.fa"),
        "--bed",          str(EXAMPLE / "synthetic.bed"),
        "-o",             str(out_tsv),
        "--out-bed",      str(out_bed),
        "--save-params",  str(params),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, (
        f"CLI exited {res.returncode}\nstdout: {res.stdout}\nstderr: {res.stderr}"
    )

    # All four output files exist and are non-empty
    for f in [out_tsv, out_bed, params]:
        assert f.exists(), f"missing output: {f}"
        assert f.stat().st_size > 0, f"empty output: {f}"

    # Per-site TSV must have all required columns
    calls_df = pd.read_csv(out_tsv, sep="\t")
    for col in ["region", "contig", "pos0", "n", "A", "rate",
                "drach", "pU", "pM", "pB", "call"]:
        assert col in calls_df.columns, f"missing column: {col}"

    # All probabilities must be in [0, 1] and sum to ~1
    for col in ["pU", "pM", "pB"]:
        assert ((calls_df[col] >= 0) & (calls_df[col] <= 1)).all(), \
            f"out-of-range posteriors in {col}"
    assert ((calls_df[["pU", "pM", "pB"]].sum(axis=1) - 1).abs() < 1e-6).all(), \
        "posteriors do not sum to 1"

    # Saved parameters must be loadable and well-formed
    pj = json.loads(params.read_text())
    assert "params" in pj
    learned = pj["params"]
    assert sorted(learned.keys()) == ["T", "kappa", "mu", "p_drach", "pi"]


def test_synthetic_example_meets_performance_bounds(tmp_path: Path):
    """The synthetic example is calibrated so the model should reach
    precision >= 0.9 and recall >= 0.7. If those bounds fail, something
    is wrong with the install."""
    out_tsv = tmp_path / "calls.tsv"
    cmd = [
        sys.executable, "-m", "hmm_m6a.cli", "-q", "run",
        "--bam",   str(EXAMPLE / "synthetic.bam"),
        "--fasta", str(EXAMPLE / "synthetic.fa"),
        "--bed",   str(EXAMPLE / "synthetic.bed"),
        "-o",      str(out_tsv),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    truth = _read_truth()
    called, _ = _read_calls(out_tsv)
    tp = len(truth & called)
    fp = len(called - truth)
    fn = len(truth - called)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)

    assert prec >= 0.90, f"precision {prec:.3f} < 0.90"
    assert rec  >= 0.70, f"recall {rec:.3f} < 0.70"


def test_synthetic_example_exact_reference_run(tmp_path: Path):
    """Tighter check: with the bundled BAM, FASTA, and seeded defaults,
    the call set should match the reference numbers exactly."""
    out_tsv = tmp_path / "calls.tsv"
    cmd = [
        sys.executable, "-m", "hmm_m6a.cli", "-q", "run",
        "--bam",   str(EXAMPLE / "synthetic.bam"),
        "--fasta", str(EXAMPLE / "synthetic.fa"),
        "--bed",   str(EXAMPLE / "synthetic.bed"),
        "-o",      str(out_tsv),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    truth = _read_truth()
    called, _ = _read_calls(out_tsv)
    tp = len(truth & called)
    fp = len(called - truth)
    fn = len(truth - called)

    # Reference: TP=32, FP=0, FN=7
    assert (tp, fp, fn) == (32, 0, 7), (
        f"Got (TP, FP, FN) = ({tp}, {fp}, {fn}); expected (32, 0, 7). "
        "If you've changed any seeds or numerics this is informational only "
        "-- delete this test if intentional."
    )
