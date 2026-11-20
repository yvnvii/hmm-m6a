"""Smoke tests for the package.

Run with `pytest`.

These tests focus on the math (forward-backward correctness, parameter
recovery on synthetic data, calling rule) without requiring a BAM file.
The pileup module is tested separately on a tiny in-memory SAM fixture.
"""
from __future__ import annotations

import numpy as np
import pytest

from hmm_m6a.hmm import HMMParams, emission_logprob, forward_backward, posterior_states
from hmm_m6a.train import TranscriptObs, baum_welch
from hmm_m6a.calling import call_sites
from hmm_m6a.pileup import Region


# ---------------------------------------------------------------------------
# Forward-backward sanity
# ---------------------------------------------------------------------------
def test_forward_backward_matches_brute_force():
    """On a tiny problem, FB must match the brute-force enumeration."""
    rng = np.random.default_rng(0)
    L = 6
    log_pi = np.log(np.array([0.7, 0.2, 0.1]))
    T = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])
    log_T = np.log(T)
    log_e = rng.normal(size=(L, 3))

    _, _, log_gamma, log_Z = forward_backward(log_e, log_pi, log_T)

    # Brute force
    from itertools import product
    log_p_paths = []
    for path in product(range(3), repeat=L):
        lp = log_pi[path[0]] + log_e[0, path[0]]
        for t in range(1, L):
            lp += log_T[path[t-1], path[t]] + log_e[t, path[t]]
        log_p_paths.append(lp)
    from scipy.special import logsumexp
    log_Z_brute = logsumexp(log_p_paths)
    assert abs(log_Z - log_Z_brute) < 1e-10

    # Posteriors normalise to 1 at every position
    gamma = np.exp(log_gamma)
    assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Synthetic GLORI simulator (slim, just for testing parameter recovery)
# ---------------------------------------------------------------------------
def _simulate_transcript(L=800, p_m6a=0.05, artifact_rate=0.04, seed=0):
    r = np.random.default_rng(seed)
    bases = np.array(list("ACGT"))
    seq = r.choice(bases, size=L, p=[0.27, 0.23, 0.23, 0.27])
    A_idx = np.where(seq == "A")[0]

    artifact = np.zeros(L, dtype=bool)
    n_starts = r.poisson(L * artifact_rate / 30.0)
    for _ in range(int(n_starts)):
        s = int(r.integers(0, L)); ln = int(r.integers(15, 60))
        artifact[s:min(s + ln, L)] = True

    truth = np.zeros(L, dtype=bool)
    for i in A_idx:
        if not artifact[i] and r.random() < p_m6a:
            truth[i] = True

    n = np.zeros(L, dtype=int)
    A_count = np.zeros(L, dtype=int)
    drach = np.zeros(L, dtype=bool)  # not used in this toy test
    for i in A_idx:
        n[i] = max(1, int(r.negative_binomial(8, 0.2)))
        if truth[i]:
            mu, kappa = 0.70, 30.0
        elif artifact[i]:
            mu, kappa = 0.25, 25.0
        else:
            mu, kappa = 0.02, 60.0
        a = mu * kappa; b = (1 - mu) * kappa
        p = r.beta(a, b)
        A_count[i] = r.binomial(n[i], p)

    return TranscriptObs(L=L, A_idx=A_idx.astype(np.int32),
                          A_count=A_count.astype(np.int32),
                          n=n.astype(np.int32), drach=drach), truth, artifact


def test_baum_welch_recovers_means():
    """BW from data-naive defaults should recover mu within a few percent."""
    transcripts = []
    for s in range(15):
        tr, _, _ = _simulate_transcript(L=800, seed=s + 1)
        transcripts.append(tr)
    params, _ = baum_welch(transcripts, HMMParams.default(), n_iter=15, verbose=False)
    # state ordering: 0=U, 1=M, 2=B
    assert abs(params.mu[0] - 0.02) < 0.01, f"mu_U off: {params.mu[0]}"
    assert abs(params.mu[1] - 0.70) < 0.05, f"mu_M off: {params.mu[1]}"
    assert abs(params.mu[2] - 0.25) < 0.05, f"mu_B off: {params.mu[2]}"
    # state ordering invariant
    assert params.mu[0] < params.mu[2] < params.mu[1]


def test_calling_rule_applies_constraints():
    """Calling rule must enforce tau, n>=n_min, and P(M) > P(B)."""
    L = 400
    tr, truth, _ = _simulate_transcript(L=L, seed=42)
    params = HMMParams.default()
    region = Region(name="t1", contig="chrTest", start=0, end=L)
    df = call_sites([(region, tr)], params, tau=0.8, n_min=10)
    assert "call" in df.columns
    called = df[df["call"]]
    # All calls must satisfy all three constraints
    assert (called["pM"] >= 0.8).all()
    assert (called["pM"] > called["pB"]).all()
    assert (called["n"] >= 10).all()


def test_hmm_params_serialise_roundtrip():
    p = HMMParams.default()
    d = p.to_dict()
    q = HMMParams.from_dict(d)
    assert np.allclose(p.pi, q.pi)
    assert np.allclose(p.T, q.T)
    assert np.allclose(p.mu, q.mu)
    assert np.allclose(p.kappa, q.kappa)
    assert np.allclose(p.p_drach, q.p_drach)


def test_pileup_strand_handling(tmp_path):
    """Minus-strand regions are complemented, and with a stranded library
    antisense reads are excluded (they would otherwise look unconverted)."""
    import pysam
    from hmm_m6a.pileup import Region, pileup_bam

    ref = "ACGTAATTGCAATTACGGATTACAGTTAACGT" * 4
    fa = tmp_path / "r.fa"
    fa.write_text(">c\n" + ref + "\n")
    pysam.faidx(str(fa))
    hdr = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "c", "LN": len(ref)}]}
    bam = tmp_path / "r.bam"
    with pysam.AlignmentFile(str(bam), "wb", header=hdr) as out:
        k = 0
        # minus-strand RNA, fully converted: genomic T -> C, reverse alignments
        for _ in range(10):
            a = pysam.AlignedSegment(out.header)
            a.query_name = f"m{k}"; k += 1
            a.flag = 16; a.reference_id = 0; a.reference_start = 0
            a.mapping_quality = 60; a.cigar = ((0, len(ref)),)
            a.query_sequence = ref.replace("T", "C")
            a.query_qualities = [40] * len(ref)
            out.write(a)
    pysam.index(str(bam))
    reg = [Region("r", "c", 0, len(ref), "-")]
    # minus-strand region: every transcript A is covered and converted
    (_, tr), = pileup_bam(str(bam), str(fa), reg, show_progress=False,
                          library="single-forward")
    assert tr.n[tr.A_idx].min() == 10 and tr.A_count[tr.A_idx].sum() == 0
    # plus-strand region: these reads are antisense; stranded -> excluded
    reg_p = [Region("r", "c", 0, len(ref), "+")]
    (_, trp), = pileup_bam(str(bam), str(fa), reg_p, show_progress=False,
                           library="single-forward")
    assert trp.n.sum() == 0
    # unstranded pooling counts them as unconverted A (the artefact we avoid)
    (_, tru), = pileup_bam(str(bam), str(fa), reg_p, show_progress=False)
    assert tru.A_count[tru.A_idx].min() == 10


def test_four_state_model_and_ordering():
    """The 4-state model (U, M, B, N) runs end to end, keeps N last after the
    state-ordering step, and assigns a run of fully retained adenosines to N
    rather than M."""
    import numpy as np
    from hmm_m6a.hmm import HMMParams, posterior_states
    from hmm_m6a.train import TranscriptObs, _enforce_state_order

    p = HMMParams.default(n_states=4)
    assert p.n_states == 4 and np.allclose(p.T.sum(axis=1), 1)
    # shuffle states and check the ordering constraint restores U, M, B, N
    perm = np.array([3, 0, 2, 1])
    shuffled = HMMParams(pi=p.pi[perm], T=p.T[perm][:, perm], mu=p.mu[perm],
                         kappa=p.kappa[perm], p_drach=p.p_drach[perm])
    back = _enforce_state_order(shuffled)
    assert np.allclose(back.mu, p.mu)

    L = 200
    A_idx = np.arange(0, L, 4)
    n = np.zeros(L, int); n[A_idx] = 40
    A = np.zeros(L, int); A[A_idx] = 1
    run = A_idx[(A_idx >= 80) & (A_idx < 120)]           # 10 adenosines, all retained
    A[run] = 40
    drach = np.zeros(L, bool)
    g, _ = posterior_states(A_idx, A, n, drach, L, p)
    assert g[run, 3].mean() > 0.9                        # N, not M
    g3, _ = posterior_states(A_idx, A, n, drach, L, HMMParams.default())
    assert g3[run, 1].mean() > 0.5                       # 3-state model calls them M


def test_read_filter_drops_poorly_converted_reads(tmp_path):
    """--max-unconverted removes reads that kept many adenosines."""
    import pysam
    from hmm_m6a.pileup import Region, pileup_bam

    ref = "GCATGACTAGCAGTACAGGATCAGTAGCAATGCAGTCAGAT" * 3
    fa = tmp_path / "r.fa"; fa.write_text(">c\n" + ref + "\n"); pysam.faidx(str(fa))
    hdr = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "c", "LN": len(ref)}]}
    bam = tmp_path / "r.bam"
    with pysam.AlignmentFile(str(bam), "wb", header=hdr) as out:
        for k in range(20):
            a = pysam.AlignedSegment(out.header)
            a.query_name = f"r{k}"; a.flag = 0; a.reference_id = 0; a.reference_start = 0
            a.mapping_quality = 60; a.cigar = ((0, len(ref)),)
            # 5 escaped molecules keep every A; 15 are fully converted
            a.query_sequence = ref if k < 5 else ref.replace("A", "G")
            a.query_qualities = [40] * len(ref)
            out.write(a)
    pysam.index(str(bam))
    reg = [Region("r", "c", 0, len(ref), "+")]
    (_, tr), = pileup_bam(str(bam), str(fa), reg, show_progress=False, library="single-forward")
    assert tr.n[tr.A_idx].min() == 20 and tr.A_count[tr.A_idx].min() == 5
    (_, tf), = pileup_bam(str(bam), str(fa), reg, show_progress=False,
                          library="single-forward", max_unconverted=2)
    assert tf.n[tf.A_idx].min() == 15 and tf.A_count[tf.A_idx].sum() == 0
