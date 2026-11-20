"""Build per-adenosine observation tuples from a coordinate-sorted BAM and a
reference FASTA.

GLORI semantics
---------------
GLORI converts unmethylated A -> I, which is read as G during sequencing.
A read covering a reference adenosine is therefore informative if the
aligned base is either A (retention; possible m6A or unconverted background)
or G (successful conversion). At each reference adenosine position t we
count:

    A_t = number of reads with aligned base A
    G_t = number of reads with aligned base G
    n_t = A_t + G_t                       (informative coverage)

Reads whose aligned base is C, T, an indel, or a skip are not counted.
This matches the convention in the original GLORI calling pipeline.

We also compute the DRACH-motif flag from the *reference* 5-mer centred on
each adenosine, using the standard:
    D in {A, G, T},  R in {A, G},  A,  C,  H in {A, C, T}.

Strand handling
---------------
Each region is scored on its own strand: for "-" regions the reference and
the read bases are complemented, so counts always refer to transcript-strand
adenosines. For stranded libraries pass `library=` so that only reads from
the region's strand are counted (GLORI A-less sense reads: "single-forward").
With library="unstranded" reads from both strands are pooled.

Outputs
-------
`pileup_bam` returns a list of `TranscriptObs` (one per region you ask for),
ready to feed into `forward_backward` or `baum_welch`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pysam
from pyfaidx import Fasta
from tqdm import tqdm

from hmm_m6a.train import TranscriptObs


DRACH_D = set("AGT")   # A, G, U/T
DRACH_R = set("AG")
DRACH_H = set("ACT")


@dataclass
class Region:
    """A genomic interval to score as one HMM chain."""

    name: str        # transcript or region identifier (used in output)
    contig: str      # contig in the BAM
    start: int       # 0-based inclusive
    end: int         # 0-based exclusive
    strand: str = "+"  # "+" or "-"


@dataclass
class AdenosineRecord:
    """One row of per-adenosine output, joined with HMM posterior."""

    region: str
    contig: str
    pos0: int          # 0-based reference position
    strand: str
    n: int             # informative coverage
    A: int             # A-like reads
    rate: float        # A / max(n, 1)
    drach: bool
    pU: float
    pM: float
    pB: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_drach_at(seq: str, i: int) -> bool:
    """DRACH 5-mer test centred at position i of the *transcript-strand* sequence."""
    if i < 2 or i > len(seq) - 3:
        return False
    a, b, c, d, e = seq[i - 2 : i + 3]
    return (a in DRACH_D) and (b in DRACH_R) and (c == "A") and (d == "C") and (e in DRACH_H)


_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def _revcomp(s: str) -> str:
    return s.translate(_COMPLEMENT)[::-1]


LIBRARIES = ("unstranded", "single-forward", "single-reverse", "R1-sense", "R1-antisense")


def _rna_strand(aln: pysam.AlignedSegment, library: str) -> str:
    """RNA strand a read came from, under the given library orientation."""
    fwd = not aln.is_reverse
    if library == "single-forward":
        sense = fwd
    elif library == "single-reverse":
        sense = not fwd
    elif library in ("R1-sense", "R1-antisense"):
        r1 = (not aln.is_paired) or aln.is_read1
        sense = (fwd == r1) if library == "R1-sense" else (fwd != r1)
    else:
        raise ValueError(f"unknown library type: {library}")
    return "+" if sense else "-"


def _read_on_strand(aln: pysam.AlignedSegment, strand: str, library: str) -> bool:
    """True if the read derives from RNA on `strand` (always True if unstranded).

    In a stranded GLORI library, a read from the opposite (antisense) strand
    shows the complement of the transcript base, so an antisense read at a
    transcript adenosine looks like an unconverted A. Such reads must be
    excluded or they inflate A-retention wherever antisense transcription
    overlaps the region.
    """
    if library == "unstranded":
        return True
    return _rna_strand(aln, library) == strand


def _read_key(aln: pysam.AlignedSegment) -> tuple:
    return (aln.query_name, aln.reference_start, aln.is_reverse, aln.is_read1)


def _poorly_converted_reads(
    bam: pysam.AlignmentFile,
    fasta: Fasta,
    region: Region,
    min_mapq: int,
    min_baseq: int,
    library: str,
    max_unconverted: int,
    margin: int = 1000,
) -> set:
    """Keys of reads overlapping `region` that carry more than `max_unconverted`
    unconverted transcript-strand adenosines anywhere along the read.

    This is the GLORI-style read filter: GLORI's own caller discards reads with
    several unconverted A's, because such molecules largely escaped the
    chemistry (they are not informative about methylation at any one site).
    """
    lo = max(0, region.start - margin)
    hi = region.end + margin
    try:
        ref = str(fasta[region.contig][lo:hi]).upper()
    except (KeyError, ValueError):
        return set()
    tgt = "A" if region.strand == "+" else "T"
    bad = set()
    for aln in bam.fetch(region.contig, region.start, region.end):
        if (aln.is_unmapped or aln.is_secondary or aln.is_supplementary
                or aln.mapping_quality < min_mapq):
            continue
        if not _read_on_strand(aln, region.strand, library):
            continue
        q = aln.query_sequence
        if q is None:
            continue
        quals = aln.query_qualities
        unc = 0
        for qp, rp in aln.get_aligned_pairs(matches_only=True):
            k = rp - lo
            if k < 0 or k >= len(ref) or ref[k] != tgt:
                continue
            if quals is not None and quals[qp] < min_baseq:
                continue
            if q[qp].upper() == tgt:
                unc += 1
        if unc > max_unconverted:
            bad.add(_read_key(aln))
    return bad


# ---------------------------------------------------------------------------
# Per-region pileup
# ---------------------------------------------------------------------------
def _pileup_region(
    bam: pysam.AlignmentFile,
    fasta: Fasta,
    region: Region,
    min_mapq: int,
    min_baseq: int,
    library: str = "unstranded",
    max_unconverted: int = -1,
) -> TranscriptObs:
    """Build a TranscriptObs for one region by walking pysam's pileup."""

    # Reference sequence on the *transcript* strand.
    # pyfaidx returns plus-strand bases; reverse-complement for minus-strand
    # regions so that the "A" we look for is the m6A-relevant adenosine in
    # the read frame.
    raw = str(fasta[region.contig][region.start : region.end]).upper()
    if region.strand == "-":
        seq = _revcomp(raw)
    else:
        seq = raw

    L = region.end - region.start
    A_count = np.zeros(L, dtype=np.int32)
    G_count = np.zeros(L, dtype=np.int32)
    drach = np.zeros(L, dtype=bool)

    # Pre-compute DRACH on the transcript-strand sequence
    for i, base in enumerate(seq):
        if base == "A":
            drach[i] = _is_drach_at(seq, i)

    bad_reads = (
        _poorly_converted_reads(bam, fasta, region, min_mapq, min_baseq, library,
                                max_unconverted)
        if max_unconverted >= 0 else set()
    )

    # We use truncate=True so pileup only emits columns inside [start, end).
    # min_base_quality / min_mapping_quality are passed to pysam directly.
    # Note: pysam returns reference positions in 0-based genomic coordinates,
    # so we map them back to a 0-based offset inside the region.
    for col in bam.pileup(
        contig=region.contig,
        start=region.start,
        stop=region.end,
        truncate=True,
        min_base_quality=min_baseq,
        min_mapping_quality=min_mapq,
        ignore_orphans=False,
        ignore_overlaps=False,
    ):
        ref_pos = col.reference_pos
        if ref_pos < region.start or ref_pos >= region.end:
            continue
        # Position within the (genomic-strand) region buffer
        i_genomic = ref_pos - region.start
        # Index on the transcript-strand sequence
        i = (L - 1 - i_genomic) if region.strand == "-" else i_genomic

        if seq[i] != "A":
            # Only adenosines on the transcript strand carry information
            continue

        n_A = 0
        n_G = 0
        for read in col.pileups:
            if read.is_del or read.is_refskip:
                continue
            if not _read_on_strand(read.alignment, region.strand, library):
                continue
            if bad_reads and _read_key(read.alignment) in bad_reads:
                continue
            qpos = read.query_position
            if qpos is None:
                continue
            base = read.alignment.query_sequence[qpos].upper()
            # If we're on the minus strand, the read base must be
            # complemented to compare against the transcript-strand sequence.
            if region.strand == "-":
                base = base.translate(_COMPLEMENT)
            if base == "A":
                n_A += 1
            elif base == "G":
                n_G += 1
            # bases C, T (or N, indels) are uninformative under GLORI

        A_count[i] = n_A
        G_count[i] = n_G

    n_total = (A_count + G_count).astype(np.int32)
    A_idx = np.where(np.frombuffer(seq.encode(), dtype="S1") == b"A")[0].astype(np.int32)

    return TranscriptObs(
        L=L, A_idx=A_idx, A_count=A_count, n=n_total, drach=drach,
    )


# ---------------------------------------------------------------------------
# Region discovery from a BAM (used when the user does not provide a BED)
# ---------------------------------------------------------------------------
def regions_from_bed(path: str) -> List[Region]:
    """Read a BED3 / BED6 file into Region objects.

    Strand defaults to '+'. Region name defaults to '{contig}:{start}-{end}'.
    """
    out: List[Region] = []
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.rstrip("\n").split("\t")
            contig, start, end = f[0], int(f[1]), int(f[2])
            name = f[3] if len(f) >= 4 else f"{contig}:{start}-{end}"
            strand = f[5] if len(f) >= 6 else "+"
            out.append(Region(name=name, contig=contig, start=start, end=end, strand=strand))
    return out


def regions_from_bam_contigs(bam_path: str, max_contig_len: int = 2_000_000) -> List[Region]:
    """Fallback: one Region per BAM contig, restricted to contigs shorter than
    max_contig_len. Useful for small targeted BAMs (e.g. RNU genes only).

    For whole-genome BAMs, supply a BED with transcript intervals instead.
    """
    out: List[Region] = []
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for ref in bam.references:
            ln = bam.get_reference_length(ref)
            if ln <= max_contig_len:
                out.append(Region(name=ref, contig=ref, start=0, end=ln))
    return out


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------
def pileup_bam(
    bam_path: str,
    fasta_path: str,
    regions: Sequence[Region],
    min_mapq: int = 20,
    min_baseq: int = 20,
    show_progress: bool = True,
    library: str = "unstranded",
    max_unconverted: int = -1,
) -> List[Tuple[Region, TranscriptObs]]:
    """Pile up per-adenosine A/G counts and compute DRACH context.

    Args:
        bam_path: coordinate-sorted, indexed BAM. If the .bai is missing this
            function will create it.
        fasta_path: indexed reference FASTA (.fai must exist or pyfaidx will
            create it).
        regions: list of Region intervals to score.
        min_mapq: minimum read MAPQ.
        min_baseq: minimum base quality at the pileup column.
        show_progress: emit a tqdm bar.
        library: read orientation of a stranded library ("single-forward" for
            GLORI A-less sense reads, "single-reverse", "R1-sense",
            "R1-antisense") or "unstranded" (keep reads from both strands).
        max_unconverted: GLORI-style read filter; drop reads with more than
            this many unconverted transcript adenosines (-1 = keep all reads).

    Returns:
        list of (Region, TranscriptObs) in the same order as `regions`.
    """
    # Auto-index BAM if needed
    if library not in LIBRARIES:
        raise ValueError(f"library must be one of {LIBRARIES}")
    if not _has_bai(bam_path):
        pysam.index(bam_path)

    bam = pysam.AlignmentFile(bam_path, "rb")
    fasta = Fasta(fasta_path, sequence_always_upper=True, rebuild=False)
    out: List[Tuple[Region, TranscriptObs]] = []
    iterator = tqdm(regions, desc="pileup", unit="region") if show_progress else regions
    for r in iterator:
        if r.contig not in fasta:
            # Reference FASTA missing this contig (e.g. a _random decoy) - skip.
            continue
        try:
            tr = _pileup_region(bam, fasta, r, min_mapq=min_mapq, min_baseq=min_baseq,
                                library=library, max_unconverted=max_unconverted)
        except (ValueError, KeyError):
            # Contig present in FASTA but absent from BAM index, etc.
            continue
        out.append((r, tr))
    bam.close()
    return out


def _has_bai(bam_path: str) -> bool:
    import os
    return os.path.exists(bam_path + ".bai") or os.path.exists(bam_path.replace(".bam", ".bai"))
