"""Build the committed example BAM + BED under workflow/example/.

Regenerate with:

    python workflow/example/make_example_data.py

Two loci on two 200 bp reference contigs, built with the same construction
this repo's own test suite uses (``tests/test_io_bam.py``), so the workflow
example and the test fixtures stay provably consistent. Every family
carries both duplex strands (A and B), and every strand is 2-5 PCR-duplicate
read pairs of the same molecule (``count > 1`` throughout — no
``(locus, mi, strand)`` row collapses from a single read pair). Family-level
consensus pools the two strands (one molecule, not a strand); each strand
keeps its own row in ``haplotype_calls.tsv``.

- **Locus1** (``synth1``) — 10 read families, a plain uninterrupted G-run
  tract (``n_runs=1``, ``interrupter_pattern`` renders as ``"-"`` on every
  row), no informative flanking SNV. The "nothing to report" case: real
  depth, no interrupter alleles, no phase signal — every row phases to
  ``unk``.

- **Locus2** (``synth2``) — 30 read families split 15/15 across two
  flanking positions (one upstream, one downstream of the polyG tract),
  co-segregating, so both clear the default ``--min-coverage 25``/
  ``--vaf-min 0.2`` informative-SNV filter and the default
  ``--min-linkage-families 10`` phasing gate — real ``hap_major``/
  ``hap_minor`` labels and a two-SNV ``hap_major_profile``/
  ``hap_minor_profile``. Two further things layered on top of that split:

  - **Interrupter alleles.** Most families carry the tract's canonical
    "A" interrupter; 6 of 30 (3 from each haplotype group, so the split
    doesn't trivially track the SNV-based one) carry an alternate "T"
    interrupter instead — same run-length tuple (7, 6), different
    interrupter identity. Both patterns clear the default
    ``--min-pattern-freq 0.1`` admission threshold, so
    ``locus_summary.tsv``'s ``admitted_patterns`` lists two real entries,
    and ``phase_interrupter_concordance_major``/``_minor`` come out at a
    real 0.8 rather than the trivial 1.0 a single-pattern locus gives.
  - **A genuine "other" vote.** One family's B-strand read carries a third
    base (neither haplotype's allele) at the upstream SNV, so its row gets
    ``n_other=1`` there while its A-strand sibling row does not — the two
    strand rows of one molecule showing different vote evidence, per
    ``haplotype_calls.tsv``'s own "a molecule can carry both strands, and
    they can disagree" note. The row still gets a definite hap_major/
    hap_minor label from its one remaining clean vote (the downstream
    SNV) — demonstrating that ``n_other > 0`` does not by itself mean
    ``phase_label="unk"``: ``unk`` comes from a tied or insufficient
    major/minor vote margin, not from off-haplotype votes. This family's
    two strands also carry equal depth, so their family-level consensus
    at the upstream SNV (which pools both strands, read-count weighted)
    is an exact tie and abstains by construction — one real consequence
    is ``snv_calls.tsv``'s upstream-SNV ``coverage`` reading 29, not 30:
    not a bug, family_consensus_counts' own documented tie rule.

A second BAM, ``example_target.bam``, is a genuinely different sample at the
same two loci — for the ``ref-phase`` quickstart, which needs a target
sample to vote against a *different*, already-phased reference. Locus1 is
unchanged (still no informative SNV, still all-``unk`` regardless of which
profile it is voted against). Locus2 carries the identical two genomic
alleles at each flanking SNV as ``example.bam`` (``ref-phase`` borrows the
reference's already-determined hap_major/hap_minor profile — it never
re-discovers alleles from the target), but a skewed 12 hap-A-group / 6
hap-B-group split instead of the reference's balanced 15/15: a different
family composition, as an independently sequenced sample would have, with
no interrupter alternation or strand-mismatch flourish (those exercise
``phase``'s own channels, not the flanking-SNV vote ``ref-phase`` cares
about).

Still small — under 700 reads total per BAM — and fast enough to extract
and phase in well under a second.
"""
from __future__ import annotations

from pathlib import Path

import pysam

EXAMPLE_DIR = Path(__file__).resolve().parent

_REF_PREFIX = "A" * 50 + "C" * 10
BED_START = 60
BED_END = 74


def _build_reference(tract: str) -> str:
    assert len(tract) == BED_END - BED_START
    return (
        _REF_PREFIX[:50] + "ACGTACGTAC" + tract + "TGCATGCATG"
        + ("A" * (200 - 50 - 10 - len(tract) - 10))
    )


# Locus1: a plain, uninterrupted 14 bp G-run (n_runs=1, no interrupter allele
# at all). Locus2: the canonical run-length-(7,6) tract with a single "A"
# interrupter — the majority pattern; a minority of Locus2 reads substitute
# "T" there instead (see main()).
REFERENCE_PLAIN = _build_reference("G" * 14)
REFERENCE_INTERRUPTED = _build_reference("GGGGGGG" + "A" + "GGGGGG")
assert len(REFERENCE_PLAIN) == len(REFERENCE_INTERRUPTED) == 200

# Locus2's interrupter position, in absolute reference coordinates:
# BED_START + 7 (the 8th tract base, 0-indexed) — the "A"/"T" swap site.
_INTERRUPTER_REFPOS = BED_START + 7

# Flanking-SNV offsets used by Locus2, in the same "canonical distance from
# the anchor" convention prolyg_phasing.io.bam._alignment_flanking uses:
# up_dist=0 is the reference base immediately outside the upstream anchor
# (position 49, one before the anchor's own 50-59); dn_dist=0 is the same
# on the downstream side (position 84, one past the anchor's own 74-83).
# Both fall inside the flat "A" padding on either side of the anchors, so
# mutating them cannot perturb anchor placement or the polyG tract itself.
_UP_SNV_REFPOS = 49
_DN_SNV_REFPOS = 84


def _build_md_tag(read_seq: str, ref_seq: str) -> str:
    assert len(read_seq) == len(ref_seq), "MD helper requires gapless equal-length alignment"
    md_parts: list[str] = []
    match_run = 0
    for r_base, ref_base in zip(read_seq.upper(), ref_seq.upper(), strict=True):
        if r_base == ref_base:
            match_run += 1
        else:
            md_parts.append(str(match_run))
            md_parts.append(ref_base)
            match_run = 0
    md_parts.append(str(match_run))
    return "".join(md_parts)


def _add_read(
    bam: pysam.AlignmentFile,
    qname: str,
    read_seq: str,
    *,
    ref_id: int,
    reference: str,
    ref_start: int,
    mapq: int = 60,
    mi: str,
    is_read1: bool = True,
    tlen: int = 120,
) -> None:
    seg = pysam.AlignedSegment(bam.header)
    seg.query_name = qname
    seg.query_sequence = read_seq
    seg.query_qualities = pysam.qualitystring_to_array("I" * len(read_seq))
    seg.flag = 0
    if is_read1:
        seg.is_read1 = True
    else:
        seg.is_read2 = True
    seg.is_paired = True
    seg.mate_is_unmapped = False
    seg.reference_id = ref_id
    seg.reference_start = ref_start
    seg.mapping_quality = mapq
    seg.cigar = [(0, len(read_seq))]  # all M
    seg.next_reference_id = ref_id
    seg.next_reference_start = ref_start + tlen - len(read_seq)
    seg.template_length = tlen
    ref_slice = reference[ref_start : ref_start + len(read_seq)]
    md = _build_md_tag(read_seq, ref_slice)
    seg.set_tags([
        ("MD", md),
        ("NM", sum(c != r for c, r in zip(read_seq, ref_slice, strict=True))),
        ("MI", mi, "Z"),
    ])
    bam.write(seg)


def _read_seq(reference: str, overrides: dict[int, str] | None = None) -> str:
    """The first 150 bp of ``reference``, with any ``{refpos: base}`` overrides."""
    read = list(reference[:150])
    for pos, base in (overrides or {}).items():
        read[pos] = base
    return "".join(read)


# Locus1's plain sequence, and Locus2's two flanking-SNV haplotype sequences —
# shared module-level so example.bam and example_target.bam draw reads
# carrying the SAME genomic alleles (ref-phase votes a target sample's reads
# against a reference sample's already-determined profile; the two samples
# must agree on what the alleles themselves are, only how many families carry
# each one is free to differ).
PLAIN_SEQ = _read_seq(REFERENCE_PLAIN)
HAP_A_SEQ = _read_seq(REFERENCE_INTERRUPTED, {_UP_SNV_REFPOS: "C", _DN_SNV_REFPOS: "T"})
HAP_B_SEQ = _read_seq(REFERENCE_INTERRUPTED, {_UP_SNV_REFPOS: "G", _DN_SNV_REFPOS: "C"})


def _depth(i: int) -> int:
    """PCR-duplicate read-pair count for family index ``i``: cycles 2, 3, 4, 5."""
    return 2 + (i % 4)


def _add_family(
    bam: pysam.AlignmentFile,
    *,
    ref_id: int,
    reference: str,
    mi_prefix: str,
    read_seq: str,
    depth: int,
) -> None:
    """One read family: both duplex strands (A and B), each ``depth``
    PCR-duplicate read pairs (same molecule, same sequence, distinct
    qnames) that collapse to one row with ``count=depth``.
    """
    for strand in ("A", "B"):
        mi = f"{mi_prefix}/{strand}"
        for k in range(depth):
            for is_read1 in (True, False):
                _add_read(
                    bam, qname=f"{mi_prefix}_{strand}_{k}", read_seq=read_seq,
                    ref_id=ref_id, reference=reference, ref_start=0,
                    mi=mi, is_read1=is_read1,
                )


def _add_family_strand_mismatch(
    bam: pysam.AlignmentFile,
    *,
    ref_id: int,
    reference: str,
    mi_prefix: str,
    read_seq_a: str,
    read_seq_b: str,
    depth: int,
) -> None:
    """Like :func:`_add_family`, but strand A and strand B carry different
    sequence content — the family's two rows genuinely disagree."""
    for strand, seq in (("A", read_seq_a), ("B", read_seq_b)):
        mi = f"{mi_prefix}/{strand}"
        for k in range(depth):
            for is_read1 in (True, False):
                _add_read(
                    bam, qname=f"{mi_prefix}_{strand}_{k}", read_seq=seq,
                    ref_id=ref_id, reference=reference, ref_start=0,
                    mi=mi, is_read1=is_read1,
                )


_HEADER = {
    "HD": {"VN": "1.6", "SO": "coordinate"},
    "SQ": [
        {"SN": "synth1", "LN": len(REFERENCE_PLAIN)},
        {"SN": "synth2", "LN": len(REFERENCE_INTERRUPTED)},
    ],
}


def _write_bam(bam_path: Path, add_reads) -> None:
    """Write, coordinate-sort, and index a BAM; ``add_reads(bam)`` populates it."""
    unsorted_path = bam_path.with_name(bam_path.stem + ".unsorted.bam")
    with pysam.AlignmentFile(str(unsorted_path), "wb", header=_HEADER) as bam:
        add_reads(bam)
    pysam.sort("-o", str(bam_path), str(unsorted_path))
    pysam.index(str(bam_path))
    unsorted_path.unlink()


def _add_reference_reads(bam: pysam.AlignmentFile) -> None:
    # Locus1: 10 plain families, no interrupter allele, no flanking-SNV signal.
    for i in range(10):
        _add_family(
            bam, ref_id=0, reference=REFERENCE_PLAIN, mi_prefix=f"20{i:03d}",
            read_seq=PLAIN_SEQ, depth=_depth(i),
        )

    # Locus2: 30 families, 15 hap-A-group + 15 hap-B-group, co-segregating
    # at both flanking SNV positions. Which group ends up labeled
    # hap_major vs hap_minor is the algorithm's own call (an exact 15/15
    # tie here), not predicted by this split.
    # 3 families per haplotype group carry an alternate "T" interrupter
    # instead of the canonical "A" — split across both groups so the
    # interrupter-pattern channel doesn't trivially track the SNV-phased
    # haplotype channel.
    alt_interrupter_indices = {2, 7, 12, 17, 22, 27}
    # This one family's B-strand row gets a genuine third-allele base at
    # the upstream SNV (see _add_family_strand_mismatch below); skip it
    # in the main loop.
    strand_mismatch_index = 0

    for i in range(30):
        if i == strand_mismatch_index:
            continue
        base_seq = HAP_A_SEQ if i < 15 else HAP_B_SEQ
        read_seq = (
            _read_seq(REFERENCE_INTERRUPTED, {
                _INTERRUPTER_REFPOS: "T",
                _UP_SNV_REFPOS: base_seq[_UP_SNV_REFPOS],
                _DN_SNV_REFPOS: base_seq[_DN_SNV_REFPOS],
            })
            if i in alt_interrupter_indices else base_seq
        )
        _add_family(
            bam, ref_id=1, reference=REFERENCE_INTERRUPTED,
            mi_prefix=f"30{i:03d}", read_seq=read_seq, depth=_depth(i),
        )

    # Family 0 (hap-A group): strand A reads the clean hap-A sequence;
    # strand B substitutes a third base ("A", neither hap-A's "C" nor
    # hap-B's "G") at the upstream SNV, leaving the downstream SNV and
    # the interrupter untouched.
    mismatch_seq_b = _read_seq(
        REFERENCE_INTERRUPTED,
        {_UP_SNV_REFPOS: "A", _DN_SNV_REFPOS: HAP_A_SEQ[_DN_SNV_REFPOS]},
    )
    _add_family_strand_mismatch(
        bam, ref_id=1, reference=REFERENCE_INTERRUPTED,
        mi_prefix=f"30{strand_mismatch_index:03d}",
        read_seq_a=HAP_A_SEQ, read_seq_b=mismatch_seq_b,
        depth=_depth(strand_mismatch_index),
    )


def _add_target_reads(bam: pysam.AlignmentFile) -> None:
    # Locus1: same plain families as the reference sample -- still no
    # informative SNV, still all-unk regardless of which profile it votes
    # against.
    for i in range(10):
        _add_family(
            bam, ref_id=0, reference=REFERENCE_PLAIN, mi_prefix=f"50{i:03d}",
            read_seq=PLAIN_SEQ, depth=_depth(i),
        )

    # Locus2: 18 families, skewed 12 hap-A-group + 6 hap-B-group -- a
    # different family composition than the reference sample's balanced
    # 15/15, as an independently sequenced sample would have, but the SAME
    # two genomic alleles at each flanking SNV (HAP_A_SEQ/HAP_B_SEQ):
    # ref-phase borrows the reference's already-determined hap_major/
    # hap_minor profile, it never re-discovers alleles from the target. No
    # interrupter alternation or strand-mismatch flourish here -- those
    # exercise phase's own channels, not the flanking-SNV vote ref-phase
    # cares about.
    for i in range(18):
        base_seq = HAP_A_SEQ if i < 12 else HAP_B_SEQ
        _add_family(
            bam, ref_id=1, reference=REFERENCE_INTERRUPTED,
            mi_prefix=f"60{i:03d}", read_seq=base_seq, depth=_depth(i),
        )


def main() -> None:
    bam_path = EXAMPLE_DIR / "example.bam"
    _write_bam(bam_path, _add_reference_reads)

    target_bam_path = EXAMPLE_DIR / "example_target.bam"
    _write_bam(target_bam_path, _add_target_reads)

    bed_path = EXAMPLE_DIR / "panel.bed"
    bed_path.write_text(
        f"synth1\t{BED_START}\t{BED_END}\tLocus1\n"
        f"synth2\t{BED_START}\t{BED_END}\tLocus2\n"
    )

    print(f"wrote {bam_path}, {bam_path}.bai, {target_bam_path}, "
          f"{target_bam_path}.bai, {bed_path}")


if __name__ == "__main__":
    main()
