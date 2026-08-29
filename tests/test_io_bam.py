"""Tests for prolyg_phasing.io.bam — synthetic BAM fixture + extract_panel end-to-end.

Builds a tiny in-memory BAM with controlled reads at one or two loci,
then runs `extract_panel` and verifies:

- anchor construction off MD tags
- alignment filters (MAPQ, SA, TLEN)
- pair-collapse (R1+R2 agree → 1 read, disagree → drop)
- per-locus QC counters
- panel-wide `n_alleles` decision

``ExtractedPanel.to_loci()`` ships as a method here (it lazy-imports
prolyG's PCR-chemistry fit machinery, which this package never installs) but
is untested in this repo's own suite for the same reason. A real extraction
feeding it cleanly is covered from prolyG's side instead —
``tests/test_extract_to_loci_integration.py``.
"""

from __future__ import annotations

from pathlib import Path

import pysam
import pytest

from prolyg_phasing.io.bam import extract_panel
from prolyg_phasing.io.panel import ExtractedPanel

# ---------------------------------------------------------------------------
# Synthetic BAM helpers
# ---------------------------------------------------------------------------


# A 200-bp synthetic reference, single chromosome. Bases 60..73 (inclusive
# half-open 60..74) carry the polyG region we'll target with the BED;
# anchored by ACGTACGTAC upstream (50..60) and TGCATGCATG downstream (74..84).
# Reference structure: ACGTACGTAC GGGGGGGAGGGGGG TGCATGCATG
#  - pre-polyG context (50..60):   ACGTACGTAC
#  - polyG region (60..74):         GGGGGGGAGGGGGG  (S=2, runs (7, 6))
#  - post-polyG context (74..84):  TGCATGCATG
_REF_PREFIX = "A" * 50 + "C" * 10
_REF_POLYG = "GGGGGGG" + "A" + "GGGGGG"  # 7 + 1 + 6 = 14 bp
_REF_SUFFIX = "T" * 10 + "G" * 116
REFERENCE = (
    _REF_PREFIX[:50] + "ACGTACGTAC" + _REF_POLYG + "TGCATGCATG"
    + ("A" * (200 - 50 - 10 - 14 - 10))
)
assert len(REFERENCE) == 200
# Expected anchors built by the walker:
# - upstream walk left from pos 60 through Gs: 0 steps; pos 59 is 'C'.
#   Anchor ends at 59, 10 bp upstream = positions 50..60 = "ACGTACGTAC".
# - downstream walk right from pos 74 through Gs: 0 steps; pos 74 is 'T'.
#   Anchor starts at 74, 10 bp = positions 74..84 = "TGCATGCATG".

CHROM = "synth"
BED_START = 60
BED_END = 74


def _build_md_tag(read_seq: str, ref_seq: str) -> str:
    """Build an MD tag from a gapless alignment of read_seq to ref_seq.

    Simplified: produces an MD string of the form ``N[base...]N`` where
    `N` is the count of consecutive matches and bracketed bases mark
    mismatches. Works only for aligned reads of identical length to the
    reference (no indels), which suffices for tests.
    """
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
    ref_start: int,
    mapq: int = 60,
    mi: str | None,
    is_read1: bool = True,
    is_reverse: bool = False,
    tlen: int = 200,
    has_sa: bool = False,
    quality_str: str | None = None,
) -> None:
    """Add one synthetic alignment to a pysam.AlignmentFile open for writing.

    ``quality_str`` defaults to all-``"I"`` (Q40). Pass a custom string to
    inject low-quality bases at specific positions for flanking tests.
    """
    seg = pysam.AlignedSegment(bam.header)
    seg.query_name = qname
    seg.query_sequence = read_seq
    if quality_str is None:
        quality_str = "I" * len(read_seq)
    assert len(quality_str) == len(read_seq), "quality_str must match read length"
    seg.query_qualities = pysam.qualitystring_to_array(quality_str)
    seg.flag = 0
    if not is_read1:
        seg.is_read2 = True
    else:
        seg.is_read1 = True
    seg.is_paired = True
    seg.mate_is_unmapped = False
    if is_reverse:
        seg.is_reverse = True
    seg.reference_id = 0
    seg.reference_start = ref_start
    seg.mapping_quality = mapq
    seg.cigar = [(0, len(read_seq))]  # all M
    seg.next_reference_id = 0
    seg.next_reference_start = ref_start + tlen - len(read_seq)
    seg.template_length = tlen
    # MD tag (used for reference reconstruction)
    ref_slice = REFERENCE[ref_start : ref_start + len(read_seq)]
    md = _build_md_tag(read_seq, ref_slice)
    tags = [("MD", md), ("NM", sum(c != r for c, r in zip(read_seq, ref_slice, strict=True)))]
    if mi is not None:
        tags.append(("MI", mi, "Z"))
    if has_sa:
        tags.append(("SA", "synth,100,+,150M,60,0;", "Z"))
    seg.set_tags(tags)
    bam.write(seg)


def _make_test_bam(tmp_path: Path, reads: list[dict]) -> Path:
    """Create a sorted, indexed BAM at tmp_path/test.bam with the given reads."""
    bam_path = tmp_path / "test.bam"
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": CHROM, "LN": len(REFERENCE)}],
    }
    unsorted = tmp_path / "test.unsorted.bam"
    with pysam.AlignmentFile(str(unsorted), "wb", header=header) as bam:
        for r in reads:
            _add_read(bam, **r)

    pysam.sort("-o", str(bam_path), str(unsorted))
    pysam.index(str(bam_path))
    return bam_path


def _make_bed(tmp_path: Path, rows: list[tuple[str, str, int, int]]) -> Path:
    """Write a 4-column BED at tmp_path/test.bed: (chrom, start, end, name) per row."""
    bed_path = tmp_path / "test.bed"
    with bed_path.open("w") as f:
        for chrom, start, end, name in rows:
            f.write(f"{chrom}\t{start}\t{end}\t{name}\n")
    return bed_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _canonical_read_seq(polyg_seq: str = _REF_POLYG, ref_start: int = 0) -> tuple[str, int]:
    """A 150-bp read spanning the BED+pad window so MD-tag reference reconstruction succeeds.

    Substitutes `polyg_seq` for the reference polyG region (positions
    BED_START..BED_END within the read). The read is gapless of length
    150 against the reference, starting at `ref_start` (default 0 so
    `[ref_start, ref_start+150)` covers `[BED_START - pad, BED_END + pad)`
    for the default pad=50).
    """
    read = list(REFERENCE[ref_start : ref_start + 150])
    # Replace the polyG region (positions BED_START..BED_END in the reference,
    # offset by ref_start into the read).
    poly_offset = BED_START - ref_start
    poly_len = BED_END - BED_START
    assert len(polyg_seq) == poly_len, (
        f"Test polyG must be {poly_len} bp to keep the alignment gapless"
    )
    for i, b in enumerate(polyg_seq):
        read[poly_offset + i] = b
    return "".join(read), ref_start


def test_extract_canonical_panel(tmp_path):
    """Two MIs at one S=2 locus; verify reads, panel n_alleles."""
    # MI 10004/A: 2 read-pairs at (7,6); MI 10004/B: 1 pair at (7,6); MI 10007/A: 1 pair at (6,7).
    # All polyG substitutions are 14-bp (= BED_END - BED_START) to keep the alignment gapless.
    seq_canonical, rs = _canonical_read_seq(_REF_POLYG)                    # (7, 6)
    seq_alt, _ = _canonical_read_seq("GGGGGG" + "A" + "GGGGGGG")           # (6, 7)

    reads: list[dict] = []
    # Two concordant pairs from MI 10004/A
    for i in range(2):
        for r1 in (True, False):
            reads.append(dict(
                qname=f"q1_{i}", read_seq=seq_canonical, ref_start=rs,
                mapq=60, mi="10004/A", is_read1=r1, tlen=120,
            ))
    # One concordant pair from MI 10004/B
    for r1 in (True, False):
        reads.append(dict(
            qname="q2", read_seq=seq_canonical, ref_start=rs,
            mapq=60, mi="10004/B", is_read1=r1, tlen=120,
        ))
    # One concordant pair from MI 10007/A, different tuple (6, 7)
    for r1 in (True, False):
        reads.append(dict(
            qname="q3", read_seq=seq_alt, ref_start=rs,
            mapq=60, mi="10007/A", is_read1=r1, tlen=120,
        ))

    bam_path = _make_test_bam(tmp_path, reads)
    bed_path = _make_bed(tmp_path, [(CHROM, BED_START, BED_END, "Locus1")])

    panel = extract_panel(bam_path, bed_path)
    assert isinstance(panel, ExtractedPanel)
    assert set(panel.loci.keys()) == {"Locus1"}

    loc = panel.loci["Locus1"]
    assert loc.anchorability_status == "anchorable"
    assert loc.ref_orientation == "+"
    assert loc.n_runs == 2
    assert loc.reference_run_lengths == (7, 6)
    assert loc.upstream_anchor == "ACGTACGTAC"
    assert loc.downstream_anchor == "TGCATGCATG"
    # 2 + 1 + 1 = 4 reads (post-pair-collapse)
    assert int(loc.count.sum()) == 4
    assert loc.n_reads == 4
    assert loc.n_read_pairs == 4
    assert loc.n_read_pairs_drop_disagree == 0
    # max observed run length in canonical (7,6) = 7; n_alleles = 7 + 1 = 8.
    assert panel.n_alleles == 8


def test_pair_disagreement_drops_fragment(tmp_path):
    """R1+R2 with differing polyG sequences → counted, dropped."""
    # (7, 6) canonical vs (6, 7) — disagreement; same alignment length.
    seq_a, rs = _canonical_read_seq(_REF_POLYG)
    seq_b, _ = _canonical_read_seq("GGGGGG" + "A" + "GGGGGGG")
    reads = [
        dict(qname="q1", read_seq=seq_a, ref_start=rs, mi="10004/A", is_read1=True, tlen=120),
        dict(qname="q1", read_seq=seq_b, ref_start=rs, mi="10004/A", is_read1=False, tlen=120),
    ]
    bam_path = _make_test_bam(tmp_path, reads)
    bed_path = _make_bed(tmp_path, [(CHROM, BED_START, BED_END, "Locus1")])

    panel = extract_panel(bam_path, bed_path)
    loc = panel.loci["Locus1"]
    assert loc.n_read_pairs == 1
    assert loc.n_read_pairs_drop_disagree == 1
    assert int(loc.count.sum()) == 0  # both reads dropped


def _ref_seed_reads(seq: str, ref_start: int, mi: str = "99999/A") -> list[dict]:
    """A clean spanning read pair so MD-tag reference reconstruction succeeds.

    Filter tests need at least one read that passes the parser-side QC
    filters so `get_reference_window` can rebuild the reference window.
    Without this, the locus comes back as `ref_window_unrecoverable`
    and the test can't assert anything about per-filter counters.
    """
    return [
        dict(qname="seed", read_seq=seq, ref_start=ref_start, mapq=60,
             mi=mi, is_read1=True, tlen=120),
        dict(qname="seed", read_seq=seq, ref_start=ref_start, mapq=60,
             mi=mi, is_read1=False, tlen=120),
    ]


def test_mapq_filter(tmp_path):
    """Low-MAPQ alignments are dropped and counted."""
    seq, rs = _canonical_read_seq(_REF_POLYG)
    reads = _ref_seed_reads(seq, rs)  # 2 alignments, will be the only surviving read
    reads.extend([
        dict(qname="q1", read_seq=seq, ref_start=rs, mapq=10,
             mi="10004/A", is_read1=True, tlen=120),
        dict(qname="q1", read_seq=seq, ref_start=rs, mapq=10,
             mi="10004/A", is_read1=False, tlen=120),
    ])
    bam_path = _make_test_bam(tmp_path, reads)
    bed_path = _make_bed(tmp_path, [(CHROM, BED_START, BED_END, "Locus1")])

    panel = extract_panel(bam_path, bed_path)
    loc = panel.loci["Locus1"]
    assert loc.n_alignments_drop_mapq == 2
    assert loc.n_reads == 1  # only the seed survives


def test_sa_tag_drops_chimeric(tmp_path):
    """SA-tagged alignments are dropped when drop_chimeric=True (default)."""
    seq, rs = _canonical_read_seq(_REF_POLYG)
    reads = _ref_seed_reads(seq, rs)
    reads.extend([
        dict(qname="q1", read_seq=seq, ref_start=rs, mi="10004/A",
             is_read1=True, tlen=120, has_sa=True),
        dict(qname="q1", read_seq=seq, ref_start=rs, mi="10004/A",
             is_read1=False, tlen=120, has_sa=True),
    ])
    bam_path = _make_test_bam(tmp_path, reads)
    bed_path = _make_bed(tmp_path, [(CHROM, BED_START, BED_END, "Locus1")])

    panel = extract_panel(bam_path, bed_path)
    loc = panel.loci["Locus1"]
    assert loc.n_alignments_drop_sa == 2
    assert loc.n_reads == 1  # only the seed survives


def test_tlen_filter(tmp_path):
    """|TLEN| > max_tlen drops the alignment."""
    seq, rs = _canonical_read_seq(_REF_POLYG)
    reads = _ref_seed_reads(seq, rs)
    reads.extend([
        dict(qname="q1", read_seq=seq, ref_start=rs, mi="10004/A",
             is_read1=True, tlen=5000),
        dict(qname="q1", read_seq=seq, ref_start=rs, mi="10004/A",
             is_read1=False, tlen=5000),
    ])
    bam_path = _make_test_bam(tmp_path, reads)
    bed_path = _make_bed(tmp_path, [(CHROM, BED_START, BED_END, "Locus1")])

    panel = extract_panel(bam_path, bed_path, max_tlen=1000)
    loc = panel.loci["Locus1"]
    assert loc.n_alignments_drop_tlen == 2
    assert loc.n_reads == 1  # only the seed survives


def test_pair_disagreement_policy_only_drop_supported(tmp_path):
    """`pair_disagreement_policy != 'drop'` raises NotImplementedError."""
    bed_path = _make_bed(tmp_path, [(CHROM, BED_START, BED_END, "Locus1")])
    bam_path = _make_test_bam(tmp_path, [])  # empty BAM
    with pytest.raises(NotImplementedError):
        extract_panel(bam_path, bed_path, pair_disagreement_policy="r1")


def test_provenance_records_kwargs(tmp_path):
    """The provenance JSON should capture the kwargs that were used."""
    bed_path = _make_bed(tmp_path, [(CHROM, BED_START, BED_END, "Locus1")])
    bam_path = _make_test_bam(tmp_path, [])

    panel = extract_panel(
        bam_path, bed_path,
        min_mapq=30, drop_chimeric=False, max_tlen=2000,
        anchor_hamming_max=2, n_alleles_margin=5,
    )
    p = panel.provenance
    assert p.min_mapq == 30
    assert p.drop_chimeric is False
    assert p.max_tlen == 2000
    assert p.anchor_hamming_max == 2
    assert p.n_alleles_margin == 5
    assert p.bam_path == str(bam_path)
    assert p.bed_path == str(bed_path)


def test_save_load_roundtrip_with_real_extraction(tmp_path):
    """End-to-end: extract → save_db → load_db, panel.db round trip preserved."""
    seq, rs = _canonical_read_seq(_REF_POLYG)
    reads: list[dict] = []
    for i in range(3):
        for r1 in (True, False):
            reads.append(dict(
                qname=f"q{i}", read_seq=seq, ref_start=rs,
                mapq=60, mi="10004/A", is_read1=r1, tlen=120,
            ))
    bam_path = _make_test_bam(tmp_path, reads)
    bed_path = _make_bed(tmp_path, [(CHROM, BED_START, BED_END, "Locus1")])

    panel = extract_panel(bam_path, bed_path)

    db = tmp_path / "panel.db"
    panel.save_db(db)
    loaded = ExtractedPanel.load_db(db)
    assert loaded.n_alleles == panel.n_alleles
    assert set(loaded.loci.keys()) == set(panel.loci.keys())


# ---------------------------------------------------------------------------
# Flanking extraction (schema v2)
# ---------------------------------------------------------------------------


def _mutate_at(seq: str, pos: int, base: str) -> str:
    return seq[:pos] + base + seq[pos + 1:]


def test_flanking_captured_with_expected_geometry(tmp_path):
    """One het locus, two reads — flanking strings get built with the right widths."""
    seq, rs = _canonical_read_seq(_REF_POLYG)
    reads = []
    for r1 in (True, False):
        reads.append(dict(
            qname="q1", read_seq=seq, ref_start=rs,
            mi="10004/A", is_read1=r1, tlen=120,
        ))
    bam_path = _make_test_bam(tmp_path, reads)
    bed_path = _make_bed(tmp_path, [(CHROM, BED_START, BED_END, "Locus1")])

    panel = extract_panel(bam_path, bed_path)
    loc = panel.loci["Locus1"]

    # Read covers ref [0, 150); upstream flanking is ref < 50 → 50 bp;
    # downstream flanking is ref >= 84 → 66 bp.
    assert loc.flanking_up_width == 50
    assert loc.flanking_dn_width == 66
    assert loc.g_walk_up == 0
    assert loc.g_walk_dn == 0
    # One dedup'd flanking string (one row in the locus CSV).
    assert len(loc.flanking_seq) == 1
    assert loc.flanking_id.tolist() == [0]
    # Combined string is up_seq + "|" + dn_seq.
    s = str(loc.flanking_seq[0])
    assert "|" in s
    up_part, dn_part = s.split("|")
    assert len(up_part) == 50
    assert len(dn_part) == 66
    # Reference has all 'A' in both flanking regions (positions 0..49 and 84..149)
    # so every covered position is 'A'.
    assert set(up_part) == {"A"}
    assert set(dn_part) == {"A"}


def test_flanking_snv_creates_two_flanking_ids(tmp_path):
    """Two reads differing at one flanking position → two flanking_ids."""
    seq_a, rs = _canonical_read_seq(_REF_POLYG)
    # Inject a flanking SNV at read position 20 (ref pos 20, upstream flanking).
    seq_b = _mutate_at(seq_a, 20, "C")
    reads = [
        dict(qname="qA", read_seq=seq_a, ref_start=rs, mi="10004/A", is_read1=True, tlen=120),
        dict(qname="qA", read_seq=seq_a, ref_start=rs, mi="10004/A", is_read1=False, tlen=120),
        dict(qname="qB", read_seq=seq_b, ref_start=rs, mi="10007/A", is_read1=True, tlen=120),
        dict(qname="qB", read_seq=seq_b, ref_start=rs, mi="10007/A", is_read1=False, tlen=120),
    ]
    bam_path = _make_test_bam(tmp_path, reads)
    bed_path = _make_bed(tmp_path, [(CHROM, BED_START, BED_END, "Locus1")])

    panel = extract_panel(bam_path, bed_path)
    loc = panel.loci["Locus1"]

    assert len(loc.flanking_seq) == 2
    # The flanking_id column distinguishes the two reads.
    assert sorted(loc.flanking_id.tolist()) == [0, 1]
    # Up-segment at distance (49 - 20) = 29 from the upstream anchor.
    up_strs = [str(s).split("|")[0] for s in loc.flanking_seq]
    # Both strings are length 50; they differ at exactly position (50 - 1 - 29) = 20
    # (canonical-far-to-near layout: position i = (width - 1 - distance)).
    assert len(set(up_strs)) == 2
    differing = [i for i in range(50) if up_strs[0][i] != up_strs[1][i]]
    assert differing == [20]
    assert {up_strs[0][20], up_strs[1][20]} == {"A", "C"}


def test_flanking_low_q_masked_to_dot(tmp_path):
    """A low-Q base in the flanking region appears as '.' in the string."""
    seq, rs = _canonical_read_seq(_REF_POLYG)
    # Default Q40 everywhere; lower position 20 to Q5 (below default min_base_q=20).
    quals = list("I" * len(seq))
    quals[20] = "&"  # Phred 5
    quality_str = "".join(quals)
    reads = [
        dict(qname="q1", read_seq=seq, ref_start=rs, mi="10004/A",
             is_read1=True, tlen=120, quality_str=quality_str),
        dict(qname="q1", read_seq=seq, ref_start=rs, mi="10004/A",
             is_read1=False, tlen=120, quality_str=quality_str),
    ]
    bam_path = _make_test_bam(tmp_path, reads)
    bed_path = _make_bed(tmp_path, [(CHROM, BED_START, BED_END, "Locus1")])

    panel = extract_panel(bam_path, bed_path)
    loc = panel.loci["Locus1"]

    up_part = str(loc.flanking_seq[0]).split("|")[0]
    # Read pos 20 in ref → upstream distance 29 → canonical string index 20.
    assert up_part[20] == "."
    # Neighbors are still 'A' (high-Q).
    assert up_part[19] == "A"
    assert up_part[21] == "A"


def test_flanking_pair_disagreement_drops_position(tmp_path):
    """R1 and R2 with disagreeing flanking bases → that position becomes '.'."""
    seq_r1, rs = _canonical_read_seq(_REF_POLYG)
    seq_r2 = _mutate_at(seq_r1, 20, "C")  # R2 carries C, R1 carries A at pos 20
    reads = [
        dict(qname="q1", read_seq=seq_r1, ref_start=rs, mi="10004/A", is_read1=True, tlen=120),
        dict(qname="q1", read_seq=seq_r2, ref_start=rs, mi="10004/A", is_read1=False, tlen=120),
    ]
    bam_path = _make_test_bam(tmp_path, reads)
    bed_path = _make_bed(tmp_path, [(CHROM, BED_START, BED_END, "Locus1")])

    panel = extract_panel(bam_path, bed_path)
    loc = panel.loci["Locus1"]
    # The polyG ia_seq itself agrees, so the pair survives.
    assert loc.n_read_pairs == 1
    assert loc.n_read_pairs_drop_disagree == 0
    up_part = str(loc.flanking_seq[0]).split("|")[0]
    assert up_part[20] == "."
    assert up_part[19] == "A"  # agreement → kept


def test_flanking_survives_save_load_roundtrip(tmp_path):
    """Extract → save_db → load_db preserves flanking_seq and flanking_id."""
    seq_a, rs = _canonical_read_seq(_REF_POLYG)
    seq_b = _mutate_at(seq_a, 100, "G")  # downstream flanking SNV
    reads = [
        dict(qname="qA", read_seq=seq_a, ref_start=rs, mi="10004/A", is_read1=True, tlen=120),
        dict(qname="qA", read_seq=seq_a, ref_start=rs, mi="10004/A", is_read1=False, tlen=120),
        dict(qname="qB", read_seq=seq_b, ref_start=rs, mi="10007/A", is_read1=True, tlen=120),
        dict(qname="qB", read_seq=seq_b, ref_start=rs, mi="10007/A", is_read1=False, tlen=120),
    ]
    bam_path = _make_test_bam(tmp_path, reads)
    bed_path = _make_bed(tmp_path, [(CHROM, BED_START, BED_END, "Locus1")])

    panel = extract_panel(bam_path, bed_path)
    db = tmp_path / "panel.db"
    panel.save_db(db)
    loaded = ExtractedPanel.load_db(db)
    a = panel.loci["Locus1"]
    b = loaded.loci["Locus1"]
    assert b.flanking_up_width == a.flanking_up_width
    assert b.flanking_dn_width == a.flanking_dn_width
    assert b.flanking_up_ref_pos_start == a.flanking_up_ref_pos_start
    assert b.flanking_dn_ref_pos_start == a.flanking_dn_ref_pos_start
    assert b.g_walk_up == a.g_walk_up
    assert b.g_walk_dn == a.g_walk_dn
    assert b.flanking_id.tolist() == a.flanking_id.tolist()
    assert [str(s) for s in b.flanking_seq] == [str(s) for s in a.flanking_seq]


def test_flanking_mate_recovery_downstream(tmp_path):
    """R1 bed-spanning + R2 fully downstream: R2's distal bases enter the flanking dict.

    R1 spans the polyG and reaches dn distance 65 (ref [0, 150) covers the
    upstream flank, polyG, and downstream flank up to ref 149).
    R2 sits at ref [170, 200) — purely past the downstream anchor, no
    polyG bases, no anchors. With mate-recovery, R2's bases populate dn
    distances 86..115 via the wide fetch. The gap between R1's reach
    (dist 65) and R2's coverage (dist 86) renders as ``.`` for distances
    66..85 — exactly the gap-preservation invariant we need.

    A deliberate SNV at ref pos 185 (dn distance 101) on R2 proves the
    base came from R2's alignment, not from any inference path.
    """
    seq_r1, rs1 = _canonical_read_seq(_REF_POLYG)
    # R2: ref [170, 200) — entirely past the downstream anchor.
    seq_r2 = REFERENCE[170:200]
    # Inject one SNV: read position 15 → ref pos 185 → dn distance 101.
    seq_r2 = _mutate_at(seq_r2, 15, "T")  # reference base at 185 is 'A'
    reads = [
        dict(qname="q1", read_seq=seq_r1, ref_start=rs1, mi="10004/A",
             is_read1=True, tlen=200),
        dict(qname="q1", read_seq=seq_r2, ref_start=170, mi="10004/A",
             is_read1=False, tlen=200),
    ]
    bam_path = _make_test_bam(tmp_path, reads)
    bed_path = _make_bed(tmp_path, [(CHROM, BED_START, BED_END, "Locus1")])

    panel = extract_panel(bam_path, bed_path)
    loc = panel.loci["Locus1"]

    # Width grows to include R2's distal reach: max dn distance is 115 → width 116.
    assert loc.flanking_dn_width == 116
    # Upstream width unchanged (R2 contributes nothing upstream).
    assert loc.flanking_up_width == 50

    # One fragment → one dedup row.
    assert len(loc.flanking_seq) == 1
    dn_part = str(loc.flanking_seq[0]).split("|")[1]
    assert len(dn_part) == 116

    # R1 contribution: dn distances 0..65, all 'A' (reference).
    assert dn_part[:66] == "A" * 66
    # Gap between R1's reach (≤ dist 65) and R2's coverage (≥ dist 86):
    # distances 66..85 — uncovered, must render as 20 dots.
    assert dn_part[66:86] == "." * 20
    # R2 contribution: distances 86..115, 'A'-flanked with one 'T' SNV at dist 101.
    assert dn_part[86:101] == "A" * 15
    assert dn_part[101] == "T"
    assert dn_part[102:116] == "A" * 14


def test_flanking_orphan_pair_dropped(tmp_path):
    """A fragment whose mates both miss the polyG bed is dropped.

    Setup: one bed-spanning filler fragment ``q_filler`` (so the anchor
    walker can reconstruct the reference window) plus an orphan
    fragment ``q_orphan`` with both mates entirely downstream of the
    polyG. Only the filler should survive — the orphan has no
    bed-spanning mate and cannot be MI-anchored.
    """
    seq_filler, rs = _canonical_read_seq(_REF_POLYG)
    seq_orphan = REFERENCE[170:200]
    reads = [
        dict(qname="q_filler", read_seq=seq_filler, ref_start=rs, mi="10004/A",
             is_read1=True, tlen=150),
        dict(qname="q_filler", read_seq=seq_filler, ref_start=rs, mi="10004/A",
             is_read1=False, tlen=150),
        dict(qname="q_orphan", read_seq=seq_orphan, ref_start=170, mi="20000/A",
             is_read1=True, tlen=30),
        dict(qname="q_orphan", read_seq=seq_orphan, ref_start=170, mi="20000/A",
             is_read1=False, tlen=30),
    ]
    bam_path = _make_test_bam(tmp_path, reads)
    bed_path = _make_bed(tmp_path, [(CHROM, BED_START, BED_END, "Locus1")])
    panel = extract_panel(bam_path, bed_path)
    loc = panel.loci["Locus1"]
    # Only the filler is in the accepted set; the orphan's MI never appears.
    assert set(loc.mi.tolist()) == {"10004"}


def test_extraction_schema_version_in_provenance(tmp_path):
    """Provenance records the schema version and min_base_q."""
    bed_path = _make_bed(tmp_path, [(CHROM, BED_START, BED_END, "Locus1")])
    bam_path = _make_test_bam(tmp_path, [])
    panel = extract_panel(bam_path, bed_path, min_base_q=25)
    assert panel.provenance.extraction_schema_version == "2"
    assert panel.provenance.min_base_q == 25
