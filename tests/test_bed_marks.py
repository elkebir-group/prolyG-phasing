"""The tract must hold a read base that the aligner aligns inside the BED interval.

Hand-built reads on synthetic loci, written to a BAM on a "+" and a "-"
contig, as `tests/_hand_built_reads.py` does for the hg38 cases:

- a copy of the flank-tract-flank pattern upstream of the locus's G run, the
  shape of PolyG4320: a read that holds only the copy has no placement, and
  a read that holds both copies places at the BED copy;
- a read whose G run lies wholly in an insertion: no base is marked;
- a locus where the optimum with every base marked surrounds a G run
  outside the BED interval, and the optimum with the BED marks surrounds
  the BED run.
"""

from __future__ import annotations

from pathlib import Path

import pysam
import pytest

import prolyg_phasing.io._flank_construction as fc
from prolyg_phasing.io._flank_placement import FlankRecord, place_flanks, place_polyg_reads
from prolyg_phasing.io.bam import extract_panel
from prolyg_phasing.io.panel import parse_run_lengths
from tests import _slow_flank_eval as slow
from tests._hand_built_reads import (
    DEL,
    INS,
    LOCI,
    LocusSequence,
    M,
    Placement,
    ReadCase,
    bed_interval,
    chromosome,
    downstream_flank,
    to_record,
    upstream_flank,
)
from tests.test_extract_hand_built import write_side_table
from tests.test_flank_placement import flank_record

ORIENTATIONS = ("+", "-")

# The PolyG4320 flanks and tract. The copy upstream of the locus holds a
# longer tract, G8 A G, so without the BED condition a read that holds both
# copies places at the copy.
COPY_UP, COPY_DN = "CTGCCATCCT", "TTGTAGCTGC"
COPY = COPY_UP + "GGGGGGGGAG" + COPY_DN
LOCUS_4320 = LocusSequence(
    "Copy4320", "+", "ACATTCAATCAT" + COPY + "ATCAACTTAC" + COPY_UP, "GGGGGAG",
    COPY_DN + "ACTTACAATCATCAATCTACATCAT")

# A G14 run between its own flanks, upstream of the BED run G8.
RUN_A_UP, RUN_A_DN = "TTACCAGTCA", "CCATACAATC"
BED_UP, BED_DN = "ACTTCAGCAT", "CAAATTTTAT"
LOCUS_TWO_RUNS = LocusSequence(
    "TwoRuns", "+", "ACATTCAATC" + RUN_A_UP + "G" * 14 + RUN_A_DN + "ATCAACTTAC" + BED_UP,
    "G" * 8, BED_DN + "ACCAATCATTACATCTACAT")


def _contig(orientation: str) -> str:
    return "plus" if orientation == "+" else "minus"


def _case(locus: LocusSequence, read: str, ref_start: int) -> ReadCase:
    return ReadCase("read", "", locus, upstream_flank(locus, 0), downstream_flank(locus, 0),
                    read, ref_start, None)


def _records(locus: LocusSequence, reads, orientation: str,
             header: pysam.AlignmentHeader | None = None, ref_id: int = 0
             ) -> list[pysam.AlignedSegment]:
    """`reads` as ``(read, ref_start, cigar, copies)`` in polyG orientation."""
    if header is None:
        header = pysam.AlignmentHeader.from_dict({
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"SN": "synth", "LN": len(chromosome(locus, orientation))}],
        })
    out = []
    for j, (read, ref_start, cigar, copies) in enumerate(reads):
        for c in range(copies):
            rec = to_record(header, _case(locus, read, ref_start),
                            Placement("hand", ref_start, cigar), orientation, f"r{j}_{c}")
            rec.reference_id = ref_id
            out.append(rec)
    return out


def _write_bam(tmp_path: Path, locus: LocusSequence, reads) -> tuple[Path, Path]:
    """BAM and BED of `reads` on a "+" and a "-" contig.

    One whole-contig read with no MI tag rebuilds the reference window; the
    flank search and the extraction skip it.
    """
    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": _contig(o), "LN": len(chromosome(locus, o))} for o in ORIENTATIONS],
    })
    unsorted, bam_path = tmp_path / "unsorted.bam", tmp_path / "sample.bam"
    with pysam.AlignmentFile(str(unsorted), "wb", header=header) as out:
        for ref_id, o in enumerate(ORIENTATIONS):
            whole = chromosome(locus, o)
            seed = pysam.AlignedSegment(header)
            seed.query_name = "reference"
            seed.query_sequence = whole
            seed.query_qualities = pysam.qualitystring_to_array("I" * len(whole))
            seed.reference_id = ref_id
            seed.reference_start = 0
            seed.mapping_quality = 60
            seed.cigartuples = [(M, len(whole))]
            seed.set_tags([("MD", str(len(whole))), ("NM", 0)])
            out.write(seed)
            for rec in _records(locus, reads, o, header, ref_id):
                out.write(rec)
    pysam.sort("-o", str(bam_path), str(unsorted))
    pysam.index(str(bam_path))
    bed = tmp_path / "panel.bed"
    bed.write_text("".join(
        f"{_contig(o)}\t{bed_interval(locus, o)[0]}\t{bed_interval(locus, o)[1]}"
        f"\t{locus.name}{o}\n" for o in ORIENTATIONS))
    return bam_path, bed


def _gapless(locus: LocusSequence, start: int, end: int, copies: int = 1):
    return (locus.reference[start:end], start, ((M, end - start),), copies)


def _place(locus: LocusSequence, reads, orientation: str):
    case = _case(locus, "", 0)
    return place_flanks(
        _records(locus, reads, orientation),
        upstream=flank_record(case, case.up, orientation),
        downstream=flank_record(case, case.dn, orientation),
        ref_orientation=orientation,
        bed_start=bed_interval(locus, orientation)[0],
        bed_end=bed_interval(locus, orientation)[1],
    )


# ---------------------------------------------------------------------------
# A copy of the pattern outside the BED interval
# ---------------------------------------------------------------------------

_N_UP = len(LOCUS_4320.upstream)
BOTH = _gapless(LOCUS_4320, 0, len(LOCUS_4320.reference))
COPY_ONLY = _gapless(LOCUS_4320, 0, _N_UP + 3)  # ends three bases into the BED run


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_a_read_that_holds_only_the_copy_has_no_placement(orientation):
    result = _place(LOCUS_4320, [COPY_ONLY], orientation)
    assert result.upstream_matched[0] and result.downstream_matched[0]
    assert result.no_bed_base[0] and not result.placed[0]
    # Without the condition the read places at the copy.
    read = COPY_ONLY[0]
    every = place_polyg_reads([read], COPY_UP, COPY_DN, slow.marks([read]))
    assert every.tracts[0] == "GGGGGGGGAG"


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_a_read_that_holds_both_copies_places_at_the_bed_copy(orientation):
    result = _place(LOCUS_4320, [BOTH], orientation)
    assert result.kept[0] and result.tracts[0] == "GGGGGAG"
    assert result.upstream_run_side[0] == _N_UP - 1
    # Without the condition the higher-value copy wins.
    read = BOTH[0]
    every = place_polyg_reads([read], COPY_UP, COPY_DN, slow.marks([read]))
    assert every.tracts[0] == "GGGGGGGGAG"


def test_extraction_counts_the_copy_only_read(tmp_path):
    bam, bed = _write_bam(tmp_path, LOCUS_4320, [BOTH, COPY_ONLY])
    case = _case(LOCUS_4320, "", 0)
    flanks = write_side_table(tmp_path / "normal", "NORMAL", {
        f"Copy4320{o}": (flank_record(case, case.up, o), flank_record(case, case.dn, o))
        for o in ORIENTATIONS})
    panel = extract_panel(bam, bed, flanks)
    for o in ORIENTATIONS:
        locus = panel.loci[f"Copy4320{o}"]
        assert locus.n_alignments_placed == 1
        assert locus.n_alignments_drop_flank == 1
        assert locus.n_alignments_drop_no_bed_base == 1
        assert locus.n_alignments_drop_no_placement == 0
        assert {parse_run_lengths(str(t)) for t in locus.seq} == {(5, 1)}


# ---------------------------------------------------------------------------
# A G run in an insertion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_a_run_wholly_in_an_insertion_has_no_marked_base(orientation):
    """PolyG151: the aligner puts the read's G12 in an insertion and deletes
    the reference G12, so no read base aligns inside the BED interval."""
    locus = LOCI["PolyG151"]
    n_up, n_dn = len(locus.upstream), len(locus.downstream)
    read = locus.upstream + "G" * 12 + locus.downstream
    cigar = ((M, n_up), (INS, 12), (DEL, 12), (M, n_dn))
    result = _place(locus, [(read, 0, cigar, 1), (read, 0, ((M, len(read)),), 1)],
                    orientation)
    assert result.no_bed_base[0] and not result.placed[0]
    assert result.kept[1] and result.tracts[1] == "G" * 12


# ---------------------------------------------------------------------------
# The optimum moves to the BED run
# ---------------------------------------------------------------------------


def test_the_optimum_surrounds_the_bed_run_not_a_longer_run_outside(tmp_path):
    """With every base marked, the G14 run outside the BED interval gives the
    larger objective. With the BED marks, only the BED run's flanks place."""
    reads = [_gapless(LOCUS_TWO_RUNS, 0, len(LOCUS_TWO_RUNS.reference), 20)]
    strings = [reads[0][0]] * 20
    every = fc.solve_flank_pair(strings, slow.marks(strings))
    assert (every.upstream, every.downstream) == (RUN_A_UP, RUN_A_DN)
    assert every.objective_half_units == 20 * 28

    bam, bed = _write_bam(tmp_path, LOCUS_TWO_RUNS, reads)
    result = fc.build_flanks(bam, bed, n_workers=1, sample="NORMAL")
    sides = result.sides.set_index(["locus", "side"])
    for o in ORIENTATIONS:
        up, dn = result.records[f"TwoRuns{o}"]
        assert (up.bases, dn.bases) == (BED_UP, BED_DN)
        row = sides.loc[(f"TwoRuns{o}", "upstream")]
        assert (row.objective, row.n_reads_placed) == (20 * 8, 20)


# ---------------------------------------------------------------------------
# A flank side with no reference position
# ---------------------------------------------------------------------------

SOFT = 4


def test_a_soft_clipped_downstream_flank_has_no_position_and_keeps_its_tracts(tmp_path):
    """PolyG151 with every read soft-clipped from the downstream edge base on,
    as at PolyG7425: no read aligns the downstream flank. The side keeps its
    flank, has no reference position, and gives no flanking bases. The
    upstream side and the tracts are as with an aligned flank."""
    locus = LOCI["PolyG151"]
    aligned = len(locus.upstream) + len(locus.tract)
    clipped = (locus.reference, 0, ((M, aligned), (SOFT, len(locus.downstream))), 20)
    bam, bed = _write_bam(tmp_path, locus, [clipped])
    result = fc.build_flanks(bam, bed, n_workers=1, sample="NORMAL")
    sides = result.sides.set_index(["locus", "side"])
    for o in ORIENTATIONS:
        up, dn = result.records[f"PolyG151{o}"]
        assert up.has_position and not dn.has_position
        assert dn.bases == locus.downstream[:10]
        row = sides.loc[(f"PolyG151{o}", "downstream")]
        assert (row.reference_window_status, row.flank_ref_positions,
                row.n_reads_positioned) == ("no_reference_position", "", 0)
    flanks = fc.write_flank_table(result, tmp_path / "normal")

    # The same reads, aligned in full, with the same flank bases at their
    # reference positions.
    case = _case(locus, "", 0)
    full_dir = tmp_path / "full"
    full_dir.mkdir()
    full_bam, full_bed = _write_bam(full_dir, locus, [_gapless(locus, 0, len(locus.reference),
                                                               20)])
    full_flanks = write_side_table(full_dir / "normal", "NORMAL", {
        f"PolyG151{o}": (flank_record(case, case.up, o), flank_record(case, case.dn, o))
        for o in ORIENTATIONS})
    panel = extract_panel(bam, bed, flanks)
    full = extract_panel(full_bam, full_bed, full_flanks)
    for o in ORIENTATIONS:
        got, want = panel.loci[f"PolyG151{o}"], full.loci[f"PolyG151{o}"]
        assert sorted(map(str, got.seq)) == sorted(map(str, want.seq)) == ["G" * 12] * 20
        assert got.n_alignments_placed == 20
        # No downstream flanking bases, and no reference tract.
        assert (got.flanking_dn_width, got.flanking_dn_ref_pos_start) == (0, -1)
        assert (got.reference_tract, got.reference_run_lengths) == ("", ())
        assert want.reference_tract == "G" * 12 and want.flanking_dn_width > 0
        # The upstream side is intact.
        assert (got.flanking_up_width, got.flanking_up_ref_pos_start) == \
            (want.flanking_up_width, want.flanking_up_ref_pos_start)
        got_up = {str(s).split("|")[0] for s in got.flanking_seq}
        want_up = {str(s).split("|")[0] for s in want.flanking_seq}
        assert got_up == want_up
        assert {str(s).split("|")[1] for s in got.flanking_seq} == {""}


def test_extraction_with_an_upstream_side_without_a_position(tmp_path):
    """The flank table gives the upstream flank no position; the reads align
    in full. Only the upstream flanking is lost."""
    locus = LOCI["PolyG151"]
    case = _case(locus, "", 0)
    reads = [_gapless(locus, 0, len(locus.reference), 5)]
    bam, bed = _write_bam(tmp_path, locus, reads)
    records = {f"PolyG151{o}": (FlankRecord(case.up.bases, (), -1),
                                flank_record(case, case.dn, o)) for o in ORIENTATIONS}
    panel = extract_panel(bam, bed, write_side_table(tmp_path / "normal", "NORMAL", records))
    for o in ORIENTATIONS:
        got = panel.loci[f"PolyG151{o}"]
        assert {str(s) for s in got.seq} == {"G" * 12}
        assert (got.flanking_up_width, got.flanking_up_ref_pos_start) == (0, -1)
        assert got.flanking_dn_width > 0
        assert not got.upstream_flank.has_position
