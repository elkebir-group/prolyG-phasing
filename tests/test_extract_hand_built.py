"""Hand-built reads through the whole extraction, with a per-donor flank table.

Each hand-built case of `tests/_hand_built_reads.py` is written to a BAM with
every alignment an aligner can choose, on a "+" and a "-" copy of its locus.
`extract_panel` reads it with a flank table that holds the case's flanks.
Every kept alignment must give the case's expected tuple (see
`tests/test_flank_placement.py`), and every dropped one must add to the
counter of its drop reason.

The other tests check what the flank moves besides the tract: the phasing
window, the reference tract and its run lengths, and the checks of the flank
table against the BED and the reference window: each run-side base lies in
the window, and the upstream flank is upstream of the downstream flank.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pysam
import pytest

import prolyg_phasing.io._flank_construction as fc
from prolyg_phasing.io._anchors import revcomp
from prolyg_phasing.io._flank_placement import FlankRecord
from prolyg_phasing.io.bam import extract_panel
from prolyg_phasing.io.panel import parse_run_lengths
from prolyg_phasing.phasing import ref_pos_at_offset
from tests._counter_relations import (
    assert_counter_relations,
    counters_of,
    neither_flank_matched,
)
from tests._hand_built_reads import (
    CHROM_PAD,
    ReadCase,
    _case,
    _dn,
    _sub,
    _up,
    chromosome,
    placements,
    to_record,
)
from tests.test_flank_placement import (
    CASE_IDS,
    CASES,
    DROP_REASONS,
    expected_outcome,
    flank_record,
)

ORIENTATIONS = ("+", "-")


def _contig(orientation: str) -> str:
    return "plus" if orientation == "+" else "minus"


def _write_case(tmp_path: Path, case: ReadCase, *, sample: str = "NORMAL",
                flank_records: dict | None = None, forms: set[str] | None = None
                ) -> tuple[Path, Path, Path]:
    """BAM, BED and flank table of one case, on a "+" and a "-" contig.

    Each contig carries every alignment of the case's read (or those of
    `forms`), one read per alignment, and one whole-contig read without an
    MI tag that rebuilds the reference window from its MD tag; extraction
    skips it.
    """
    locus = case.locus
    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": _contig(o), "LN": len(chromosome(locus, o))} for o in ORIENTATIONS],
    })
    unsorted = tmp_path / "unsorted.bam"
    bam_path = tmp_path / "sample.bam"
    with pysam.AlignmentFile(str(unsorted), "wb", header=header) as out:
        for ref_id, orientation in enumerate(ORIENTATIONS):
            whole = chromosome(locus, orientation)
            seed = pysam.AlignedSegment(header)
            seed.query_name = "reference"
            seed.query_sequence = whole
            seed.query_qualities = pysam.qualitystring_to_array("I" * len(whole))
            seed.reference_id = ref_id
            seed.reference_start = 0
            seed.mapping_quality = 60
            seed.cigartuples = [(0, len(whole))]
            seed.set_tags([("MD", str(len(whole))), ("NM", 0)])
            out.write(seed)
            for i, p in enumerate(placements(case)):
                if forms is not None and p.form not in forms:
                    continue
                rec = to_record(header, case, p, orientation, f"{case.case_id}_{i}")
                rec.reference_id = ref_id
                out.write(rec)
    pysam.sort("-o", str(bam_path), str(unsorted))
    pysam.index(str(bam_path))

    tract_start = CHROM_PAD + len(locus.upstream)
    bed = tmp_path / "panel.bed"
    rows = []
    for o in ORIENTATIONS:
        n = len(chromosome(locus, o))
        start = tract_start if o == "+" else n - tract_start - len(locus.tract)
        rows.append(f"{_contig(o)}\t{start}\t{start + len(locus.tract)}\t{locus.name}{o}\n")
    bed.write_text("".join(rows))

    if flank_records is None:
        flank_records = {f"{locus.name}{o}": (flank_record(case, case.up, o),
                                             flank_record(case, case.dn, o))
                         for o in ORIENTATIONS}
    flanks = write_side_table(tmp_path / "normal", sample, flank_records)
    return bam_path, bed, flanks


def write_side_table(prefix: Path, sample: str,
                     records: dict[str, tuple[FlankRecord, FlankRecord]]) -> Path:
    """A side TSV in the form `build-flanks` writes, for the given flanks."""
    rows = []
    for locus, pair in records.items():
        for side, record in zip(("upstream", "downstream"), pair, strict=True):
            rows.append({
                "locus": locus, "side": side,
                "reference_window_status": ("recovered" if record.has_position
                                            else "no_reference_position"),
                "ref_orientation": "", "flank_bases": record.bases,
                "flank_ref_positions": ",".join(map(str, record.ref_positions)),
                "objective": 0.0, "n_reads": 0, "n_reads_placed": 0, "n_tied_pairs": 1,
                "n_reads_positioned": 0, "n_reads_at_position": 0,
            })
    result = fc.FlankConstruction(sample, records, pd.DataFrame(rows, columns=fc.SIDE_COLUMNS))
    return fc.write_flank_table(result, prefix)


def _overlaps_tract(case: ReadCase, placement) -> bool:
    """Whether the alignment overlaps the BED interval, the reference tract.

    The extraction's read filter keeps only such alignments. A form that
    soft-clips the whole tract does not overlap it.
    """
    span = sum(length for op, length in placement.cigar if op in (0, 2))
    tract_start = len(case.locus.upstream)
    return (placement.ref_start < tract_start + len(case.locus.tract)
            and placement.ref_start + span > tract_start)


def _rows_by_read(locus) -> dict[str, str]:
    """Tract of each kept read, keyed by its read name (the MI prefix)."""
    out = {}
    for mi, seq, count in zip(locus.mi, locus.seq, locus.count, strict=True):
        assert count == 1
        out[str(mi)] = str(seq)
    return out


# ---------------------------------------------------------------------------
# Every hand-built case, every alignment, through the extraction
# ---------------------------------------------------------------------------


def _drop_counter(reason: str) -> str:
    return f"n_alignments_drop_{reason}"


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_extraction_gives_the_expected_outcome_on_every_alignment(tmp_path, case: ReadCase):
    bam, bed, flanks = _write_case(tmp_path, case)
    panel = extract_panel(bam, bed, flanks)
    every = placements(case)
    forms = [p.form for p in every]
    # The alignments the read filter sends to the flank placement.
    placed = [i for i, p in enumerate(every) if _overlaps_tract(case, p)]
    assert placed
    want = expected_outcome(case)
    for o in ORIENTATIONS:
        locus = panel.loci[f"{case.locus.name}{o}"]
        assert locus.reference_window_status == "recovered"
        assert locus.ref_orientation == o
        assert locus.n_alignments_overlap_locus == len(placed) + 1  # and the MI-less read
        assert_counter_relations(locus)
        kept = _rows_by_read(locus)
        if isinstance(want, str):
            assert kept == {}
            assert locus.n_alignments_drop_flank == len(placed)
            for reason in DROP_REASONS:
                got = getattr(locus, _drop_counter(reason))
                assert got == (len(placed) if reason == want else 0), reason
            continue
        assert locus.n_alignments_drop_flank == 0
        assert locus.n_alignments_placed == len(placed)
        assert set(kept) == {f"{case.case_id}_{i}" for i in placed}
        for name, tract in kept.items():
            assert parse_run_lengths(tract) == want, (o, forms[int(name.rsplit("_")[-1])])


def test_both_sides_count_when_both_fail(tmp_path):
    up151 = _up("PolyG151")
    case = _case("both_sides_fail", "PolyG151: G at the edge T, two downstream edits",
                 "PolyG151", up_ctx=up151[:-1] + "G", tract="G" * 12,
                 dn_ctx=_sub(_sub(_dn("PolyG151"), 3, "C"), 6, "A"),
                 drop_reason="upstream_flank_not_placed", true_tract=None)
    bam, bed, flanks = _write_case(tmp_path, case, forms={"gapless"})
    panel = extract_panel(bam, bed, flanks)
    for o in ORIENTATIONS:
        locus = panel.loci[f"PolyG151{o}"]
        assert locus.n_alignments_drop_flank == 1
        assert locus.n_alignments_placed == 0
        assert locus.n_alignments_drop_upstream_flank_not_placed == 1
        assert locus.n_alignments_drop_downstream_flank_not_placed == 1
        assert locus.n_alignments_drop_no_placement == 0
        assert locus.n_alignments_drop_no_bed_base == 0
        assert locus.n_alignments_drop_negative_value == 0
        # One alignment, in both side counters: the side counters sum past the drops.
        assert_counter_relations(locus)
        assert neither_flank_matched(counters_of(locus)) == 1


# ---------------------------------------------------------------------------
# What a moved flank moves besides the tract
# ---------------------------------------------------------------------------

# PolyG5177 with a G at both edge bases: the upstream flank moves out 4
# bases, past the edge T and the GGG behind it, and the downstream flank
# moves out 1 base, past the edge A.
MOVED = next(c for c in CASES if c.case_id == "edge_variant_hom")
INVARIANT = next(c for c in CASES if c.case_id == "invariant_side")


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_reference_tract_runs_between_moved_flanks(tmp_path, orientation):
    bam, bed, flanks = _write_case(tmp_path, MOVED)
    locus = extract_panel(bam, bed, flanks).loci[f"PolyG5177{orientation}"]
    # Reference: ...CTCAC T GGG T G11 A TAGAG...; the run-side bases are the
    # outer T upstream and the T after the edge A downstream.
    assert locus.reference_tract == "GGGT" + "G" * 11 + "A"
    assert locus.reference_run_lengths == (3, 11, 0)
    assert locus.upstream_flank == flank_record(MOVED, MOVED.up, orientation)


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_phasing_window_starts_past_each_moved_flank(tmp_path, orientation):
    bam, bed, flanks = _write_case(tmp_path, MOVED)
    locus = extract_panel(bam, bed, flanks).loci[f"PolyG5177{orientation}"]
    up = flank_record(MOVED, MOVED.up, orientation)
    dn = flank_record(MOVED, MOVED.dn, orientation)
    step = 1 if orientation == "+" else -1
    # The flanking base nearest each flank is the reference base just past
    # the flank's outer base.
    up_w, dn_w = locus.flanking_up_width, locus.flanking_dn_width
    nearest_up = ref_pos_at_offset(orientation, locus.flanking_up_ref_pos_start, up_w - 1)
    nearest_dn = ref_pos_at_offset(orientation, locus.flanking_dn_ref_pos_start, 0)
    assert nearest_up == up.outer_position - step
    assert nearest_dn == dn.outer_position + step
    # The read starts 5 bases into the 30 upstream bases; the flank starts
    # 4 + 10 bases from the edge base. Downstream it keeps 25 bases past
    # the run-side base, 10 of them the flank.
    assert (up_w, dn_w) == (30 - 5 - 4 - 10, 25 - 10)

    # The flanking string of a gapless-in-the-flanks read holds the reference
    # bases at the positions the window maps them to.
    chrom = chromosome(MOVED.locus, orientation)
    forms = [p.form for p in placements(MOVED)]
    name = f"{MOVED.case_id}_{forms.index('gap_shift')}"
    row = list(locus.mi).index(name)
    up_text, dn_text = str(locus.flanking_seq[locus.flanking_id[row]]).split("|")
    for start, text in ((locus.flanking_up_ref_pos_start, up_text),
                        (locus.flanking_dn_ref_pos_start, dn_text)):
        for i, base in enumerate(text):
            if base == ".":
                continue
            ref = chrom[ref_pos_at_offset(orientation, start, i)]
            assert base == (ref if orientation == "+" else revcomp(ref))


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_a_flank_with_a_gap_against_the_reference(tmp_path, orientation):
    """A donor flank with TT where the reference has TTT covers 11 reference
    bases. Extraction accepts it, the reads place with the 1-base gap, and the
    phasing window starts past the outer base."""
    case = INVARIANT
    donor_up = "CATATTAGAT"
    n = len(case.locus.reference)
    records = {}
    for o in ORIENTATIONS:
        # Reference CATATTTAGAT at polyG indices 19 to 29.
        positions = tuple(CHROM_PAD + i if o == "+" else CHROM_PAD + n - 1 - i
                          for i in range(19, 30))
        up = FlankRecord(donor_up, positions, len(positions) - 1)
        records[f"PolyG151{o}"] = (up, flank_record(case, case.dn, o))
    bam, bed, flanks = _write_case(tmp_path, case, flank_records=records)
    locus = extract_panel(bam, bed, flanks).loci[f"PolyG151{orientation}"]
    up = records[f"PolyG151{orientation}"][0]
    assert len(up.ref_positions) == 11
    assert locus.upstream_flank == up
    assert {parse_run_lengths(t) for t in locus.seq} == {(12,)}
    nearest_up = ref_pos_at_offset(orientation, locus.flanking_up_ref_pos_start,
                                   locus.flanking_up_width - 1)
    assert nearest_up == up.outer_position + (-1 if orientation == "+" else 1)


# ---------------------------------------------------------------------------
# The flank table against the BED and the reference
# ---------------------------------------------------------------------------


def _records(case: ReadCase) -> dict[str, tuple[FlankRecord, FlankRecord]]:
    return {f"{case.locus.name}{o}": (flank_record(case, case.up, o),
                                     flank_record(case, case.dn, o))
            for o in ORIENTATIONS}


def test_flank_table_missing_a_locus_is_an_error(tmp_path):
    records = _records(INVARIANT)
    del records["PolyG151-"]
    bam, bed, flanks = _write_case(tmp_path, INVARIANT, flank_records=records)
    with pytest.raises(ValueError, match=r"no flanks for 1 BED loci: \['PolyG151-'\]"):
        extract_panel(bam, bed, flanks)


def test_flank_table_with_an_extra_locus_is_an_error(tmp_path):
    records = _records(INVARIANT)
    records["PolyG9999"] = records["PolyG151+"]
    bam, bed, flanks = _write_case(tmp_path, INVARIANT, flank_records=records)
    with pytest.raises(ValueError, match=r"loci not in the BED: \['PolyG9999'\]"):
        extract_panel(bam, bed, flanks)


def test_a_run_side_base_outside_the_reference_window_breaks_an_invariant(tmp_path):
    records = _records(INVARIANT)
    up, dn = records["PolyG151+"]
    shifted = tuple(p + 200 for p in dn.ref_positions)
    records["PolyG151+"] = (up, dataclasses.replace(dn, ref_positions=shifted))
    bam, bed, flanks = _write_case(tmp_path, INVARIANT, flank_records=records)
    with pytest.raises(RuntimeError, match="PolyG151[+]: the downstream flank's run-side base "
                                         "at reference position .* is outside"):
        extract_panel(bam, bed, flanks)


def test_an_upstream_flank_past_the_downstream_flank_breaks_an_invariant(tmp_path):
    records = _records(INVARIANT)
    up, dn = records["PolyG151+"]
    shift = dn.run_side_position - up.run_side_position
    moved = tuple(p + shift for p in up.ref_positions)
    records["PolyG151+"] = (dataclasses.replace(up, ref_positions=moved), dn)
    bam, bed, flanks = _write_case(tmp_path, INVARIANT, flank_records=records)
    with pytest.raises(RuntimeError, match="PolyG151[+]: the upstream flank is not upstream"):
        extract_panel(bam, bed, flanks)


def test_a_locus_with_no_flank_pair_has_that_status_and_no_reads(tmp_path):
    bam, bed, flanks = _write_case(tmp_path, INVARIANT)
    table = pd.read_csv(flanks, sep="\t", dtype=str, keep_default_na=False)
    plus = table.locus == "PolyG151+"
    table.loc[plus, "reference_window_status"] = "no_flank_pair"
    table.loc[plus, ["flank_bases", "flank_ref_positions"]] = ""
    table.to_csv(flanks, sep="\t", index=False)
    panel = extract_panel(bam, bed, flanks)
    locus = panel.loci["PolyG151+"]
    assert locus.reference_window_status == "no_flank_pair"
    assert len(locus.seq) == 0 and locus.upstream_flank is None
    assert (locus.flanking_up_ref_pos_start, locus.flanking_dn_ref_pos_start) == (-1, -1)
    assert panel.loci["PolyG151-"].reference_window_status == "recovered"


def test_an_mi_less_read_counts_as_a_drop(tmp_path):
    """The whole-contig read of each contig overlaps the BED interval and has
    no MI tag. Overlap less the MAPQ, SA, TLEN and MI drops is the reads that
    reach the flank step: those placed plus those the flanks drop."""
    bam, bed, flanks = _write_case(tmp_path, INVARIANT)
    panel = extract_panel(bam, bed, flanks)
    for o in ORIENTATIONS:
        locus = panel.loci[f"PolyG151{o}"]
        assert locus.n_alignments_drop_no_mi == 1
        reach_flanks = (locus.n_alignments_overlap_locus - locus.n_alignments_drop_mapq
                        - locus.n_alignments_drop_sa - locus.n_alignments_drop_tlen
                        - locus.n_alignments_drop_no_mi)
        assert reach_flanks == locus.n_alignments_placed + locus.n_alignments_drop_flank
        assert reach_flanks == sum(_overlaps_tract(INVARIANT, p) for p in placements(INVARIANT))


def test_provenance_names_the_flank_table_sample(tmp_path):
    bam, bed, flanks = _write_case(tmp_path, INVARIANT, sample="SD_NORMAL")
    assert extract_panel(bam, bed, flanks).provenance.flank_table_sample == "SD_NORMAL"
