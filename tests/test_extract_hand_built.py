"""Hand-built reads through the whole extraction, with a per-donor flank table.

Each hand-built case of `tests/_hand_built_reads.py` is written to a BAM with
every alignment an aligner can choose, on a "+" and a "-" copy of its locus.
`extract_panel` reads it with a flank table that holds the case's flanks.
Every kept alignment must give the true tuple, and every dropped one must
add to the counter of its drop reason.

The other tests check what the flank moves besides the tract: the phasing
window, the reference tract and its run lengths, and the checks of the flank
table against the BED and the reference window.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import pytest

import prolyg_phasing.io._flank_construction as fc
from prolyg_phasing.io._anchors import revcomp
from prolyg_phasing.io._flank_placement import SIDE_DROP_REASONS, FlankRecord
from prolyg_phasing.io.bam import extract_panel
from prolyg_phasing.io.panel import parse_run_lengths
from prolyg_phasing.phasing import offset_at_ref_pos, ref_pos_at_offset
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
    ASSIGNED_TUPLE,
    CASE_IDS,
    CASES,
    DROP_REASON_BY_FORM,
    _flank_record,
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
        flank_records = {f"{locus.name}{o}": (_flank_record(case, case.up, o),
                                             _flank_record(case, case.dn, o))
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
                "locus": locus, "side": side, "reference_window_status": "recovered",
                "ref_orientation": "", "flank_d": 0,
                "flank_bases": record.bases,
                "flank_ref_positions": ",".join(map(str, record.ref_positions)),
                "masked_offsets": ",".join(map(str, record.masked_offsets)),
                "no_read_error_candidate": 0, "n_candidates": 0, "n_reads": 0,
                "n_marker_placed": 0,
            })
    result = fc.FlankConstruction(sample, records, pd.DataFrame(), pd.DataFrame(rows),
                                  fc.ErrorRate(0.0, 0, 1, np.array([0.0])))
    return fc.write_flank_tables(result, prefix)[1]


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


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_extraction_gives_the_true_tuple_on_every_alignment(tmp_path, case: ReadCase):
    bam, bed, flanks = _write_case(tmp_path, case)
    panel = extract_panel(bam, bed, flanks)
    every = placements(case)
    forms = [p.form for p in every]
    # The alignments the read filter sends to the flank placement.
    placed = [i for i, p in enumerate(every) if _overlaps_tract(case, p)]
    assert placed
    for o in ORIENTATIONS:
        locus = panel.loci[f"{case.locus.name}{o}"]
        assert locus.reference_window_status == "recovered"
        assert locus.ref_orientation == o
        assert locus.n_alignments_overlap_locus == len(placed) + 1  # and the MI-less read
        kept = _rows_by_read(locus)
        if case.drop_reason is not None:
            assert kept == {}
            expected = Counter(
                DROP_REASON_BY_FORM.get(case.case_id, {}).get(forms[i], case.drop_reason)
                for i in placed)
            assert locus.n_alignments_drop_flank == len(placed)
            for reason in SIDE_DROP_REASONS:
                got = (getattr(locus, f"n_alignments_drop_upstream_{reason}")
                       + getattr(locus, f"n_alignments_drop_downstream_{reason}"))
                assert got == expected[reason], reason
            assert locus.n_alignments_drop_flanks_crossed == expected["flanks_crossed"]
            continue
        assert locus.n_alignments_drop_flank == 0
        expected_tuple = ASSIGNED_TUPLE.get(case.case_id, case.true_tuple)
        assert set(kept) == {f"{case.case_id}_{i}" for i in placed}
        for name, tract in kept.items():
            assert parse_run_lengths(tract) == expected_tuple, (o, forms[int(name.rsplit("_")[-1])])


def test_both_sides_count_when_both_fail(tmp_path):
    up151 = _up("PolyG151")
    case = _case("both_sides_fail", "PolyG151: G at the edge T, two downstream edits",
                 "PolyG151", up_ctx=up151[:-1] + "G", tract="G" * 12,
                 dn_ctx=_sub(_sub(_dn("PolyG151"), 3, "C"), 6, "A"),
                 drop_reason="g_at_run_side", true_tract=None)
    bam, bed, flanks = _write_case(tmp_path, case, forms={"gapless"})
    panel = extract_panel(bam, bed, flanks)
    for o in ORIENTATIONS:
        locus = panel.loci[f"PolyG151{o}"]
        assert locus.n_alignments_drop_flank == 1
        assert locus.n_alignments_drop_upstream_g_at_run_side == 1
        assert locus.n_alignments_drop_downstream_flank_not_placed == 1
        assert locus.n_alignments_drop_upstream_flank_not_placed == 0
        assert locus.n_alignments_drop_flanks_crossed == 0


# ---------------------------------------------------------------------------
# What a moved flank moves besides the tract
# ---------------------------------------------------------------------------

# PolyG5177 with a G at both edge bases: the upstream flank moves out 4
# bases, past the edge T and the GGG behind it, and the downstream flank
# moves out 1 base, past the edge A.
MOVED = next(c for c in CASES if c.case_id == "edge_variant_hom")


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_reference_tract_runs_between_moved_flanks(tmp_path, orientation):
    bam, bed, flanks = _write_case(tmp_path, MOVED)
    locus = extract_panel(bam, bed, flanks).loci[f"PolyG5177{orientation}"]
    # Reference: ...CTCAC T GGG T G11 A TAGAG...; the run-side bases are the
    # outer T upstream and the T after the edge A downstream.
    assert locus.reference_tract == "GGGT" + "G" * 11 + "A"
    assert locus.reference_run_lengths == (3, 11, 0)
    assert locus.upstream_flank == _flank_record(MOVED, MOVED.up, orientation)


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_phasing_window_starts_past_each_moved_flank(tmp_path, orientation):
    bam, bed, flanks = _write_case(tmp_path, MOVED)
    locus = extract_panel(bam, bed, flanks).loci[f"PolyG5177{orientation}"]
    up = _flank_record(MOVED, MOVED.up, orientation)
    dn = _flank_record(MOVED, MOVED.dn, orientation)
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
def test_masked_flank_position_is_not_in_the_phasing_window(tmp_path, orientation):
    """A position the donor carries a G at, inside the flank, is masked in the
    match and lies inside the flanked interval, so phasing does not see it."""
    case = next(c for c in CASES if c.case_id == "masked_plus_mismatch")
    bam, bed, flanks = _write_case(tmp_path, case)
    locus = extract_panel(bam, bed, flanks).loci[f"PolyG151{orientation}"]
    up = _flank_record(case, case.up, orientation)
    masked_position = up.ref_positions[case.up.masked_offsets[0]]
    for start, width in ((locus.flanking_up_ref_pos_start, locus.flanking_up_width),
                         (locus.flanking_dn_ref_pos_start, locus.flanking_dn_width)):
        assert offset_at_ref_pos(orientation, start, masked_position, width) is None
    nearest_up = ref_pos_at_offset(orientation, locus.flanking_up_ref_pos_start,
                                   locus.flanking_up_width - 1)
    assert nearest_up == up.outer_position + (-1 if orientation == "+" else 1)


# ---------------------------------------------------------------------------
# The flank table against the BED and the reference
# ---------------------------------------------------------------------------

INVARIANT = next(c for c in CASES if c.case_id == "invariant_side")


def _records(case: ReadCase) -> dict[str, tuple[FlankRecord, FlankRecord]]:
    return {f"{case.locus.name}{o}": (_flank_record(case, case.up, o),
                                     _flank_record(case, case.dn, o))
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


def test_flank_that_differs_from_the_reference_is_an_error(tmp_path):
    records = _records(INVARIANT)
    up, dn = records["PolyG151+"]
    records["PolyG151+"] = (dataclasses.replace(up, bases="C" + up.bases[1:]), dn)
    bam, bed, flanks = _write_case(tmp_path, INVARIANT, flank_records=records)
    with pytest.raises(ValueError, match="PolyG151[+]: the upstream flank"):
        extract_panel(bam, bed, flanks)


def test_provenance_names_the_flank_table_sample(tmp_path):
    bam, bed, flanks = _write_case(tmp_path, INVARIANT, sample="SD_NORMAL")
    assert extract_panel(bam, bed, flanks).provenance.flank_table_sample == "SD_NORMAL"
