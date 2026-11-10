"""Tests for flank construction from the matched normal.

The candidate list, the call, the error rate and the flank choice are tested
on hand-set counts. The count is tested on hand-built reads with every
alignment an aligner can choose (`tests/_hand_built_reads.py`), in both
locus orientations. One end-to-end test writes a small BAM, runs the
parallel count pass, and checks the flank records against the flanks that
the hand-built cases set by hand, through the side TSV.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import pytest

import prolyg_phasing.io._flank_construction as fc
from prolyg_phasing.io._anchors import revcomp
from prolyg_phasing.io._flank_placement import place_flanks, polyg_alignments, read_flank_table
from tests._hand_built_reads import (
    CHROM_PAD,
    FLANK_LEN,
    LOCI,
    ReadCase,
    chromosome,
    downstream_flank,
    placements,
    run_tuple,
    to_record,
    upstream_flank,
)

ORIENTATIONS = ("+", "-")


def _polyg_chromosome(locus, orientation: str = "+") -> str:
    """The synthetic chromosome of one orientation, read in polyG orientation."""
    chrom = chromosome(locus, orientation)
    return chrom if orientation == "+" else revcomp(chrom)


def _edges(locus) -> dict[str, int]:
    """Window index of each edge base on the synthetic chromosome, polyG orientation."""
    up = CHROM_PAD + len(locus.upstream) - 1
    return {"upstream": up, "downstream": up + 1 + len(locus.tract)}


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


def test_candidates_skip_g_and_stop_at_the_span():
    # PolyG5177 upstream, outward from the edge T: T GGG T C A C T C G A C A T G A G G A ...
    locus = LOCI["PolyG5177"]
    window = _polyg_chromosome(locus)
    side = fc.list_candidates(window, _edges(locus)["upstream"], upstream=True)
    outward = locus.upstream[::-1]
    expected_d = [d for d in range(20) if outward[d] != "G"]
    assert side.d == tuple(expected_d)
    assert side.d[:3] == (0, 4, 5)  # the GGG behind the edge T is not a candidate
    assert max(side.d) < 20
    assert outward[20] != "G"  # a non-G base at d = 20 exists and is not listed
    assert side.ref_bases == "".join(outward[d] for d in expected_d)
    # The outer marker is the 10 bases past the span.
    assert side.marker == outward[20:30][::-1]

    down = fc.list_candidates(window, _edges(locus)["downstream"], upstream=False)
    expected_dn = [d for d in range(20) if locus.downstream[d] != "G"]
    assert down.d == tuple(expected_dn)
    assert down.marker == locus.downstream[20:30]


def test_candidates_mark_a_neighboring_reference_ggg():
    locus = LOCI["PolyG5177"]
    window = _polyg_chromosome(locus)
    side = fc.list_candidates(window, _edges(locus)["upstream"], upstream=True)
    near = dict(zip(side.d, side.near_ggg, strict=True))
    assert near[0]  # the edge T sits between the repeat and a GGG
    assert near[4]  # the T outside the GGG
    assert not near[5]


def test_candidate_span_shrinks_to_keep_the_marker_in_the_window():
    window = "ACGTACGTACTTTT" + "G" * 10 + "TTTTACGTACGTAC"
    side = fc.list_candidates(window, 13, upstream=True)
    assert side.d == (0, 1, 2, 3)  # 14 bases before the run leave room for 4 + 10
    assert side.marker == window[:10]


# ---------------------------------------------------------------------------
# Call and error rate
# ---------------------------------------------------------------------------


def _g_share_limit(e: float) -> float:
    """G share at which p = e and p = 0.5 explain the reads equally well."""
    return math.log(2 * (1 - e)) / math.log((1 - e) / e)


def test_call_flips_at_the_limit_that_e_sets():
    e = 3.6e-3
    limit = _g_share_limit(e)
    assert 0.12 < limit < 0.125  # e = 3.6e-3 puts the limit near a G share of 0.12
    n = np.array([2000, 2000])
    x = np.array([math.floor(limit * 2000) - 1, math.ceil(limit * 2000) + 1])
    _, _, g = fc.call_candidates(n, x, e)
    assert g.tolist() == [False, True]


def test_call_takes_the_better_donor_explanation():
    e = 3.6e-3
    ll_error, ll_g, g = fc.call_candidates(np.array([100, 100]), np.array([50, 100]), e)
    assert g.all()
    assert ll_g[0] == pytest.approx(100 * math.log(0.5))
    assert ll_g[1] == pytest.approx(100 * math.log(1 - e))


def test_call_without_reads_is_read_error():
    ll_error, ll_g, g = fc.call_candidates(np.array([0]), np.array([0]), 3.6e-3)
    assert ll_error[0] == 0 and ll_g[0] == 0
    assert not g[0]


def _pool(*rows):
    d, near, n, x = (np.array(c) for c in zip(*rows, strict=True))
    return fc.estimate_error_rate(d, near, n, x)


# (d, near_ggg, n, x). The first row is always pooled; the second row breaks
# one condition, then meets it at the boundary.
_POOLED = (12, False, 1000, 2)


@pytest.mark.parametrize(
    ("excluded", "pooled"),
    [
        ((9, False, 100, 1), (10, False, 100, 1)),       # distance from the repeat
        ((15, True, 100, 1), (15, False, 100, 1)),       # next to a reference GGG
        ((15, False, 19, 0), (15, False, 20, 0)),        # read count
        ((15, False, 100, 5), (15, False, 100, 4)),      # G share 0.05 against 0.04
    ],
)
def test_error_rate_exclusions(excluded, pooled):
    out = _pool(_POOLED, excluded)
    assert (out.x, out.n, out.n_positions) == (2, 1000, 1)
    kept = _pool(_POOLED, pooled)
    assert (kept.x, kept.n, kept.n_positions) == (2 + pooled[3], 1000 + pooled[2], 2)
    assert kept.e == kept.x / kept.n


# ---------------------------------------------------------------------------
# Flank choice
# ---------------------------------------------------------------------------


def _choose(name: str, side_name: str, g_in_donor_d: set[int], orientation: str = "+"):
    locus = LOCI[name]
    window = _polyg_chromosome(locus, orientation)
    side = fc.list_candidates(window, _edges(locus)[side_name], upstream=side_name == "upstream")
    calls = np.array([d in g_in_donor_d for d in side.d])
    record, d, flag = fc.choose_flank(side, calls, window, ref_orientation=orientation,
                                      window_start=0)
    return locus, record, d, flag


def test_flank_stays_at_an_invariant_edge():
    locus, record, d, flag = _choose("PolyG151", "upstream", set())
    assert (d, flag) == (0, False)
    assert record.bases == upstream_flank(locus, 0).bases
    assert record.run_side_offset == FLANK_LEN - 1
    assert record.masked_offsets == ()


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_flank_moves_past_g_candidates(orientation):
    # PolyG5177: the edge T (d = 0) upstream and the edge A downstream are G in the donor.
    locus, up, d_up, _ = _choose("PolyG5177", "upstream", {0}, orientation)
    _, dn, d_dn, _ = _choose("PolyG5177", "downstream", {0}, orientation)
    assert (d_up, d_dn) == (4, 1)
    assert up.bases == upstream_flank(locus, 4).bases
    assert dn.bases == downstream_flank(locus, 1).bases
    # Reference positions point at the flank bases on the chromosome in the
    # BAM's orientation: the same bases at "+", their complements at "-".
    chrom = chromosome(locus, orientation)
    for record in (up, dn):
        stored = "".join(chrom[p] for p in record.ref_positions)
        assert stored == (record.bases if orientation == "+" else revcomp(record.bases)[::-1])


def test_flank_masks_a_g_candidate_inside_it():
    # PolyG4732 upstream, outward: A(d=0) G A(d=2) C(d=3) A(d=4). With G in the
    # donor at d = 0 and d = 4, the run-side base is the A at d = 2, and d = 4
    # falls inside the flank.
    locus = LOCI["PolyG4732"]
    window = _polyg_chromosome(locus)
    side = fc.list_candidates(window, _edges(locus)["upstream"], upstream=True)
    assert side.d[:4] == (0, 2, 3, 4)
    calls = np.array([d in {0, 4} for d in side.d])
    record, d, flag = fc.choose_flank(side, calls, window, ref_orientation="+",
                                      window_start=0)
    assert (d, flag) == (2, False)
    # d = 4 is two bases out from the run-side base: offset 9 - 2 = 7.
    assert record.masked_offsets == (7,)
    assert record.bases == upstream_flank(locus, 2).bases


def test_flank_flags_a_side_with_no_read_error_candidate():
    locus = LOCI["PolyG151"]
    window = _polyg_chromosome(locus)
    side = fc.list_candidates(window, _edges(locus)["downstream"], upstream=False)
    record, d, flag = fc.choose_flank(side, np.ones(len(side.d), dtype=bool), window,
                                      ref_orientation="+", window_start=0)
    assert (d, flag) == (0, True)
    assert record.bases == downstream_flank(locus, 0).bases
    # Every other candidate inside the flank is masked.
    assert record.masked_offsets == tuple(d for d in side.d if 0 < d < FLANK_LEN)


# ---------------------------------------------------------------------------
# Count
# ---------------------------------------------------------------------------


def _molecule_case(name: str, read: str, *, ref_start: int = 0, ref_end: int | None = None
                   ) -> ReadCase:
    locus = LOCI[name]
    return ReadCase(
        f"{name}_molecule", "", locus, upstream_flank(locus, 0), downstream_flank(locus, 0),
        read, ref_start, None, ref_end=len(locus.reference) if ref_end is None else ref_end,
    )


def _count_records(records, locus, orientation: str, side_name: str):
    chrom = chromosome(locus, orientation)
    reads, cigars, starts = polyg_alignments(
        [r.query_sequence for r in records], [r.cigartuples for r in records],
        [r.reference_start for r in records], ref_orientation=orientation,
        window_start=0, window_len=len(chrom))
    side = fc.list_candidates(_polyg_chromosome(locus, orientation), _edges(locus)[side_name],
                              upstream=side_name == "upstream")
    bases, placed = fc.read_candidate_bases(reads, cigars, starts, side)
    return side, bases, placed


def _records(case: ReadCase, orientation: str, forms=None):
    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6"},
        "SQ": [{"SN": "synth", "LN": len(chromosome(case.locus, orientation))}],
    })
    out = []
    for i, p in enumerate(placements(case)):
        if forms is None or p.form in forms:
            out.append((p, to_record(header, case, p, orientation, f"r{i}")))
    return out


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_count_reads_the_read_not_the_pileup(orientation):
    # PolyG454: reference CC G10, the molecule carries C G6. A one-gap alignment
    # deletes CGGGG; the left-aligned gap starts at the edge C, so the pileup
    # shows that C as deleted. Read by position from the outer marker, the edge
    # base shows a G and the outer C is unchanged.
    locus = LOCI["PolyG454"]
    read = locus.upstream[:-1] + "G" * 6 + locus.downstream
    case = _molecule_case("PolyG454", read)
    found = _records(case, orientation, forms={"gap_shift"})
    assert found
    left = min(found, key=lambda pr: pr[0].cigar[0][1])
    edge_ref = (CHROM_PAD + len(locus.upstream) - 1 if orientation == "+"
                else len(chromosome(locus, orientation)) - 1
                - (CHROM_PAD + len(locus.upstream) - 1))
    aligned = dict((ref, q) for q, ref in left[1].get_aligned_pairs() if ref is not None)
    assert aligned[edge_ref] is None  # the pileup has no base at the edge C

    side, bases, placed = _count_records([r for _, r in found], locus, orientation, "upstream")
    assert placed.all()
    by_d = dict(zip(side.d, bases.T, strict=True))
    assert (by_d[0] == ord("G")).all()
    assert (by_d[1] == ord("C")).all()
    n, x = fc.count_candidates(bases)
    assert (n[0], x[0], n[1], x[1]) == (len(found), len(found), len(found), 0)


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_count_drops_a_read_that_does_not_reach_the_outer_marker(orientation):
    locus = LOCI["PolyG5177"]
    full = _molecule_case("PolyG5177", locus.reference)
    # Starts 3 bases into the upstream marker: 7 of its 10 bases are in the read.
    short = _molecule_case("PolyG5177", locus.reference[3:], ref_start=3)
    records = [r for _, r in _records(full, orientation)] + \
        [r for _, r in _records(short, orientation)]
    side, bases, placed = _count_records(records, locus, orientation, "upstream")
    assert placed.tolist() == [True, False]
    n, x = fc.count_candidates(bases)
    assert (n == 1).all() and (x == 0).all()
    # The downstream marker is in both reads.
    _, _, placed_dn = _count_records(records, locus, orientation, "downstream")
    assert placed_dn.tolist() == [True, True]


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_count_covers_only_candidates_inside_the_read(orientation):
    # The read places the downstream marker but starts 2 bases before the
    # downstream edge base, so only the downstream candidates count.
    locus = LOCI["PolyG5177"]
    start = len(locus.upstream) + len(locus.tract) - 2
    case = _molecule_case("PolyG5177", locus.reference[start:], ref_start=start)
    records = [r for _, r in _records(case, orientation)]
    side, bases, placed = _count_records(records, locus, orientation, "downstream")
    assert placed.all()
    n, x = fc.count_candidates(bases)
    assert (n == 1).all()
    _, _, placed_up = _count_records(records, locus, orientation, "upstream")
    assert not placed_up.any()


# ---------------------------------------------------------------------------
# End to end: BAM -> count pass -> flanks -> side TSV -> FlankRecord
# ---------------------------------------------------------------------------


def _write_bam(tmp_path: Path, name: str, molecules: list[tuple[str, int]]) -> tuple[Path, Path]:
    """A BAM with one synthetic chromosome per orientation, and its BED.

    `molecules` lists ``(read, copies)`` full-length reads in polyG
    orientation. One gapless read of the whole chromosome lets the
    extraction rebuild the reference window from its MD tag. One low-MAPQ
    read carries the reference molecule and must be filtered out.
    """
    locus = LOCI[name]
    chroms = {o: f"{name}_{'plus' if o == '+' else 'minus'}" for o in ORIENTATIONS}
    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": chroms[o], "LN": len(chromosome(locus, o))} for o in ORIENTATIONS],
    })
    unsorted = tmp_path / "unsorted.bam"
    bam_path = tmp_path / "normal.bam"
    with pysam.AlignmentFile(str(unsorted), "wb", header=header) as out:
        for ref_id, orientation in enumerate(ORIENTATIONS):
            whole = chromosome(locus, orientation)
            seg = pysam.AlignedSegment(header)
            seg.query_name = "reference"
            seg.query_sequence = whole
            seg.query_qualities = pysam.qualitystring_to_array("I" * len(whole))
            seg.reference_id = ref_id
            seg.reference_start = 0
            seg.mapping_quality = 60
            seg.cigartuples = [(0, len(whole))]
            seg.set_tags([("MD", str(len(whole))), ("NM", 0), ("MI", "reference/A", "Z")])
            out.write(seg)
            items = [(m, c, 60) for m, c in molecules] + [(locus.reference, 1, 0)]
            for j, (molecule, copies, mapq) in enumerate(items):
                case = _molecule_case(name, molecule)
                p = placements(case)[0]
                for c in range(copies):
                    seg = to_record(header, case, p, orientation, f"m{j}_{c}")
                    seg.reference_id = ref_id
                    seg.mapping_quality = mapq
                    out.write(seg)
    pysam.sort("-o", str(bam_path), str(unsorted))
    pysam.index(str(bam_path))
    bed = tmp_path / "panel.bed"
    tract_start = CHROM_PAD + len(locus.upstream)
    bed.write_text("".join(
        f"{chroms[o]}\t{tract_start}\t{tract_start + len(locus.tract)}\t{name}{o}\n"
        for o in ORIENTATIONS))
    return bam_path, bed


def test_build_flanks_end_to_end(tmp_path):
    # PolyG5177 homozygous edge variant: CTCACT G13 TAGAG in 40 reads, the
    # reference molecule in 2 reads.
    locus = LOCI["PolyG5177"]
    variant = locus.upstream[:-4] + "G" * 13 + locus.downstream[1:]
    bam_path, bed = _write_bam(tmp_path, "PolyG5177", [(variant, 40), (locus.reference, 2)])

    result = fc.build_flanks(bam_path, bed, n_workers=2, sample="normal")

    assert set(result.records) == {"PolyG5177+", "PolyG5177-"}
    sides = result.sides.set_index(["locus", "side"])
    for o in ORIENTATIONS:
        up_row = sides.loc[(f"PolyG5177{o}", "upstream")]
        dn_row = sides.loc[(f"PolyG5177{o}", "downstream")]
        assert (up_row.flank_d, dn_row.flank_d) == (4, 1)
        assert up_row.no_read_error_candidate == 0
        # The reference read and the 42 molecules; the low-MAPQ read is filtered.
        assert up_row.n_reads == 43
        up, dn = result.records[f"PolyG5177{o}"]
        assert up.bases == upstream_flank(locus, 4).bases
        assert dn.bases == downstream_flank(locus, 1).bases

        # The flanks place the variant reads with their true tuple.
        records = [r for r in pysam.AlignmentFile(str(bam_path)).fetch(
            f"PolyG5177_{'plus' if o == '+' else 'minus'}") if r.query_name.startswith("m0_")]
        chrom = chromosome(locus, o)
        out = place_flanks(
            [r.query_sequence for r in records], [r.cigartuples for r in records],
            np.array([r.reference_start for r in records]),
            upstream=up, downstream=dn, ref_orientation=o, reference_window=chrom,
            reference_window_start=0)
        assert {run_tuple(t) for t in out.tracts} == {(13,)}

    cand = result.candidates
    edge = cand[(cand.locus == "PolyG5177+") & (cand.side == "upstream") & (cand.d == 0)].iloc[0]
    assert (edge.n, edge.x, edge.call) == (43, 40, "g_in_donor")
    # Both orientations give the same counts.
    plus = cand[cand.locus == "PolyG5177+"].drop(columns="locus").reset_index(drop=True)
    minus = cand[cand.locus == "PolyG5177-"].drop(columns="locus").reset_index(drop=True)
    assert plus.equals(minus)

    paths = fc.write_flank_tables(result, tmp_path / "normal")
    assert all(p.exists() for p in paths)
    table = read_flank_table(paths[1])
    assert table.records == result.records
    assert table.sample == "normal"
    assert set(pd.read_csv(paths[0], sep="\t")["sample"]) == {"normal"}
    assert json.loads(paths[2].read_text())["sample"] == "normal"


def test_side_table_round_trips_masked_offsets(tmp_path):
    locus = LOCI["PolyG4732"]
    window = _polyg_chromosome(locus)
    records = []
    rows = []
    for side_name, g_d in (("upstream", {0, 4}), ("downstream", set())):
        side = fc.list_candidates(window, _edges(locus)[side_name],
                                  upstream=side_name == "upstream")
        calls = np.array([d in g_d for d in side.d])
        record, d, flag = fc.choose_flank(side, calls, window, ref_orientation="-",
                                          window_start=7)
        records.append(record)
        rows.append({"locus": "L", "side": side_name, "ref_orientation": "-", "flank_d": d,
                     "flank_bases": record.bases,
                     "flank_ref_positions": ",".join(map(str, record.ref_positions)),
                     "masked_offsets": ",".join(map(str, record.masked_offsets)),
                     "no_read_error_candidate": int(flag), "n_candidates": len(side.d),
                     "n_reads": 0, "n_marker_placed": 0})
    assert records[0].masked_offsets == (7,)
    result = fc.FlankConstruction("S1", {"L": tuple(records)}, pd.DataFrame(),
                                  pd.DataFrame(rows),
                                  fc.ErrorRate(0.01, 1, 100, np.array([0.01])))
    paths = fc.write_flank_tables(result, tmp_path / "x")
    assert read_flank_table(paths[1]).records == {"L": tuple(records)}


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="Linux only")
def test_worker_count_leaves_free_cores():
    n = fc.worker_count()
    assert 1 <= n < max(2, len(os.sched_getaffinity(0)))


# ---------------------------------------------------------------------------
# The sample name
# ---------------------------------------------------------------------------


def _bam_with_read_groups(tmp_path: Path, groups: list[dict]) -> Path:
    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": "synth", "LN": 100}]}
    if groups:
        header["RG"] = groups
    path = tmp_path / "rg.bam"
    with pysam.AlignmentFile(str(path), "wb", header=header):
        pass
    return path


def test_sample_name_comes_from_the_read_group(tmp_path):
    bam = _bam_with_read_groups(tmp_path, [{"ID": "A", "SM": "N1"}, {"ID": "B", "SM": "N1"}])
    assert fc.sample_name(bam) == "N1"
    assert fc.sample_name(bam, "N1") == "N1"
    with pytest.raises(ValueError, match="differs"):
        fc.sample_name(bam, "other")


def test_sample_name_without_a_read_group_needs_the_argument(tmp_path):
    bam = _bam_with_read_groups(tmp_path, [{"ID": "A"}])
    assert fc.sample_name(bam, "N1") == "N1"
    with pytest.raises(ValueError, match="no read group names a sample"):
        fc.sample_name(bam)


def test_sample_name_rejects_two_samples(tmp_path):
    bam = _bam_with_read_groups(tmp_path, [{"ID": "A", "SM": "N1"}, {"ID": "B", "SM": "N2"}])
    with pytest.raises(ValueError, match="more than one sample"):
        fc.sample_name(bam)


def test_flank_table_without_a_sample_is_rejected(tmp_path):
    path = tmp_path / "old_sides.tsv"
    path.write_text("locus\tside\tflank_bases\tflank_ref_positions\tmasked_offsets\n")
    with pytest.raises(ValueError, match="no sample column"):
        read_flank_table(path)
