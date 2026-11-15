"""Tests for flank construction: the pair of flanks that maximizes the objective.

The solver is checked three ways. Hand-built read sets with a known optimum
(one or two donor haplotypes) must give that pair and its tracts. On small
random read sets, a brute-force search with the plain reference of
`tests/_slow_flank_eval.py` over every candidate pair must give the same
optimum and the same tied pairs. The candidate index must list exactly the
matches that the library's match rule gives. One end-to-end test writes a
small BAM, runs `build_flanks` and reads the side TSV back.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import pytest

import prolyg_phasing.io._flank_construction as fc
from prolyg_phasing.io._flank_placement import (
    FLANK_PLACE_VALUES,
    MAX_MATCH_LEN,
    NO_MATCH,
    FlankRecord,
    code_matrix,
    decode,
    edge_windows,
    marks_matrix,
    place_flanks,
    place_polyg_reads,
    read_flank_table,
    tract_edges,
    window_costs,
)
from prolyg_phasing.io.bam import extract_panel
from prolyg_phasing.io.panel import parse_run_lengths
from tests import _slow_flank_eval as slow
from tests._hand_built_reads import (
    CHROM_PAD,
    LOCI,
    ReadCase,
    _sub,
    bed_interval,
    chromosome,
    downstream_flank,
    placements,
    run_tuple,
    to_record,
    upstream_flank,
)

ORIENTATIONS = ("+", "-")


# ---------------------------------------------------------------------------
# The candidate index lists the matches of the match rule
# ---------------------------------------------------------------------------


def test_window_neighbors_are_the_matches_of_the_match_rule():
    """Every string the index lists for a window matches it at the listed cost,
    and a string it does not list does not match."""
    rng = random.Random(5)
    reads = ["".join(rng.choices("ACGTTGGN", k=60)) for _ in range(40)]
    matrix, _ = code_matrix(reads)
    rows, ends = tract_edges(matrix)
    windows = np.unique(edge_windows(matrix, rows, ends), axis=0)[:150]
    listed: dict[tuple[int, int], int] = {}
    for code, window, rest in fc.window_neighbors(windows, np.full(len(windows), 1 << 20)):
        for x, w, r in zip(code.tolist(), window.tolist(), rest.tolist(), strict=True):
            cost = (1 << 20) - r
            listed[(x, w)] = min(cost, listed.get((x, w), cost))
    by_window: dict[int, dict[int, int]] = {}
    for (x, w), cost in listed.items():
        by_window.setdefault(w, {})[x] = cost
    for w, window in enumerate(windows):
        found = by_window.get(w, {})
        xs = list(found)
        # A sample of strings near the window that the index does not list.
        tail = window[MAX_MATCH_LEN - 10:]
        for _ in range(30):
            near = np.where(tail > 3, rng.randrange(4), tail).copy()
            for _ in range(rng.choice([1, 2])):
                near[rng.randrange(10)] = rng.randrange(4)
            xs.append(int(near.astype(np.int64) @ FLANK_PLACE_VALUES))
        for x in xs:
            flank = np.array([(x >> (2 * (9 - j))) & 3 for j in range(10)], dtype=np.uint8)
            cost = window_costs(window[None, :], flank)[0][0]
            assert cost == found.get(x, NO_MATCH), (decode(x), window)


# ---------------------------------------------------------------------------
# Hand-built read sets with a known optimum
# ---------------------------------------------------------------------------

_RNG = random.Random(7)


def _background(n: int) -> str:
    """A, C, T and isolated G: no G run outside the tract."""
    out: list[str] = []
    for _ in range(n):
        c = _RNG.choice("ACTACTACG")
        if c == "G" and out and out[-1] == "G":
            c = "A"
        out.append(c)
    return "".join(out)


LEFT, RIGHT = _background(60), _background(60)
UP, DN = "ACTTCAGCAT", "CAAATTTTAT"


def _reads_from(hap: str, n: int, rl: int = 100, jitter: int = 15) -> list[str]:
    out = []
    mid = len(hap) // 2
    for _ in range(n):
        s = max(0, min(len(hap) - rl, mid - rl // 2 + _RNG.randint(-jitter, jitter)))
        out.append(hap[s:s + rl])
    return out


def _counted(reads: list[str]) -> list[tuple[str, frozenset[int], int]]:
    """Distinct reads with every base marked, and their counts, for the slow reference."""
    return [(s, slow.every_base(s), reads.count(s)) for s in sorted(set(reads))]


def _check(reads: list[str], u: str, v: str, tracts: set[str]) -> fc.FlankPair:
    """The expected pair is optimal, F equals the slow reference, and every
    optimal pair gives the expected tracts."""
    pair = fc.solve_flank_pair(reads, slow.marks(reads))
    counted = _counted(reads)
    assert (u, v) in pair.optimal_pairs
    assert pair.objective_half_units == slow.objective(counted, u, v)[0]
    for pu, pv in pair.optimal_pairs:
        got = place_polyg_reads(reads, pu, pv, slow.marks(reads))
        assert set(got.tracts[got.kept]) == tracts, (pu, pv)
    return pair


def test_a_ggg_t_piece_next_to_the_run_joins_the_tract():
    hap = LEFT + UP + "GGGTGGGGGG" + DN + RIGHT
    _check(_reads_from(hap, 60), UP, DN, {"GGGTGGGGGG"})


def test_a_copy_of_the_flank_past_the_tract_is_ignored():
    hap = LEFT + UP + "G" * 10 + DN + RIGHT[:15] + UP + RIGHT[15:]
    reads = _reads_from(hap, 50, rl=130, jitter=3)
    pair = _check(reads, UP, DN, {"G" * 10})
    assert pair.n_reads_placed == len(reads)


def test_of_two_copies_of_the_flank_the_nearer_one_places():
    hap = LEFT[:30] + UP + LEFT[30:45] + UP + "G" * 10 + DN + RIGHT
    reads = _reads_from(hap, 50, rl=130, jitter=3)
    assert any(r.count(UP) == 2 for r in reads)
    pair = _check(reads, UP, DN, {"G" * 10})
    assert pair.n_reads_placed == len(reads)


def test_the_g_run_beats_an_empty_tract():
    hap = LEFT + UP + "G" * 12 + DN + RIGHT
    reads = _reads_from(hap, 50)
    pair = _check(reads, UP, DN, {"G" * 12})
    counted = _counted(reads)
    far_u, far_v = LEFT[30:40], LEFT[40:50]
    if far_u[-1] != "G" and far_v[0] != "G":
        assert slow.objective(counted, far_u, far_v)[0] < pair.objective_half_units


def test_one_read_error_in_the_flank_still_places():
    hap = LEFT + UP + "G" * 10 + DN + RIGHT
    i0 = len(LEFT) + 4
    variants = [hap[:i0] + "G" + hap[i0 + 1:], hap[:i0] + "A" + hap[i0:],
                hap[:i0] + hap[i0 + 1:], hap[:i0] + hap[i0 + 2:]]
    errors = [r for h in variants for r in _reads_from(h, 3, jitter=3)]
    reads = _reads_from(hap, 30, jitter=3) + errors
    pair = _check(reads, UP, DN, {"G" * 10})
    assert pair.n_reads_placed == len(reads)


def test_a_run_ending_in_a_non_g_error_is_not_placed_at_the_old_edge():
    """Half the reads end the run in a T. A gap and a substitution never go
    together, so those reads have no placement under (UP, DN)."""
    clean = _reads_from(LEFT + UP + "G" * 10 + DN + RIGHT, 30, jitter=3)
    error = _reads_from(LEFT + UP + "G" * 9 + "T" + DN + RIGHT, 30, jitter=3)
    reads = clean + error
    counted = _counted(reads)
    assert slow.objective(counted, UP, DN) == (30 * 20, 30)
    pair = fc.solve_flank_pair(reads, slow.marks(reads))
    for pu, pv in pair.optimal_pairs:
        assert slow.objective(counted, pu, pv)[0] == pair.objective_half_units


def test_reads_of_300_bases():
    """Edges past base 255 and runs past 255 bases from the read start place."""
    hap = _background(400) + UP + "G" * 12 + DN + _background(200)
    reads = [hap[s:s + 300] for s in range(133, 153)]
    assert {len(r) for r in reads} == {300}
    assert all(r.index(UP) + 9 > 255 for r in reads)
    pair = _check(reads, UP, DN, {"G" * 12})
    assert pair.objective_half_units == 20 * 24


def test_a_homopolymer_length_change_in_the_flank_is_a_gap():
    """Allele A5 (60 reads) and A3 (40 reads) of the upstream flank: the A5
    string matches the A3 reads with a 2-base gap, cost 5 half units."""
    u5 = "TCAAAAACAT"
    r5 = _reads_from(LEFT + u5 + "G" * 10 + DN + RIGHT, 60, jitter=3)
    r3 = _reads_from(LEFT + "TCAAACAT" + "G" * 10 + DN + RIGHT, 40, jitter=3)
    pair = _check(r5 + r3, u5, DN, {"G" * 10})
    assert pair.objective_half_units == 60 * 20 + 40 * (20 - 5)
    assert pair.n_reads_placed == 100


def test_reads_with_a_negative_value_add_zero():
    """40 decoy reads hold near-copies of both flanks around G + 20 bases + G:
    value -80 half units. They do not move the pair."""
    hap = LEFT + UP + "G" * 10 + DN + RIGHT
    decoy = (_background(30) + "ACTTCAGCTT" + "G" + "ACTTACATCATTACCATACA" + "G"
             + "CAAATTTAAT" + _background(40))
    reads = _reads_from(hap, 60, jitter=3) + [decoy] * 40
    assert slow.best_placement(decoy, UP, DN, slow.every_base(decoy))[0] == -80
    pair = _check(reads, UP, DN, {"G" * 10})
    assert pair.objective_half_units == 1200
    assert pair.n_reads_placed == 100


def test_a_non_g_base_and_three_g_next_to_the_run_join_the_tract():
    """No candidate flank stops in front of a non-G base and three G."""
    for hap, tract in ((LEFT + UP + "G" * 10 + "CGGG" + DN + RIGHT, "G" * 10 + "CGGG"),
                       (LEFT + UP + "GGGA" + "G" * 10 + DN + RIGHT, "GGGA" + "G" * 10)):
        reads = _reads_from(hap, 60)
        pair = _check(reads, UP, DN, {tract})
        distinct = sorted(set(reads))
        search = fc._LocusSearch(distinct, marks_matrix(slow.marks(distinct), 100),
                                 [1] * len(distinct))
        us = [decode(int(c)) for c in search.u.candidates]
        vs = [decode(int(c))[::-1] for c in search.v.candidates]
        assert not any(u[-4:-1] == "GGG" for u in us)
        assert not any(v[1:4] == "GGG" for v in vs)
        assert pair.n_reads_placed == len(reads)


# ---------------------------------------------------------------------------
# The search against a brute-force search with the slow reference
# ---------------------------------------------------------------------------


def _random_read_set(rng: random.Random) -> list[str]:
    """Reads of one or two haplotypes around a G run, with read errors."""
    haps = []
    for _ in range(rng.choice([1, 2])):
        run = "".join(rng.choices("GGGGGGT", k=rng.randint(4, 9)))
        haps.append(_background(12) + UP + run + rng.choice([DN, "CAAATTAT", "TAAATTTTAT"])
                    + _background(12))
    reads = []
    for _ in range(rng.randint(3, 6)):
        s = list(rng.choice(haps))
        for _ in range(rng.randint(0, 2)):
            i = rng.randrange(len(s))
            op = rng.random()
            if op < 0.5:
                s[i] = rng.choice("ACGT")
            elif op < 0.75:
                s.insert(i, rng.choice("ACGT"))
            else:
                del s[i]
        start = rng.randint(0, 6)
        reads.append("".join(s)[start:start + rng.randint(36, 50)])
    return reads


def test_search_matches_a_brute_force_search_on_random_read_sets():
    rng = random.Random(13)
    n_checked = 0
    for _ in range(25):
        reads = _random_read_set(rng)
        counted = _counted(reads)
        us, vs = slow.candidates(counted)
        values = {(u, v): slow.objective(counted, u, v)[0] for u in us for v in vs}
        best = max(values.values(), default=0)
        pair = fc.solve_flank_pair(reads, slow.marks(reads))
        if best <= 0:
            assert pair is None
            continue
        n_checked += 1
        assert pair.objective_half_units == best
        assert set(pair.optimal_pairs) == {p for p, f in values.items() if f == best}
        assert pair.n_reads_placed == slow.objective(counted, pair.upstream,
                                                     pair.downstream)[1]
    assert n_checked >= 20


def _two_run_read_set(rng: random.Random) -> tuple[list[str], list[frozenset[int]]]:
    """Reads that hold two G runs, each between its own flanks, and their marks.

    The second run lies in the reads' later half. A read's marks are an
    interval around its first run, around its second run, a short random
    interval, or nothing, so the same string can carry different marks.
    """
    run1 = "G" * rng.randint(5, 9)
    run2 = "G" * rng.randint(5, 9)
    hap = (_background(8) + UP + run1 + DN + _background(6) + "TTACCAGTCA" + run2
           + "CCATACAATC" + _background(8))
    start2 = hap.index("TTACCAGTCA") + 10
    reads, marks = [], []
    for _ in range(rng.randint(4, 8)):
        s = list(hap)
        for _ in range(rng.randint(0, 1)):
            s[rng.randrange(len(s))] = rng.choice("ACGT")
        read = "".join(s)
        choice = rng.random()
        if choice < 0.4:
            m = range(len(_background(8)) + 10, len(_background(8)) + 10 + len(run1))
        elif choice < 0.7:
            m = range(start2, start2 + len(run2))
        elif choice < 0.9:
            lo = rng.randrange(len(read) - 3)
            m = range(lo, lo + 3)
        else:
            m = range(0)
        reads.append(read)
        marks.append(frozenset(m))
    return reads, marks


def test_search_matches_a_brute_force_search_with_marks():
    """The marks choose which G run, if any, a read's placement may surround."""
    rng = random.Random(29)
    n_checked = 0
    for _ in range(25):
        reads, marks = _two_run_read_set(rng)
        keys = sorted(set(zip(reads, marks, strict=True)), key=lambda k: (k[0], sorted(k[1])))
        counted = [(r, m, sum(1 for x in zip(reads, marks, strict=True) if x == (r, m)))
                   for r, m in keys]
        us, vs = slow.candidates(counted)
        values = {(u, v): slow.objective(counted, u, v)[0] for u in us for v in vs}
        best = max(values.values(), default=0)
        pair = fc.solve_flank_pair(reads, slow.marks(reads, marks))
        if best <= 0:
            assert pair is None
            continue
        n_checked += 1
        assert pair.objective_half_units == best
        assert set(pair.optimal_pairs) == {p for p, f in values.items() if f == best}
        assert pair.n_reads_placed == slow.objective(counted, pair.upstream,
                                                     pair.downstream)[1]
    assert n_checked >= 20


def test_no_pair_reaches_a_positive_objective():
    reads = ["ACTACTACTACTACTACTACT"] * 5
    assert fc.solve_flank_pair(reads, slow.marks(reads)) is None
    assert fc.solve_flank_pair([], slow.marks([])) is None


# ---------------------------------------------------------------------------
# The side table
# ---------------------------------------------------------------------------


def _locus_flanks(records, name="L") -> fc.LocusFlanks:
    pair = fc.FlankPair("ACTTCAGCAT", "CAAATTTTAT", 41, (("ACTTCAGCAT", "CAAATTTTAT"),),
                        30, 25, placement=None)
    positions = tuple(fc.FlankPosition(r, 20, 18) if r.has_position
                      else fc.FlankPosition(r, 0, 0) for r in records)
    return fc.LocusFlanks(name, "+", pair, positions, 30)


def test_side_table_round_trips_flanks_of_any_reference_length(tmp_path):
    up = FlankRecord("ACTTCAGCAT", tuple(range(100, 109)), 8)
    dn = FlankRecord("CAAATTTTAT", tuple(range(130, 141)), 0)
    result = fc.construct_flanks([("L", _locus_flanks((up, dn))), ("NoWindow", None)],
                                 sample="S1")
    path = fc.write_flank_table(result, tmp_path / "x")
    table = read_flank_table(path)
    assert table.records == {"L": (up, dn)}
    assert table.no_reference_window == {"NoWindow"}
    sides = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    assert list(sides.columns) == ["sample", *fc.SIDE_COLUMNS]
    row = sides.iloc[0]
    assert (row.objective, row.n_reads, row.n_reads_placed, row.n_tied_pairs) == \
        ("20.5", "30", "25", "1")
    assert (row.n_reads_positioned, row.n_reads_at_position) == ("20", "18")


def test_a_side_with_no_reference_position_round_trips(tmp_path):
    up = FlankRecord("ACTTCAGCAT", tuple(range(100, 110)), 9)
    dn = FlankRecord("CAAATTTTAT", (), 0)
    result = fc.construct_flanks([("L", _locus_flanks((up, dn)))], sample="S1")
    path = fc.write_flank_table(result, tmp_path / "x")
    sides = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False).set_index("side")
    assert sides.loc["upstream", "reference_window_status"] == "recovered"
    row = sides.loc["downstream"]
    assert (row.reference_window_status, row.flank_bases, row.flank_ref_positions) == \
        ("no_reference_position", "CAAATTTTAT", "")
    assert (row.objective, row.n_reads_placed, row.n_reads_positioned,
            row.n_reads_at_position) == ("20.5", "25", "0", "0")
    assert read_flank_table(path).records == {"L": (up, dn)}
    # An upstream side with no position reads back as an upstream record.
    empty_up = FlankRecord("ACTTCAGCAT", (), -1)
    result = fc.construct_flanks([("L", _locus_flanks((empty_up, dn)))], sample="S1")
    back = read_flank_table(fc.write_flank_table(result, tmp_path / "y")).records["L"]
    assert back == (empty_up, dn) and back[0].is_upstream and not back[1].is_upstream


def test_a_flank_table_status_must_match_its_positions(tmp_path):
    rows = [_side_row("upstream", "no_reference_position", "ACTTCAGCAT", "100,101"),
            _side_row("downstream", "recovered", "CAAATTTTAT", "130,131")]
    with pytest.raises(ValueError, match="has the status 'no_reference_position' but "
                                         "reference positions"):
        read_flank_table(_side_table(tmp_path, rows))
    rows[0] = _side_row("upstream", "recovered", "ACTTCAGCAT", "")
    with pytest.raises(ValueError, match="has the status 'recovered' but no reference"):
        read_flank_table(_side_table(tmp_path, rows))


def test_a_locus_with_no_positive_pair_gets_its_own_status(tmp_path):
    result = fc.construct_flanks([("L", fc.LocusFlanks("L", "+", None, None, 7))],
                                 sample="S1")
    assert result.records == {}
    path = fc.write_flank_table(result, tmp_path / "x")
    sides = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    assert set(sides.reference_window_status) == {"no_flank_pair"}
    assert set(sides.flank_bases) == set(sides.flank_ref_positions) == {""}
    assert set(zip(sides.objective, sides.n_reads, sides.n_reads_placed, strict=True)) == \
        {("0.0", "7", "0")}
    table = read_flank_table(path)
    assert table.no_flank_pair == {"L"} and table.records == {}
    assert table.loci == {"L"}


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="Linux only")
def test_worker_count_leaves_free_cores():
    n = fc.worker_count()
    assert 1 <= n < max(2, len(os.sched_getaffinity(0)))


# ---------------------------------------------------------------------------
# End to end: BAM -> flank pair -> side TSV -> FlankRecord
# ---------------------------------------------------------------------------


def _molecule_case(name: str, read: str) -> ReadCase:
    locus = LOCI[name]
    return ReadCase(f"{name}_molecule", "", locus, upstream_flank(locus, 0),
                    downstream_flank(locus, 0), read, 0, None,
                    ref_end=len(locus.reference))


def _write_bam(tmp_path: Path, name: str, molecules: list[tuple[str, int]]) -> tuple[Path, Path]:
    """A BAM with one synthetic chromosome per orientation, and its BED.

    `molecules` lists ``(read, copies)`` full-length reads in polyG
    orientation, each written with its first alignment in `placements`, or
    ``(read, copies, form)`` with alignment `form` of that list. One
    gapless read of the whole chromosome lets the
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
            items = [(m[0], m[1], 60, m[2] if len(m) > 2 else 0) for m in molecules]
            items.append((locus.reference, 1, 0, 0))
            for j, (molecule, copies, mapq, form) in enumerate(items):
                case = _molecule_case(name, molecule)
                p = placements(case)[form]
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
        row = sides.loc[(f"PolyG5177{o}", "upstream")]
        # The reference read and the 42 molecules; the low-MAPQ read is
        # filtered. The 40 variant reads place; the reference reads have no
        # G after the downstream flank's run-side T.
        assert (row.n_reads, row.n_reads_placed, row.n_tied_pairs) == (43, 40, 1)
        assert row.objective == 40 * 13
        up, dn = result.records[f"PolyG5177{o}"]
        want_up, want_dn = upstream_flank(locus, 4), downstream_flank(locus, 1)
        assert (up.bases, dn.bases) == (want_up.bases, want_dn.bases)
        n = len(locus.reference)
        for got, want in ((up, want_up), (dn, want_dn)):
            if o == "+":
                positions = tuple(CHROM_PAD + want.start + i for i in range(10))
            else:
                positions = tuple(CHROM_PAD + n - 1 - (want.start + i) for i in range(10))
            assert got.ref_positions == positions

        # The flanks place the variant reads with their true tuple.
        records = [r for r in pysam.AlignmentFile(str(bam_path)).fetch(
            f"PolyG5177_{'plus' if o == '+' else 'minus'}") if r.query_name.startswith("m0_")]
        bed_start, bed_end = bed_interval(locus, o)
        out = place_flanks(records, upstream=up, downstream=dn, ref_orientation=o,
                           bed_start=bed_start, bed_end=bed_end)
        assert {run_tuple(t) for t in out.tracts} == {(13,)}

    path = fc.write_flank_table(result, tmp_path / "normal")
    table = read_flank_table(path)
    assert table.records == result.records
    assert table.sample == "normal"


# ---------------------------------------------------------------------------
# Reference positions from the alignments
# ---------------------------------------------------------------------------


def _coordinate(name: str, orientation: str, index: int) -> int:
    """Reference coordinate of polyG index `index` of a locus on its synthetic contig."""
    n = len(LOCI[name].reference)
    return CHROM_PAD + index if orientation == "+" else CHROM_PAD + n - 1 - index


def _positions(name: str, orientation: str, first: int, last: int) -> tuple[int, ...]:
    """Coordinates of polyG indices `first` to `last`, in polyG order."""
    return tuple(_coordinate(name, orientation, i) for i in range(first, last + 1))


def _built(tmp_path: Path, name: str, molecules) -> dict:
    bam_path, bed = _write_bam(tmp_path, name, molecules)
    return fc.build_flanks(bam_path, bed, n_workers=1, sample="normal").records


def test_a_flank_equal_to_the_reference(tmp_path):
    locus = LOCI["PolyG151"]
    records = _built(tmp_path, "PolyG151", [(locus.reference, 20)])
    up_spec, dn_spec = upstream_flank(locus, 0), downstream_flank(locus, 0)
    for o in ORIENTATIONS:
        up, dn = records[f"PolyG151{o}"]
        assert (up.bases, dn.bases) == (up_spec.bases, dn_spec.bases)
        assert up.ref_positions == _positions("PolyG151", o, up_spec.start, up_spec.run_side)
        assert dn.ref_positions == _positions("PolyG151", o, dn_spec.start, dn_spec.start + 9)


def test_a_donor_flank_with_three_substitutions(tmp_path):
    """The donor flank is three edits from the reference: no reference search
    matches it, and the alignments place it on the reference flank's bases.
    Extraction with that flank gives every read its tract."""
    locus = LOCI["PolyG151"]
    # Upstream flank ATATTTAGAT at polyG indices 20 to 29; ACATATACAT in the donor.
    donor = _sub(_sub(_sub(locus.reference, 21, "C"), 24, "A"), 27, "C")
    bam_path, bed = _write_bam(tmp_path, "PolyG151", [(donor, 20)])
    result = fc.build_flanks(bam_path, bed, n_workers=1, sample="normal")
    for o in ORIENTATIONS:
        up, _ = result.records[f"PolyG151{o}"]
        assert up.bases == "ACATATACAT"
        assert up.ref_positions == _positions("PolyG151", o, 20, 29)
    flanks = fc.write_flank_table(result, tmp_path / "normal")
    panel = extract_panel(bam_path, bed, flanks)
    for o in ORIENTATIONS:
        locus_out = panel.loci[f"PolyG151{o}"]
        assert locus_out.n_alignments_placed == 20
        assert {parse_run_lengths(t) for t in locus_out.seq} == {(12,)}


def test_a_donor_insertion_inside_the_flank(tmp_path):
    """A C between reference indices 22 and 23: the donor flank TACTTTAGAT
    covers 9 reference bases, from index 21 to the edge base at 29."""
    locus = LOCI["PolyG151"]
    donor = locus.reference[:23] + "C" + locus.reference[23:]
    records = _built(tmp_path, "PolyG151", [(donor, 20)])
    for o in ORIENTATIONS:
        up, _ = records[f"PolyG151{o}"]
        assert up.bases == "TACTTTAGAT"
        assert up.ref_positions == _positions("PolyG151", o, 21, 29)


def _deletion_form(name: str, molecule: str, deleted: int) -> int:
    """Index in `placements` of the alignment that deletes reference index `deleted`."""
    every = placements(_molecule_case(name, molecule))
    return next(i for i, p in enumerate(every)
                if p.form == "gap_shift" and p.cigar[0][1] == deleted)


@pytest.mark.parametrize(("n_left", "n_right", "outer"), [(11, 10, 9), (12, 10, 10)])
def test_the_most_common_coordinate_wins_and_a_tie_goes_nearer_the_run(
        tmp_path, n_left, n_right, outer):
    """PolyG454 downstream AAACCACTTAAG: the donor lacks one A of the AA at
    downstream indices 9 and 10. The aligner deletes either A, so the outer
    flank base maps to index 10 (left A deleted) or 9 (right A deleted). The
    whole-contig read also places, and maps it to 9. So 11 left-deletion reads
    tie with it and 10 right-deletion reads; 12 outvote them."""
    locus = LOCI["PolyG454"]
    start = len(locus.upstream) + len(locus.tract)
    donor = locus.reference[:start + 10] + locus.reference[start + 11:]
    left = _deletion_form("PolyG454", donor, start + 9)
    right = _deletion_form("PolyG454", donor, start + 10)
    records = _built(tmp_path, "PolyG454", [(donor, n_left, left), (donor, n_right, right)])
    for o in ORIENTATIONS:
        _, dn = records[f"PolyG454{o}"]
        assert dn.bases == "AAACCACTTA"
        assert dn.ref_positions == _positions("PolyG454", o, start, start + outer)


def _locus_reads(reads: list[str], coords: list[list[int]], bed: tuple[int, int] = (0, 1 << 30)
                 ) -> fc.LocusReads:
    """Reads on a "+" locus whose alignments give polyG index i the coordinate
    ``coords[r][i]`` (-1: none), one aligned block per base. Every base is
    marked; `bed` is the BED interval."""
    blocks = [[(i, c, 1) for i, c in enumerate(row) if c >= 0] for row in coords]
    return fc.LocusReads("L", "+", reads, slow.marks(reads), blocks, *bed)


def _vote(loc: fc.LocusReads):
    return fc.flank_records_from_alignments(loc, fc.solve_flank_pair(loc.reads, loc.marks))


_READ = "TTACCAGTCA" + "ACTTCAGCAT" + "G" * 12 + "CAAATTTTAT" + "ACCAGTCATT"


def _coords_with(outer: int, run: int) -> list[int]:
    """Coordinates of `_READ` from 100, with its upstream flank's outer base
    (index 10) and run-side base (index 19) set."""
    coords = list(range(100, 100 + len(_READ)))
    coords[10], coords[19] = outer, run
    return coords


def test_a_flank_range_against_the_polyg_direction_is_an_invariant_error():
    """No real alignment maps two read bases out of order."""
    pair = fc.solve_flank_pair([_READ], slow.marks([_READ]))
    assert (pair.upstream, pair.downstream) == ("ACTTCAGCAT", "CAAATTTTAT")
    loc = _locus_reads([_READ], [_coords_with(120, 119)])
    with pytest.raises(RuntimeError, match="L: the upstream flank ACTTCAGCAT runs from"):
        fc.flank_records_from_alignments(loc, pair)


def test_a_flank_base_without_a_coordinate_leaves_no_position():
    pair = fc.solve_flank_pair([_READ], slow.marks([_READ]))
    coords = list(range(100, 100 + len(_READ)))
    coords[41] = -1  # the downstream outer base is soft-clipped
    up, dn = fc.flank_records_from_alignments(_locus_reads([_READ], [coords]), pair)
    assert up.record.ref_positions == tuple(range(110, 120))
    assert (up.n_reads_positioned, up.n_reads_at_position) == (1, 1)
    assert dn == fc.FlankPosition(FlankRecord("CAAATTTTAT", (), 0), 0, 0)


def test_the_pair_vote_differs_from_two_separate_votes():
    """Separate votes on each end give outer 112 (4 reads) and run side 121
    (4 reads), but no read gives that pair. Of the reads with both ends, 3
    give (110, 119) and 2 give (112, 121)."""
    rows = ([_coords_with(110, 119)] * 3 + [_coords_with(112, 121)] * 2
            + [_coords_with(112, -1)] * 2 + [_coords_with(-1, 121)] * 2)
    up, _ = _vote(_locus_reads([_READ] * len(rows), rows))
    assert up.record.ref_positions == tuple(range(110, 120))
    assert (up.n_reads_positioned, up.n_reads_at_position) == (5, 3)


def test_a_voted_coordinate_outside_the_reference_window_gives_no_position():
    """BED [120, 132): the window is [70, 182). The upstream outer base votes
    for 60, so the upstream side has no position; it keeps its vote counts."""
    up, dn = _vote(_locus_reads([_READ] * 3, [_coords_with(60, 119)] * 3, bed=(120, 132)))
    assert not up.record.has_position and up.record.bases == "ACTTCAGCAT"
    assert (up.n_reads_positioned, up.n_reads_at_position) == (3, 3)
    assert dn.record.ref_positions == tuple(range(132, 142))


def test_an_upstream_flank_past_the_downstream_flank_gives_neither_a_position():
    coords = list(range(100, 100 + len(_READ)))
    coords[10], coords[19] = 141, 150   # upstream: outer, run side
    coords[32], coords[41] = 140, 149   # downstream: run side, outer
    up, dn = _vote(_locus_reads([_READ], [coords]))
    assert not up.record.has_position and not dn.record.has_position
    assert (up.record.bases, dn.record.bases) == ("ACTTCAGCAT", "CAAATTTTAT")


def test_modal_pair_breaks_a_tie_by_the_run_side_then_the_outer_base():
    outer, run = np.array([5, 6, 4]), np.array([9, 9, 10])
    # The run side nearest the G run wins first: 10 toward larger coordinates.
    assert fc.modal_pair(outer, run, 1) == (4, 10, 1)
    # Toward smaller coordinates, run side 9 is nearer; then outer 5 over 6.
    assert fc.modal_pair(outer, run, -1) == (5, 9, 1)
    assert fc.modal_pair(np.array([5, 6, 6]), np.array([9, 9, 9]), 1) == (6, 9, 2)
    assert fc.modal_pair(np.zeros(0, np.int64), np.zeros(0, np.int64), 1) is None


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
    path.write_text("locus\tside\tflank_bases\tflank_ref_positions\n")
    with pytest.raises(ValueError, match="no sample column"):
        read_flank_table(path)


# ---------------------------------------------------------------------------
# A locus with no reference window
# ---------------------------------------------------------------------------


def _two_locus_bam(tmp_path: Path) -> tuple[Path, Path]:
    """PolyG5177 with a whole-contig read, and PolyG151 with no read over its window.

    Every read of PolyG151 starts and ends inside the contig padding, so no
    read covers the BED interval plus 50 bases on each side.
    """
    loci = {"window": LOCI["PolyG5177"], "no_window": LOCI["PolyG151"]}
    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": contig, "LN": len(chromosome(locus, "+"))}
               for contig, locus in loci.items()],
    })
    unsorted, bam_path = tmp_path / "unsorted.bam", tmp_path / "normal.bam"
    with pysam.AlignmentFile(str(unsorted), "wb", header=header) as out:
        for ref_id, (contig, locus) in enumerate(loci.items()):
            if contig == "window":
                whole = chromosome(locus, "+")
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
            case = _molecule_case(locus.name, locus.reference)
            p = placements(case)[0]
            for c in range(30):
                seg = to_record(header, case, p, "+", f"{contig}_{c}")
                seg.reference_id = ref_id
                out.write(seg)
    pysam.sort("-o", str(bam_path), str(unsorted))
    pysam.index(str(bam_path))
    bed = tmp_path / "panel.bed"
    tract_start = CHROM_PAD + 30
    bed.write_text("".join(
        f"{contig}\t{tract_start}\t{tract_start + len(locus.tract)}\t{locus.name}\n"
        for contig, locus in loci.items()))
    return bam_path, bed


def test_locus_without_a_reference_window_gets_a_status_row(tmp_path):
    from prolyg_phasing.io.bam import extract_panel
    from prolyg_phasing.io.panel import ExtractedPanel

    bam_path, bed = _two_locus_bam(tmp_path)
    result = fc.build_flanks(bam_path, bed, n_workers=1, sample="NORMAL")
    assert set(result.records) == {"PolyG5177"}
    path = fc.write_flank_table(result, tmp_path / "normal")

    sides = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    no_window = sides[sides.locus == "PolyG151"]
    assert list(no_window.side) == ["upstream", "downstream"]
    assert set(no_window.reference_window_status) == {"ref_window_unrecoverable"}
    for column in ("ref_orientation", "flank_bases", "flank_ref_positions", "objective",
                   "n_reads", "n_reads_placed", "n_tied_pairs", "n_reads_positioned",
                   "n_reads_at_position"):
        assert set(no_window[column]) == {""}, column
    window = sides[sides.locus == "PolyG5177"]
    assert set(window.reference_window_status) == {"recovered"}
    assert all(d.isdigit() for d in window.n_reads)  # integers, not floats with a NA

    table = read_flank_table(path)
    assert table.no_reference_window == {"PolyG151"}
    assert set(table.records) == {"PolyG5177"}
    assert table.records == result.records

    panel = extract_panel(bam_path, bed, path)
    locus = panel.loci["PolyG151"]
    assert locus.reference_window_status == "ref_window_unrecoverable"
    assert locus.n_reads == 0 and len(locus.seq) == 0
    assert locus.upstream_flank is None and locus.downstream_flank is None
    assert panel.loci["PolyG5177"].reference_window_status == "recovered"
    assert panel.loci["PolyG5177"].n_reads > 0

    db = tmp_path / "panel.db"
    panel.save_db(db)
    loaded = ExtractedPanel.load_db(db).loci["PolyG151"]
    assert loaded.reference_window_status == "ref_window_unrecoverable"
    assert loaded.upstream_flank is None and loaded.downstream_flank is None


def test_a_status_row_still_counts_against_the_bed(tmp_path):
    from prolyg_phasing.io.bam import extract_panel

    bam_path, bed = _two_locus_bam(tmp_path)
    flanks = fc.write_flank_table(fc.build_flanks(bam_path, bed, n_workers=1, sample="N"),
                                  tmp_path / "normal")
    rows = bed.read_text().splitlines(keepends=True)
    only_window = tmp_path / "window.bed"
    only_window.write_text(rows[0])
    with pytest.raises(ValueError, match=r"loci not in the BED: \['PolyG151'\]"):
        extract_panel(bam_path, only_window, flanks)
    sides = pd.read_csv(flanks, sep="\t", dtype=str, keep_default_na=False)
    short = tmp_path / "short_sides.tsv"
    sides[sides.locus != "PolyG151"].to_csv(short, sep="\t", index=False)
    with pytest.raises(ValueError, match=r"no flanks for 1 BED loci: \['PolyG151'\]"):
        extract_panel(bam_path, bed, short)


def _side_table(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "sides.tsv"
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def _side_row(side: str, status: str, bases: str = "", positions: str = "") -> dict:
    return {"sample": "N", "locus": "L", "side": side, "reference_window_status": status,
            "flank_bases": bases, "flank_ref_positions": positions}


def test_flank_table_rejects_sides_with_different_statuses(tmp_path):
    path = _side_table(tmp_path, [
        _side_row("upstream", "ref_window_unrecoverable"),
        _side_row("downstream", "recovered", "TTTTTTTTTT", ",".join(map(str, range(10))))])
    with pytest.raises(ValueError, match="differ in reference window status"):
        read_flank_table(path)


def test_flank_table_rejects_flank_fields_without_a_window(tmp_path):
    path = _side_table(tmp_path, [
        _side_row("upstream", "ref_window_unrecoverable", "TTTTTTTTTT"),
        _side_row("downstream", "ref_window_unrecoverable")])
    with pytest.raises(ValueError, match="carries flank fields"):
        read_flank_table(path)


def test_a_window_with_no_edge_base_still_gives_the_strand(tmp_path):
    """The G run reaches the end of the reference window. The window still
    gives the strand; the one read has no MI tag, so no read is kept and the
    locus has no flank pair."""
    contig = "A" * 10 + "G" * 100 + "A" * 10
    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "g", "LN": len(contig)}]})
    bam_path = tmp_path / "g.bam"
    with pysam.AlignmentFile(str(bam_path), "wb", header=header) as out:
        seg = pysam.AlignedSegment(header)
        seg.query_name = "whole"
        seg.query_sequence = contig
        seg.query_qualities = pysam.qualitystring_to_array("I" * len(contig))
        seg.reference_id = 0
        seg.reference_start = 0
        seg.mapping_quality = 60
        seg.cigartuples = [(0, len(contig))]
        seg.set_tags([("MD", str(len(contig))), ("NM", 0)])
        out.write(seg)
    pysam.index(str(bam_path))
    with pysam.AlignmentFile(str(bam_path)) as bam:
        loc = fc.locus_reads(bam, "G60", "g", 60, 70)
        assert (loc.ref_orientation, loc.reads) == ("+", [])
        solved = fc.solve_locus(bam, "G60", "g", 60, 70)
    assert solved.pair is None and solved.n_reads == 0
