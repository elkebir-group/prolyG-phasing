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
    place_flanks,
    place_polyg_reads,
    read_flank_table,
    tract_edges,
    window_costs,
)
from tests import _slow_flank_eval as slow
from tests._hand_built_reads import (
    CHROM_PAD,
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


def _check(reads: list[str], u: str, v: str, tracts: set[str]) -> fc.FlankPair:
    """The expected pair is optimal, F equals the slow reference, and every
    optimal pair gives the expected tracts."""
    pair = fc.solve_flank_pair(reads)
    counted = sorted({s: reads.count(s) for s in reads}.items())
    assert (u, v) in pair.optimal_pairs
    assert pair.objective_half_units == slow.objective(counted, u, v)[0]
    for pu, pv in pair.optimal_pairs:
        got = place_polyg_reads(reads, pu, pv)
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
    counted = sorted({s: reads.count(s) for s in reads}.items())
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
    counted = sorted({s: reads.count(s) for s in reads}.items())
    assert slow.objective(counted, UP, DN) == (30 * 20, 30)
    pair = fc.solve_flank_pair(reads)
    for pu, pv in pair.optimal_pairs:
        assert slow.objective(counted, pu, pv)[0] == pair.objective_half_units


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
    assert slow.best_placement(decoy, UP, DN)[0] == -80
    pair = _check(reads, UP, DN, {"G" * 10})
    assert pair.objective_half_units == 1200
    assert pair.n_reads_placed == 100


def test_a_non_g_base_and_three_g_next_to_the_run_join_the_tract():
    """No candidate flank stops in front of a non-G base and three G."""
    for hap, tract in ((LEFT + UP + "G" * 10 + "CGGG" + DN + RIGHT, "G" * 10 + "CGGG"),
                       (LEFT + UP + "GGGA" + "G" * 10 + DN + RIGHT, "GGGA" + "G" * 10)):
        reads = _reads_from(hap, 60)
        pair = _check(reads, UP, DN, {tract})
        search = fc._LocusSearch(sorted(set(reads)), [1] * len(set(reads)))
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
        counted = sorted({s: reads.count(s) for s in reads}.items())
        us, vs = slow.candidates(counted)
        values = {(u, v): slow.objective(counted, u, v)[0] for u in us for v in vs}
        best = max(values.values(), default=0)
        pair = fc.solve_flank_pair(reads)
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
    assert fc.solve_flank_pair(["ACTACTACTACTACTACTACT"] * 5) is None
    assert fc.solve_flank_pair([]) is None


# ---------------------------------------------------------------------------
# The side table
# ---------------------------------------------------------------------------


def _locus_flanks(records, name="L") -> fc.LocusFlanks:
    pair = fc.FlankPair("ACTTCAGCAT", "CAAATTTTAT", 41, (("ACTTCAGCAT", "CAAATTTTAT"),),
                        30, 25)
    return fc.LocusFlanks(name, "+", pair, records)


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


def test_a_flank_that_does_not_match_the_reference_stops_construction():
    up = FlankRecord("ACTTCAGCAT", tuple(range(100, 110)), 9)
    with pytest.raises(ValueError, match=r"1 flanks do not match the reference window: "
                                         r"\['L downstream'\]"):
        fc.construct_flanks([("L", _locus_flanks((up, None)))], sample="S1")


def test_a_locus_with_no_positive_pair_stops_construction():
    with pytest.raises(ValueError, match=r"no flank pair reaches F > 0 at 1 loci: \['L'\]"):
        fc.construct_flanks([("L", fc.LocusFlanks("L", "+", None, (None, None)))],
                            sample="S1")


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
        out = place_flanks([r.query_sequence for r in records], upstream=up, downstream=dn,
                           ref_orientation=o)
        assert {run_tuple(t) for t in out.tracts} == {(13,)}

    path = fc.write_flank_table(result, tmp_path / "normal")
    table = read_flank_table(path)
    assert table.records == result.records
    assert table.sample == "normal"


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
                   "n_reads", "n_reads_placed", "n_tied_pairs"):
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


def test_a_window_with_no_edge_base_is_an_error(tmp_path):
    """The G run reaches the end of the reference window: no flank can exist."""
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
    with pysam.AlignmentFile(str(bam_path)) as bam, \
            pytest.raises(ValueError, match="G60: the reference window holds no edge base"):
        fc.locus_reads(bam, "G60", "g", 60, 70)
