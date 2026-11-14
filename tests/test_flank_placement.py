"""Tests for the flank match and for each read's tract under one pair of flanks.

The match and the placement are checked against the plain reference in
`tests/_slow_flank_eval.py`, which shares no code with the library. Each
hand-built case in `tests/_hand_built_reads.py` is one read with a known
true run tuple at a real hg38 locus, placed with the case's flanks in both
locus orientations. The tract depends only on the read string, so every
alignment form of a case must give the same result.

The expected outcome of a case is its true tuple or its drop reason,
except for the cases in `OUTCOME`, where the objective assigns another
tuple. Each entry states why.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from prolyg_phasing.io._anchors import revcomp
from prolyg_phasing.io._flank_placement import (
    FLANK_LEN,
    MAX_MATCH_LEN,
    NO_MATCH,
    PAD,
    SHAPES,
    FlankPlacement,
    FlankRecord,
    code_matrix,
    flank_codes,
    locate_flank,
    place_flanks,
    place_polyg_reads,
    window_costs,
)
from tests import _slow_flank_eval as slow
from tests._hand_built_reads import (
    CHROM_PAD,
    LOCI,
    G,
    ReadCase,
    all_records,
    build_cases,
    chromosome,
    placements,
    run_stutter_with_flank_error_cases,
    run_tuple,
    soft_clipped_flank_case,
)

CASES = build_cases() + [soft_clipped_flank_case()]
CASE_IDS = [c.case_id for c in CASES]
STUTTER_CASES = run_stutter_with_flank_error_cases()
ALL_CASES = CASES + STUTTER_CASES

DROP_REASONS = ("upstream_flank_not_placed", "downstream_flank_not_placed", "no_placement",
                "negative_value")

# Cases whose outcome under the objective is not their true tuple or drop
# reason, and why.
OUTCOME: dict[str, tuple[int, ...] | str] = {
    # A tract starts and ends with G. The reference allele T T G10 has no G
    # after the flank's outer T, so the flank matches at the inner T with a
    # gap, and the inner T stays outside the tract.
    "edge_variant_het_ref": (10,),
    # The same: the read of T G10 with the outer T deleted, or of G10.
    "boundary_t_deletion": (10,),
    # TTTGC A G9: the flank ends at the C, and no G follows it. The A after
    # it is a non-G base between the flank and the G run, so the read has no
    # match of the flank at a tract edge.
    "edge_interrupter": "upstream_flank_not_placed",
    # The G at the outer T is read as an extra G of the flank's GG, a 1-base
    # gap. The tract is G10, the tract that T G10 has under the objective.
    "false_g_moved": (10,),
    # G9 with a G at the downstream T is also G10 with one T of TT deleted,
    # a 1-base gap in the downstream flank. The objective reads one G more
    # than the mate shows.
    "run_deletion_g_at_downstream_run_side": (10,),
    "two_g_deletion_g_at_downstream_run_side": (9,),
    # The same on the upstream side: one T of TTT deleted.
    "run_deletion_g_at_upstream_run_side": (10,),
}


def expected_outcome(case: ReadCase) -> tuple[int, ...] | str:
    """The tuple or the drop reason the case must give."""
    if case.case_id in OUTCOME:
        return OUTCOME[case.case_id]
    return case.drop_reason if case.drop_reason is not None else case.true_tuple


def outcome(result: FlankPlacement, i: int) -> tuple[int, ...] | str:
    """Read `i`'s tuple, or its single drop reason."""
    if result.kept[i]:
        return run_tuple(result.tracts[i])
    reasons = drop_reasons(result, i)
    assert len(reasons) == 1, reasons
    return reasons.pop()


def drop_reasons(result: FlankPlacement, i: int) -> set[str]:
    """Every drop reason of read `i`: one per side with no match, or one per read."""
    reasons = set()
    if not result.upstream_matched[i]:
        reasons.add("upstream_flank_not_placed")
    if not result.downstream_matched[i]:
        reasons.add("downstream_flank_not_placed")
    if result.no_placement[i]:
        reasons.add("no_placement")
    if result.negative_value[i]:
        reasons.add("negative_value")
    return reasons


def flank_record(case: ReadCase, spec, orientation: str) -> FlankRecord:
    """The flank record of a case side on its synthetic chromosome."""
    n = len(case.locus.reference)
    if orientation == "+":
        positions = tuple(CHROM_PAD + spec.start + i for i in range(FLANK_LEN))
    else:
        positions = tuple(CHROM_PAD + n - 1 - (spec.start + i) for i in range(FLANK_LEN))
    return FlankRecord(bases=spec.bases, ref_positions=positions,
                       run_side_offset=spec.run_side_offset)


def _place(case: ReadCase, orientation: str) -> FlankPlacement:
    records = [rec for _, rec in all_records(case, orientation)]
    return place_flanks(
        [r.query_sequence for r in records],
        upstream=flank_record(case, case.up, orientation),
        downstream=flank_record(case, case.dn, orientation),
        ref_orientation=orientation,
    )


# ---------------------------------------------------------------------------
# The fixtures themselves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ALL_CASES, ids=[c.case_id for c in ALL_CASES])
@pytest.mark.parametrize("orientation", ["+", "-"])
def test_every_placement_is_a_valid_alignment(case: ReadCase, orientation: str):
    """The MD tag and CIGAR rebuild the synthetic reference under every alignment."""
    chrom = chromosome(case.locus, orientation)
    records = list(all_records(case, orientation))
    assert records
    for placement, rec in records:
        ref = rec.get_reference_sequence().upper()
        assert ref == chrom[rec.reference_start:rec.reference_end], placement
        seq = rec.query_sequence if orientation == "+" else revcomp(rec.query_sequence)
        assert seq == case.read


@pytest.mark.parametrize("case", ALL_CASES, ids=[c.case_id for c in ALL_CASES])
def test_flanks_match_the_reference(case: ReadCase):
    """Each flank is 10 reference bases with a non-G run-side base."""
    ref = case.locus.reference
    for flank in (case.up, case.dn):
        assert len(flank.bases) == FLANK_LEN
        assert ref[flank.start:flank.start + FLANK_LEN] == flank.bases
        assert flank.bases[flank.run_side_offset] != "G"


def test_gap_shifts_cover_the_homopolymer():
    """A one-base deletion in CCC is listed at every C."""
    case = next(c for c in CASES if c.case_id == "post_run_deletion")
    shifts = [p for p in placements(case) if p.form == "gap_shift"]
    assert len(shifts) == 3


def test_true_tuples_of_the_edge_cases():
    """The tuples the hand-built molecules carry."""
    by_id = {c.case_id: c for c in CASES}
    assert by_id["edge_variant_hom"].true_tuple == (13,)
    assert by_id["edge_variant_het_ref"].true_tuple == (0, 10)
    assert by_id["edge_variant_het_g"].true_tuple == (12,)
    assert by_id["deleted_edge_base"].true_tuple == (6,)
    assert by_id["edge_interrupter"].true_tuple == (0, 9)
    assert by_id["two_candidates_moved"].true_tuple == (13,)
    assert by_id["flank_moves_four"].true_tuple == (10, 2)


def test_run_tuple_matches_the_extractor_parse():
    from prolyg_phasing.io.panel import parse_run_lengths

    for tract in ("", "GGG", "TGGG", "GGGT", "GGATGG", "AG", "TT", "GGGTTGGGAA"):
        assert run_tuple(tract) == parse_run_lengths(tract)


def test_loci_table_is_polyg_oriented():
    """Every locus reads as Gs between its edge bases."""
    for locus in LOCI.values():
        assert set(locus.tract) == {"G"}
        assert locus.upstream[-1] != "G"
        assert locus.downstream[0] != "G"


# ---------------------------------------------------------------------------
# The match
# ---------------------------------------------------------------------------


def _window(substring: str) -> np.ndarray:
    """A one-row window that ends with `substring`, PAD before it."""
    codes = code_matrix([substring])[0][0][-MAX_MATCH_LEN:]
    out = np.full(MAX_MATCH_LEN, PAD, dtype=np.uint8)
    out[MAX_MATCH_LEN - len(codes):] = codes
    return out[None, :]


def _reference_window_cost(substring: str, flank: str) -> int:
    """The lowest slow-reference cost over the substring's last 8 to 12 bases."""
    costs = [slow.substring_cost(substring[-k:], flank)
             for k in range(FLANK_LEN - 2, FLANK_LEN + 3) if k <= len(substring)]
    costs = [c for c in costs if c is not None]
    return min(costs) if costs else NO_MATCH


def test_shapes_and_costs():
    """One gapless shape, and gaps of 1 and 2 bases between two inner flank bases."""
    kinds = {(s.kind, s.gap): 0 for s in SHAPES}
    for s in SHAPES:
        kinds[(s.kind, s.gap)] += 1
    assert kinds == {("none", 0): 1, ("insertion", 1): 9, ("deletion", 1): 8,
                     ("insertion", 2): 9, ("deletion", 2): 7}
    assert {(s.gap, s.cost) for s in SHAPES} == {(0, 0), (1, 4), (2, 5)}
    for s in SHAPES:
        assert s.read_slots[0] == 0 and s.read_slots[-1] == s.length - 1
        assert s.flank_slots[0] == 0 and s.flank_slots[-1] == FLANK_LEN - 1


def _mutants(flank: str, rng: random.Random) -> list[str]:
    """Substrings near `flank`: every single and double edit kind, and random strings."""
    out = [flank]
    for _ in range(40):
        s = list(flank)
        for _ in range(rng.choice([1, 1, 2])):
            op = rng.choice(["sub", "ins", "del"])
            i = rng.randrange(len(s) + (op == "ins"))
            if op == "sub" and s:
                s[i % len(s)] = rng.choice("ACGTN")
            elif op == "ins":
                s[i:i] = rng.choices("ACGT", k=rng.choice([1, 2]))
            elif len(s) > 2:
                del s[i % len(s):i % len(s) + rng.choice([1, 2])]
        out.append("".join(s))
    out += ["".join(rng.choices("ACGT", k=rng.randint(8, 12))) for _ in range(5)]
    return out


def test_window_costs_match_the_slow_reference():
    rng = random.Random(3)
    for _ in range(200):
        flank = "".join(rng.choices("ACGT", k=FLANK_LEN))
        subs = _mutants(flank, rng)
        windows = np.vstack([_window(("A" * 4 + s) if rng.random() < 0.5 else s)
                             for s in subs])
        got = window_costs(windows, flank_codes(flank))[0]
        for w, c in zip(windows, got, strict=True):
            text = "".join("ACGTN."[x] for x in w).lstrip(".")
            assert c == _reference_window_cost(text, flank), (flank, text)


def test_a_gap_and_a_substitution_never_together():
    flank = "ACTGCATCCA"
    gapped = flank[:4] + "T" + flank[4:]                  # one inserted base: cost 4
    both = gapped[:7] + "G" + gapped[8:]                  # and one substitution
    assert window_costs(_window(gapped), flank_codes(flank))[0][0] == 4
    assert window_costs(_window(both), flank_codes(flank))[0][0] == NO_MATCH


def test_a_gap_at_the_flank_end_is_not_a_match():
    flank = "ACTGCATCCA"
    assert window_costs(_window(flank[:-1]), flank_codes(flank))[0][0] == NO_MATCH
    assert window_costs(_window("T" + flank[1:-1] + "A"), flank_codes(flank))[0][0] == 2


def test_a_read_n_is_a_substitution():
    flank = "ACTGCATCCA"
    assert window_costs(_window(flank[:3] + "N" + flank[4:]), flank_codes(flank))[0][0] == 2
    assert window_costs(_window("N" + flank[1:3] + "N" + flank[4:]),
                        flank_codes(flank))[0][0] == NO_MATCH


# ---------------------------------------------------------------------------
# Each read's best placement
# ---------------------------------------------------------------------------

UP, DN = "ACTTCAGCAT", "CAAATTTTAT"


def _random_read(rng: random.Random) -> str:
    """A read around a G run, with read errors in and next to the flanks."""
    hap = ("".join(rng.choices("ACT", k=rng.randint(0, 12))) + UP
           + "".join(rng.choices("GGGGGGGT", k=rng.randint(3, 14)))
           + DN + "".join(rng.choices("ACGT", k=rng.randint(0, 12))))
    s = list(hap)
    for _ in range(rng.randint(0, 4)):
        i = rng.randrange(len(s))
        op = rng.random()
        if op < 0.5:
            s[i] = rng.choice("ACGTN")
        elif op < 0.75:
            s[i:i] = rng.choices("ACGT", k=rng.choice([1, 2]))
        else:
            del s[i:i + rng.choice([1, 2])]
    return "".join(s)


def test_placement_matches_the_slow_reference_on_random_reads():
    rng = random.Random(11)
    reads = [_random_read(rng) for _ in range(800)]
    # Two copies of the flank in one read, before and after the run.
    reads += [UP + "GGGG" + UP + "GGGGGG" + DN + "GG" + DN, UP + "GGT" + DN + "GGGG" + DN]
    result = place_polyg_reads(reads, UP, DN)
    n_placed = 0
    for i, read in enumerate(reads):
        want = slow.best_placement(read, UP, DN)
        assert result.upstream_matched[i] == bool(slow.u_positions(read, UP))
        assert result.downstream_matched[i] == bool(slow.v_positions(read, DN))
        if want is None:
            assert not result.placed[i] and result.tracts[i] is None
            continue
        n_placed += 1
        value, a, b = want
        assert result.placed[i]
        assert (result.value[i], result.upstream_run_side[i],
                result.downstream_run_side[i]) == (value, a, b)
        assert result.kept[i] == (value >= 0)
        if value >= 0:
            assert result.tracts[i] == read[a + 1:b]
    assert n_placed > 400


def test_equal_value_placements_keep_the_shortest_tract():
    """`TGGTGGTGGT` matches at each TGG copy, and each GGT adds 0. Of the
    placements with the top value, the largest `a` and then the smallest `b`
    win, so no copy joins the tract."""
    periodic = "TGGTGGTGGT"
    downstream = UP + "GGGGG" + "TGGTGGTGGTGGT" + "AAAAAAAA"
    upstream = "CCCC" + "TGGTGGTGGTGGT" + "GGGGG" + DN
    result = place_polyg_reads([downstream], UP, periodic)
    assert result.tracts[0] == "GGGGG"
    result = place_polyg_reads([upstream], periodic, DN)
    assert result.tracts[0] == "GGGGG"


def _repeat_unit_read(rng: random.Random, unit: str) -> str:
    """A read whose flanks are copies of a repeated unit, with read errors."""
    hap = (unit * rng.randint(3, 6) + "G" * rng.randint(3, 12)
           + "T" + unit * rng.randint(3, 6) + "".join(rng.choices("ACT", k=6)))
    s = list(hap)
    for _ in range(rng.randint(0, 2)):
        i = rng.randrange(len(s))
        op = rng.random()
        if op < 0.5:
            s[i] = rng.choice("ACGT")
        elif op < 0.75:
            s.insert(i, rng.choice("ACGT"))
        else:
            del s[i]
    return "".join(s)


def test_tracts_match_the_slow_reference_where_placements_tie():
    """At flanks of a repeated unit many placements share the top value; the
    chosen (a, b), not only the value, must agree with the reference."""
    rng = random.Random(17)
    n_ties = 0
    for unit, up, dn in (("TGG", "TGGTGGTGGT", "TTGGTGGTGG"),
                         ("AGG", "AGGAGGAGGA", "TAGGAGGAGG"),
                         ("TGGG", "GGTGGGTGGT", "TTGGGTGGGT")):
        reads = [_repeat_unit_read(rng, unit) for _ in range(300)]
        result = place_polyg_reads(reads, up, dn)
        for i, read in enumerate(reads):
            want = slow.best_placement(read, up, dn)
            if want is None:
                assert not result.placed[i]
                continue
            assert (result.value[i], result.upstream_run_side[i],
                    result.downstream_run_side[i]) == want
            ups, downs = slow.u_positions(read, up), slow.v_positions(read, dn)
            tops = [(a, b) for a in ups for b in downs if b > a and
                    slow.score(read[a + 1:b]) - ups[a] - downs[b] == want[0]]
            n_ties += len(tops) > 1
    assert n_ties > 50


def test_negative_value_and_no_placement_are_counted_not_kept():
    decoy = "ACTTCAGCTT" + "G" + "ACTTACATCATTACCATACA" + "G" + "CAAATTTAAT"
    reversed_order = "G" + DN + "TA" + UP + "GGGG"
    result = place_polyg_reads([decoy, reversed_order], UP, DN)
    # G + 20 other bases + G, and one substitution in each flank.
    assert result.placed[0] and result.value[0] == 2 * 2 - 4 * 20 - 2 - 2
    assert not result.kept[0] and result.tracts[0] is None
    assert result.negative_value[0]
    assert result.upstream_matched[1] and result.downstream_matched[1]
    assert result.no_placement[1] and not result.placed[1]


# ---------------------------------------------------------------------------
# The hand-built cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ALL_CASES, ids=[c.case_id for c in ALL_CASES])
@pytest.mark.parametrize("orientation", ["+", "-"])
def test_every_alignment_gives_the_expected_outcome(case: ReadCase, orientation: str):
    result = _place(case, orientation)
    want = expected_outcome(case)
    for i, p in enumerate(placements(case)):
        assert outcome(result, i) == want, p.form


@pytest.mark.parametrize("case", ALL_CASES, ids=[c.case_id for c in ALL_CASES])
def test_the_slow_reference_gives_the_expected_outcome(case: ReadCase):
    """The expected outcome follows from the reference, not from the code under test."""
    want = expected_outcome(case)
    p = slow.best_placement(case.read, case.up.bases, case.dn.bases)
    if isinstance(want, str):
        side = {"upstream_flank_not_placed": slow.u_positions,
                "downstream_flank_not_placed": slow.v_positions}[want]
        bases = case.up.bases if want.startswith("upstream") else case.dn.bases
        assert p is None and not side(case.read, bases)
    else:
        assert p is not None and p[0] >= 0
        assert run_tuple(case.read[p[1] + 1:p[2]]) == want


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_both_orientations_give_the_same_tract(case: ReadCase):
    plus, minus = _place(case, "+"), _place(case, "-")
    assert list(plus.tracts) == list(minus.tracts)
    for field in ("upstream_matched", "downstream_matched", "no_placement", "negative_value",
                  "value"):
        assert list(getattr(plus, field)) == list(getattr(minus, field)), field


def test_every_drop_reason_occurs_in_the_cases():
    seen = {expected_outcome(c) for c in ALL_CASES if isinstance(expected_outcome(c), str)}
    assert seen == {"upstream_flank_not_placed", "downstream_flank_not_placed"}


def test_run_side_indices_bound_the_tract():
    """The polyG-orientation read between the two run-side indices is the tract.

    The pair merge reads the tract qualities over these indices.
    """
    for case in CASES:
        for orientation in "+-":
            records = [rec for _, rec in all_records(case, orientation)]
            result = _place(case, orientation)
            for i, rec in enumerate(records):
                if not result.kept[i]:
                    continue
                read = rec.query_sequence if orientation == "+" \
                    else revcomp(rec.query_sequence)
                lo, hi = result.upstream_run_side[i] + 1, result.downstream_run_side[i]
                assert read[lo:hi] == result.tracts[i]
                assert type(result.tracts[i]) is str


def test_a_tract_starts_and_ends_with_g():
    for case in CASES:
        result = _place(case, "+")
        for tract in result.tracts[result.kept]:
            assert tract[0] == G and tract[-1] == G


# ---------------------------------------------------------------------------
# The flank in the reference
# ---------------------------------------------------------------------------

_WINDOW = "TTACCAGTCA" + "ACTTCAGCAT" + "GGGGGGGGGGGG" + "CAAATTTTAT" + "ACCAGTCATT"
_SPLIT = 26


def test_locate_flank_exact_on_each_side():
    assert locate_flank(_WINDOW, "ACTTCAGCAT", upstream=True, split=_SPLIT) == (10, 19)
    assert locate_flank(_WINDOW, "CAAATTTTAT", upstream=False, split=_SPLIT) == (32, 41)


def test_locate_flank_by_the_read_match_rule():
    # One substitution: the same reference bases.
    assert locate_flank(_WINDOW, "ACTTGAGCAT", upstream=True, split=_SPLIT) == (10, 19)
    # The donor flank has one base more than the reference: 9 reference bases.
    assert locate_flank(_WINDOW, "CTTCAGGCAT", upstream=True, split=_SPLIT) == (11, 19)
    # The donor flank lacks a reference base: 11 reference bases.
    assert locate_flank(_WINDOW, "CAAATTTATA", upstream=False, split=_SPLIT) == (32, 42)
    # Two edits: no match.
    assert locate_flank(_WINDOW, "CAAAGGTTAT", upstream=False, split=_SPLIT) is None


def test_locate_flank_takes_the_copy_nearest_the_repeat():
    copy = "TTACCAGTCA"
    window = copy + "CA" + copy + "GGGGGGGG" + copy + "CA" + copy
    assert locate_flank(window, copy, upstream=True, split=26) == (12, 21)
    assert locate_flank(window, copy, upstream=False, split=26) == (30, 39)


def test_locate_flank_prefers_the_lower_cost_over_the_nearer_copy():
    window = "TTACCAGTCA" + "CA" + "TTACGAGTCA" + "GGGGGGGG" + "AAAA"
    assert locate_flank(window, "TTACCAGTCA", upstream=True, split=26) == (0, 9)


def test_locate_flank_stays_on_its_side():
    assert locate_flank(_WINDOW, "CAAATTTTAT", upstream=True, split=_SPLIT) is None
    assert locate_flank(_WINDOW, "ACTTCAGCAT", upstream=False, split=_SPLIT) is None
