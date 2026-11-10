"""Hand-built-read tests for the per-read flank placement kernel.

Each case in `tests/_hand_built_reads.py` is one read with a known true run
tuple, at a real hg38 locus. Every alignment an aligner can choose for that
read string is attached as a `pysam` record. The kernel must return the
true tuple, or the stated drop reason, for every alignment and in both
locus orientations.

The fixture tests check that every generated alignment is a valid
alignment with a correct MD tag. The plain dynamic-programming reference
`flank_edit_distance` checks the kernel's band-1 search.
"""

from __future__ import annotations

import dataclasses
import itertools
import random

import numpy as np
import pytest

import prolyg_phasing.io._flank_placement as flank_placement
from prolyg_phasing.io._anchors import revcomp
from prolyg_phasing.io._flank_placement import (
    FlankPlacement,
    FlankRecord,
    banded_flank_distance,
    place_flanks,
)
from tests._hand_built_reads import (
    CHROM_PAD,
    FLANK_LEN,
    LOCI,
    G,
    ReadCase,
    _case,
    _del,
    _dn,
    _sub,
    _up,
    all_records,
    build_cases,
    chromosome,
    flank_edit_cost,
    flank_edit_distance,
    placements,
    run_stutter_with_flank_error_cases,
    run_tuple,
    soft_clipped_flank_case,
)

CASES = build_cases() + [soft_clipped_flank_case()]
CASE_IDS = [c.case_id for c in CASES]
# Every hand-built case, those pinned as known failures included.
ALL_CASES = CASES + run_stutter_with_flank_error_cases()

# The paths each case must use across its alignments. "fast": the alignment
# covers the flank with M operations and the read matches it at every
# unmasked position. "cigar": the alignment has one indel inside the flank.
# "local": the window search around the projected position.
EXPECTED_PATHS: dict[str, set[str]] = {
    "edge_variant_hom": {"fast", "local"},
    "edge_variant_het_ref": {"fast"},
    "edge_variant_het_g": {"fast", "local"},
    "deleted_edge_base": {"fast", "local"},
    "false_g_invariant": {"fast"},
    "edge_interrupter": {"fast", "local"},
    "tgg_flank_extra_copy": {"fast"},
    "tgg_flank_missing_copy": {"fast"},
    "tandem_copy_flank": {"fast"},
    "post_run_deletion": {"fast", "cigar", "local"},
    "post_run_insertion": {"fast", "local"},
    "post_run_deletion_aa": {"fast", "cigar", "local"},
    "pre_run_deletion": {"fast", "cigar", "local"},
    "snv_in_flank": {"fast", "local"},
    "non_g_run_side": {"fast", "local"},
    "false_g_moved": {"fast"},
    "invariant_side": {"fast"},
    "two_candidates_moved": {"fast", "local"},
    "flank_moves_four": {"fast"},
    "masked_plus_mismatch": {"fast", "local"},
    "two_edits": {"fast"},
    "near_end_downstream": {"fast", "local"},
    "near_end_upstream": {"fast", "local"},
    "boundary_t_deletion": {"fast", "local"},
    "boundary_t_insertion": {"fast", "local"},
    "tgg_flank_substitution": {"fast", "local"},
    "tandem_copy_error_in_near_copy": {"fast", "local"},
    "soft_clip_in_flank": {"fast", "local"},
    "read_ends_in_tract": {"fast"},
    "tract_and_edge_deleted": {"fast", "local"},
    "two_base_deletion_in_flank": {"fast"},
}

# Alignment forms whose drop reason differs from the case's. A read that
# lost its whole tract differs from the reference by more bases than the
# window search covers. The gapless forms show that change as mismatches
# over the downstream flank, so the window cannot reach the placement that
# crosses the upstream one, and the read drops as "flank_not_placed".
DROP_REASON_BY_FORM: dict[str, dict[str, str]] = {
    "tract_and_edge_deleted": {
        "gapless_start_anchored": "flank_not_placed",
        "gapless_start_anchored_tail_clipped": "flank_not_placed",
    },
}

# Reads whose string also comes, with no error or with one flank error
# fewer, from a molecule with another tuple, so no rule that reads the
# string can recover the molecule the read was built from. The kernel puts
# a missing copy of the run-side base in the flank, and an extra non-G base
# after the run in the tract, as a boundary zero run: a flank indel costs
# twice a mismatch. These are the tuples that gives.
ASSIGNED_TUPLE: dict[str, tuple[int, ...]] = {
    # Reference allele T T G10 with one T deleted reads as T G10, which is
    # also a molecule with the edge T deleted.
    "boundary_t_deletion": (10,),
    # G14 with a T inserted after the run reads as G14 T T..., which is also
    # a molecule with a T interrupter at the end of the tract.
    "boundary_t_insertion": (14, 0),
    # G14 with a C inserted in CCC reads as G14 T CCCC..., which is also a
    # molecule with a T at the end of the tract and a C run-side base.
    "post_run_insertion": (14, 0),
}


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


def test_tandem_insertion_shifts_cover_the_repeat():
    """A TGG insertion is listed at every position along the (TGG)n and into the run."""
    case = next(c for c in CASES if c.case_id == "tgg_flank_extra_copy")
    shifts = [p for p in placements(case) if p.form == "gap_shift"]
    # (TGG)5 T GG: the inserted copy slides over the 16 repeat bases and 2
    # bases into the G run, from read offset 9 to read offset 27 inclusive.
    assert [p.cigar[0][1] for p in shifts] == list(range(9, 28))


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


# ---------------------------------------------------------------------------
# The plain DP reference
# ---------------------------------------------------------------------------


def test_reference_distance_exact_and_one_edit():
    read = "AAATATTTAGATGGGG"
    flank = "ATATTTAGAT"
    b = read.index("TGGGG")
    assert flank_edit_distance(flank, read, b, upstream=True) == 0
    assert flank_edit_distance(flank, read[:4] + read[5:], b - 1, upstream=True) == 1
    assert flank_edit_distance(flank, read[:6] + "C" + read[7:], b, upstream=True) == 1
    # Read index 6 holds flank offset 4.
    assert flank_edit_distance(flank, read[:6] + "C" + read[7:], b, upstream=True,
                               masked=(4,)) == 0


def test_reference_distance_downstream_mirrors_upstream():
    rng = random.Random(7)
    for _ in range(200):
        flank = "".join(rng.choice("ACGT") for _ in range(FLANK_LEN))
        read = "".join(rng.choice("ACGT") for _ in range(16))
        b = rng.randrange(len(read))
        up = flank_edit_distance(flank, read, b, upstream=True)
        dn = flank_edit_distance(flank[::-1], read[::-1], len(read) - 1 - b, upstream=False)
        assert up == dn


def test_run_tuple_matches_the_extractor_parse():
    from prolyg_phasing.io.panel import parse_run_lengths

    for tract in ("", "GGG", "TGGG", "GGGT", "GGATGG", "AG", "TT", "GGGTTGGGAA"):
        assert run_tuple(tract) == parse_run_lengths(tract)


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


def drop_reasons(result: FlankPlacement, i: int) -> set[str]:
    """Every drop reason of read `i`: one per failed side, and a crossing."""
    reasons = {result.upstream_drop[i], result.downstream_drop[i]} - {None}
    if result.flanks_crossed[i]:
        reasons.add("flanks_crossed")
    return reasons


def _flank_record(case: ReadCase, spec, orientation: str) -> FlankRecord:
    n = len(case.locus.reference)
    if orientation == "+":
        positions = tuple(CHROM_PAD + spec.start + i for i in range(FLANK_LEN))
    else:
        positions = tuple(CHROM_PAD + n - 1 - (spec.start + i) for i in range(FLANK_LEN))
    return FlankRecord(
        bases=spec.bases,
        ref_positions=positions,
        run_side_offset=spec.run_side_offset,
        masked_offsets=spec.masked_offsets,
    )


def _place(case: ReadCase, orientation: str) -> FlankPlacement:
    records = [rec for _, rec in all_records(case, orientation)]
    return place_flanks(
        [r.query_sequence for r in records],
        [r.cigartuples for r in records],
        np.array([r.reference_start for r in records], dtype=np.int64),
        upstream=_flank_record(case, case.up, orientation),
        downstream=_flank_record(case, case.dn, orientation),
        ref_orientation=orientation,
        reference_window=chromosome(case.locus, orientation),
        reference_window_start=0,
    )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("orientation", ["+", "-"])
def test_every_placement_gives_the_true_tuple(case: ReadCase, orientation: str):
    result = _place(case, orientation)
    forms = [p.form for p in placements(case)]
    for i, form in enumerate(forms):
        if case.drop_reason is not None:
            expected_drop = DROP_REASON_BY_FORM.get(case.case_id, {}).get(form, case.drop_reason)
            assert drop_reasons(result, i) == {expected_drop}, form
            assert not result.kept[i] and result.tracts[i] is None, form
            continue
        assert drop_reasons(result, i) == set(), form
        assert result.kept[i], form
        expected = ASSIGNED_TUPLE.get(case.case_id, case.true_tuple)
        assert run_tuple(result.tracts[i]) == expected, form


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("orientation", ["+", "-"])
def test_paths_per_case(case: ReadCase, orientation: str):
    result = _place(case, orientation)
    used = set(result.upstream_path) | set(result.downstream_path)
    used.discard(None)
    assert used == EXPECTED_PATHS[case.case_id]


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_both_orientations_give_the_same_tract(case: ReadCase):
    plus, minus = _place(case, "+"), _place(case, "-")
    assert list(plus.tracts) == list(minus.tracts)
    for field in ("upstream_drop", "downstream_drop", "flanks_crossed", "upstream_path",
                  "downstream_path"):
        assert list(getattr(plus, field)) == list(getattr(minus, field)), field


def _all_strings(alphabet: str, max_len: int):
    for n in range(max_len + 1):
        for t in itertools.product(alphabet, repeat=n):
            yield "".join(t)


def test_band1_search_matches_the_reference_exhaustively():
    """Every flank of length 4 and read of length up to 5 over {A, G, T}."""
    flanks = ["".join(t) for t in itertools.product("AGT", repeat=4)]
    reads = [r for r in _all_strings("AGT", 5) if r]
    for flank in flanks:
        for upstream in (True, False):
            rows = [(r, b) for r in reads for b in range(len(r))]
            got = banded_flank_distance(
                flank, [r for r, _ in rows], np.array([b for _, b in rows]),
                upstream=upstream, masked_offsets=(),
            )
            want = [flank_edit_cost(flank, r, b, upstream=upstream) for r, b in rows]
            assert list(got) == want


def test_one_edit_limit_on_random_placements():
    """One mismatch or one 1-base indel is admitted; two edits of any kind never are.

    12,000 placements of a 10-base flank with 0 to 2 random edits, checked
    against the plain DP edit distance.
    """
    rng = random.Random(3)
    admitted = rejected = 0
    for _ in range(300):
        flank = "".join(rng.choice("ACGT") for _ in range(FLANK_LEN))
        reads = []
        for _ in range(40):
            read = list(flank)
            for _ in range(rng.randrange(3)):
                i = rng.randrange(len(read))
                kind = rng.randrange(3)
                if kind == 0:
                    read[i] = rng.choice("ACGT")
                elif kind == 1:
                    del read[i]
                else:
                    read.insert(i, rng.choice("ACGT"))
            reads.append("AAAAA" + "".join(read) + "CCCCC")
        run_side = np.array([len(r) - 6 for r in reads])
        got = banded_flank_distance(flank, reads, run_side, upstream=True, masked_offsets=())
        for read, b, cost in zip(reads, run_side, got.tolist(), strict=True):
            one_edit = flank_edit_distance(flank, read, int(b), upstream=True) <= 1
            assert (cost < flank_placement._FAR) == one_edit
            admitted += one_edit
            rejected += not one_edit
    assert admitted > 1000 and rejected > 1000


def test_band1_search_matches_the_reference_on_random_reads():
    rng = random.Random(19)
    for _ in range(200):
        flank = "".join(rng.choice("ACGT") for _ in range(FLANK_LEN))
        masked = tuple(sorted(rng.sample(range(FLANK_LEN), rng.randrange(3))))
        reads = []
        for _ in range(64):
            read = list(flank)
            for _ in range(rng.randrange(3)):
                i = rng.randrange(len(read))
                kind = rng.randrange(3)
                if kind == 0:
                    read[i] = rng.choice("ACGT")
                elif kind == 1:
                    del read[i]
                else:
                    read.insert(i, rng.choice("ACGT"))
            reads.append("".join(rng.choice("ACGT") for _ in range(5)) + "".join(read)
                         + "".join(rng.choice("ACGT") for _ in range(5)))
        for upstream in (True, False):
            # The edited flank sits between 5 random bases on each side; its
            # run-side base is its last base upstream and its first downstream.
            run_side = np.array([len(r) - 6 if upstream else 5 for r in reads])
            got = banded_flank_distance(flank, reads, run_side, upstream=upstream,
                                        masked_offsets=masked)
            want = [flank_edit_cost(flank, r, int(b), upstream=upstream, masked=masked)
                    for r, b in zip(reads, run_side, strict=True)]
            assert list(got) == want


def test_every_path_is_used_on_both_sides():
    used = {"upstream": set(), "downstream": set()}
    for case in CASES:
        for orientation in "+-":
            result = _place(case, orientation)
            used["upstream"] |= set(result.upstream_path) - {None}
            used["downstream"] |= set(result.downstream_path) - {None}
    assert used["upstream"] == {"fast", "cigar", "local"}
    assert used["downstream"] == {"fast", "cigar", "local"}


def test_every_drop_reason_is_used():
    seen = set()
    for case in CASES:
        for orientation in "+-":
            result = _place(case, orientation)
            for i in range(len(result.kept)):
                seen |= drop_reasons(result, i)
    assert seen == set(flank_placement.DROP_REASONS)


@pytest.mark.parametrize("orientation", ["+", "-"])
def test_window_must_cover_the_length_change(monkeypatch, orientation: str):
    """PolyG454 reads C G6 against reference C C G10: a change of 5 bases.

    The gapless alignment puts the downstream flank 5 bases off its
    projection. A window of 4 cannot reach the true position, so the read
    is lost; the module window of 8 places it.
    """
    case = next(c for c in CASES if c.case_id == "deleted_edge_base")
    forms = [p.form for p in placements(case)]
    i = forms.index("gapless_end_anchored")
    assert run_tuple(_place(case, orientation).tracts[i]) == (6,)
    monkeypatch.setattr(flank_placement, "_LOCAL_WINDOW", 4)
    assert _place(case, orientation).tracts[i] != case.true_tract


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


def test_loci_table_is_polyg_oriented():
    """Every locus reads as Gs between its edge bases."""
    for locus in LOCI.values():
        assert set(locus.tract) == {"G"}
        assert locus.upstream[-1] != "G"
        assert locus.downstream[0] != "G"


# ---------------------------------------------------------------------------
# A drop reason per side
# ---------------------------------------------------------------------------


def _place_reads(case: ReadCase, orientation: str, cigars=None) -> FlankPlacement:
    """Place every alignment of `case`, optionally with the CIGARs replaced."""
    records = [rec for _, rec in all_records(case, orientation)]
    return place_flanks(
        [r.query_sequence for r in records],
        cigars if cigars is not None else [r.cigartuples for r in records],
        np.array([r.reference_start for r in records], dtype=np.int64),
        upstream=_flank_record(case, case.up, orientation),
        downstream=_flank_record(case, case.dn, orientation),
        ref_orientation=orientation,
        reference_window=chromosome(case.locus, orientation),
        reference_window_start=0,
    )


@pytest.mark.parametrize("orientation", ["+", "-"])
def test_both_sides_report_their_own_reason(orientation):
    """A G at the upstream run-side base and two edits in the downstream flank."""
    up151 = _up("PolyG151")
    dn151 = _dn("PolyG151")
    case = _case("both_sides_fail", "PolyG151: G at the edge T, two downstream edits",
                 "PolyG151", up_ctx=up151[:-1] + "G", tract=G * 12,
                 dn_ctx=_sub(_sub(dn151, 3, "C"), 6, "A"),
                 drop_reason="g_at_run_side", true_tract=None)
    forms = [p.form for p in placements(case)]
    result = _place_reads(case, orientation)
    i = forms.index("gapless")
    assert result.upstream_drop[i] == "g_at_run_side"
    assert result.downstream_drop[i] == "flank_not_placed"
    assert not result.flanks_crossed[i] and not result.kept[i]
    assert result.upstream_path[i] is None and result.downstream_path[i] is None


@pytest.mark.parametrize("orientation", ["+", "-"])
def test_a_read_inside_the_tract_is_not_covered_on_either_side(orientation):
    locus = LOCI["PolyG151"]
    case = _case("inside_tract", "PolyG151: a read of 10 tract Gs", "PolyG151",
                 up_ctx="", tract=G * 10, dn_ctx="", start=len(locus.upstream) + 1,
                 drop_reason="flank_not_covered", true_tract=None)
    case = dataclasses.replace(case, ref_end=case.ref_start + 10)
    result = _place_reads(case, orientation)
    assert set(result.upstream_drop) == {"flank_not_covered"}
    assert set(result.downstream_drop) == {"flank_not_covered"}


def test_a_crossing_has_no_side_reason():
    case = next(c for c in CASES if c.case_id == "tract_and_edge_deleted")
    result = _place(case, "+")
    crossed = np.flatnonzero(result.flanks_crossed)
    assert crossed.size
    for i in crossed:
        assert result.upstream_drop[i] is None and result.downstream_drop[i] is None
        assert result.upstream_path[i] is not None and result.downstream_path[i] is not None


# ---------------------------------------------------------------------------
# A masked flank position on each path
# ---------------------------------------------------------------------------

# PolyG151 upstream flank ATATTTAGAT: the donor has a G at offset 2.
_UP151 = _up("PolyG151")
_MASKED_G = _sub(_UP151, len(_UP151) - 8, "G")
MASKED_CASES = {
    "fast": _case("masked_fast", "PolyG151: G at masked offset 2", "PolyG151",
                  up_masked=(2,), up_ctx=_MASKED_G, tract=G * 12),
    "cigar": _case("masked_cigar", "PolyG151: G at masked offset 2, one T of TTT deleted",
                   "PolyG151", up_masked=(2,), up_ctx=_del(_MASKED_G, len(_UP151) - 6),
                   tract=G * 12),
    "local": _case("masked_local", "PolyG151: G at masked offset 2 and a mismatch",
                   "PolyG151", up_masked=(2,),
                   up_ctx=_sub(_MASKED_G, len(_UP151) - 4, "C"), tract=G * 12),
}


@pytest.mark.parametrize("path", sorted(MASKED_CASES))
@pytest.mark.parametrize("orientation", ["+", "-"])
def test_masked_position_on_each_path(path, orientation):
    case = MASKED_CASES[path]
    result = _place_reads(case, orientation)
    assert result.kept.all()
    assert {run_tuple(t) for t in result.tracts} == {(12,)}
    forms = [p.form for p in placements(case)]
    form = {"fast": "gapless", "cigar": "gap_shift",
            "local": "gapless"}[path]
    on_form = {result.upstream_path[i] for i, f in enumerate(forms) if f == form}
    assert on_form == {path}
    # Without the mask the G is one more edit: the fast read takes the
    # window search, and the other two lose the read.
    unmasked = dataclasses.replace(case, up=dataclasses.replace(case.up, masked_offsets=()))
    plain = _place_reads(unmasked, orientation)
    i = forms.index(form)
    if path == "fast":
        assert plain.upstream_path[i] == "local"
    else:
        assert plain.upstream_drop[i] == "flank_not_placed"


# ---------------------------------------------------------------------------
# The one-M arithmetic path against the CIGAR walk
# ---------------------------------------------------------------------------


def _split_m(cigar):
    """The same alignment, written as more than one CIGAR operation."""
    (op, length), = cigar
    if length < 3:
        return [(5, 2), (op, length)]
    return [(7, 1), (8, length - 2), (0, 1)]


@pytest.mark.parametrize("orientation", ["+", "-"])
def test_one_m_arithmetic_matches_the_cigar_walk(orientation):
    rng = random.Random(5)
    seqs, cigars, starts, split = [], [], [], []
    case = next(c for c in CASES if c.case_id == "edge_variant_hom")
    chrom = chromosome(case.locus, orientation)
    full = chrom
    # Every read start and length over the chromosome, with read errors:
    # projections inside the read, past its ends, and on both flanks.
    for _ in range(3000):
        start = rng.randrange(0, len(full) - 5)
        length = rng.randrange(5, len(full) - start + 1)
        read = list(full[start:start + length])
        for _ in range(rng.randrange(3)):
            read[rng.randrange(length)] = rng.choice("ACGTNacgtn")
        seqs.append("".join(read))
        cigars.append([(rng.choice((0, 7, 8)), length)])
        starts.append(start)
        split.append(_split_m(cigars[-1]))
    kw = dict(upstream=_flank_record(case, case.up, orientation),
              downstream=_flank_record(case, case.dn, orientation),
              ref_orientation=orientation, reference_window=chrom,
              reference_window_start=0)
    one = place_flanks(seqs, cigars, np.array(starts), **kw)
    walked = place_flanks(seqs, split, np.array(starts), **kw)
    for field in dataclasses.fields(FlankPlacement):
        assert list(getattr(one, field.name)) == list(getattr(walked, field.name)), field.name
    assert one.kept.any() and (~one.kept).any()
    assert set(one.upstream_path) >= {"fast", "local", None}


# ---------------------------------------------------------------------------
# The CIGAR walk over the block against a walk of one read at a time
# ---------------------------------------------------------------------------


def _reference_events(cigar, start, lo, hi):
    """Indel and soft clip over reference ``[lo, hi)``, by a plain walk of one CIGAR."""
    p = start
    indel = clip = False
    last = len(cigar) - 1
    for k, (op, length) in enumerate(cigar):
        if op in (0, 7, 8):
            p += length
        elif op in (2, 3):
            if lo < p + length and p < hi:
                indel = True
            p += length
        elif op == 1:
            if lo < p < hi:
                indel = True
        elif op == 4:
            if (k == 0 and p > lo) or (k == last and p < hi):
                clip = True
    return indel, clip


def _random_alignments(rng, chrom, n):
    """Reads with random CIGARs: M, =, X, I, D, N, P, soft and hard clips."""
    query_ops = {0, 1, 4, 6, 7, 8}
    seqs, cigars, starts = [], [], []
    for _ in range(n):
        ops = []
        if rng.random() < 0.3:
            ops.append((5, rng.randrange(1, 4)))
        if rng.random() < 0.3:
            ops.append((4, rng.randrange(1, 8)))
        for _ in range(rng.randrange(1, 5)):
            ops.append((rng.choice([0, 0, 0, 7, 8]), rng.randrange(1, 30)))
            r = rng.random()
            if r < 0.3:
                ops.append((1, rng.randrange(1, 4)))
            elif r < 0.6:
                ops.append((2, rng.randrange(1, 4)))
            elif r < 0.65:
                ops.append((3, rng.randrange(1, 4)))
            elif r < 0.7:
                ops.append((6, rng.randrange(1, 3)))
        ops.append((0, rng.randrange(1, 30)))
        if rng.random() < 0.3:
            ops.append((4, rng.randrange(1, 8)))
        if rng.random() < 0.2:
            ops.append((5, rng.randrange(1, 4)))
        start = rng.randrange(0, len(chrom) - 20)
        length = sum(n_bases for op, n_bases in ops if op in query_ops)
        bases = chrom[start:start + length].ljust(length, "G")
        seqs.append("".join(c if rng.random() > 0.1 else rng.choice("ACGTG") for c in bases))
        cigars.append(ops)
        starts.append(start)
    return seqs, cigars, np.array(starts)


@pytest.mark.parametrize("orientation", ["+", "-"])
def test_block_walk_matches_one_read_at_a_time(orientation):
    rng = random.Random(23)
    case = next(c for c in CASES if c.case_id == "edge_variant_hom")
    chrom = chromosome(case.locus, orientation)
    polyg_reference = chrom if orientation == "+" else revcomp(chrom)
    seqs, cigars, starts = _random_alignments(rng, chrom, 2000)
    positions = [CHROM_PAD + 25, CHROM_PAD + 46]
    intervals = [(CHROM_PAD + 16, CHROM_PAD + 26), (CHROM_PAD + 46, CHROM_PAD + 56)]
    middle = 63.5

    matrix, lengths = flank_placement._polyg_matrix(seqs, orientation == "-")
    aln = flank_placement._polyg_cigar_arrays(cigars, starts, ref_orientation=orientation,
                                              window_start=0, window_len=len(chrom))
    flank_placement._shift_block_indels(aln, matrix, lengths, polyg_reference, middle)
    projection, on_m, indel, clip = flank_placement._alignment_geometry(
        aln, lengths, positions, intervals)

    reads, _, _ = flank_placement.polyg_alignments(
        seqs, cigars, starts, ref_orientation=orientation, window_start=0,
        window_len=len(chrom))
    n_single = 0
    for i in range(len(seqs)):
        cig, start = flank_placement._polyg_cigar(
            cigars[i], int(starts[i]), ref_orientation=orientation, window_start=0,
            window_len=len(chrom))
        assert start == aln.start[i]
        n_single += sum(op in (1, 2) for op, _ in cig) == 1
        flank_placement._shift_indels(cig, start, reads[i], polyg_reference, middle)
        n_ops = int(aln.n_ops[i])
        assert [list(x) for x in zip(aln.ops[i, :n_ops].tolist(),
                                     aln.lengths[i, :n_ops].tolist(), strict=True)] == cig
        for s, position in enumerate(positions):
            assert (projection[s, i], on_m[s, i]) == flank_placement._project(
                cig, start, position, len(reads[i]))
        for s, (lo, hi) in enumerate(intervals):
            assert (indel[s, i], clip[s, i]) == _reference_events(cig, start, lo, hi)
    assert n_single > 100


# ---------------------------------------------------------------------------
# Reads where one mate's tract gains a base (PolyG8057, donor A's normal)
# ---------------------------------------------------------------------------

STUTTER_CASES = run_stutter_with_flank_error_cases()


def _accepted_outcomes(case: ReadCase) -> set:
    """The mate's tuple; a read with a G at a run-side base may also drop and count."""
    outcomes = {case.true_tuple}
    if "g_at" in case.case_id:
        outcomes.add("g_at_run_side")
    return outcomes


@pytest.mark.parametrize("case", STUTTER_CASES, ids=[c.case_id for c in STUTTER_CASES])
@pytest.mark.parametrize("orientation", ["+", "-"])
def test_run_stutter_with_a_flank_error_gives_the_mate_tuple(case: ReadCase, orientation):
    result = _place(case, orientation)
    for i, form in enumerate(p.form for p in placements(case)):
        if result.kept[i]:
            got = run_tuple(result.tracts[i])
        else:
            reasons = drop_reasons(result, i)
            got = reasons.pop() if len(reasons) == 1 else tuple(sorted(reasons))
        assert got in _accepted_outcomes(case), form
