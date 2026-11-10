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
    ReadCase,
    all_records,
    build_cases,
    chromosome,
    flank_edit_distance,
    placements,
    run_tuple,
    soft_clipped_flank_case,
)

CASES = build_cases() + [soft_clipped_flank_case()]
CASE_IDS = [c.case_id for c in CASES]

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
    "post_run_insertion": {"fast", "cigar", "local"},
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
    "boundary_t_insertion": {"fast", "cigar", "local"},
    "tgg_flank_substitution": {"fast", "local"},
    "tandem_copy_error_in_near_copy": {"fast", "local"},
    "soft_clip_in_flank": {"fast", "local"},
}

# Reads that are ambiguous by construction. Each read string also comes,
# with no error, from a molecule with another tuple, so no rule that reads
# the string can recover the molecule the read was built from. The kernel
# puts the extra or missing base in the flank, and these are the tuples
# that gives.
ASSIGNED_TUPLE: dict[str, tuple[int, ...]] = {
    # Reference allele T T G10 with one T deleted reads as T G10, which is
    # also a molecule with the edge T deleted.
    "boundary_t_deletion": (10,),
    # G14 with a T inserted after the run reads as G14 T T..., which is also
    # a molecule with a T interrupter at the end of the tract.
    "boundary_t_insertion": (14,),
}


# ---------------------------------------------------------------------------
# The fixtures themselves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
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


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
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
# The kernel (expected to fail until it exists)
# ---------------------------------------------------------------------------

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
        [r.query_qualities for r in records],
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
            assert result.drop_reason[i] == case.drop_reason, form
            continue
        assert result.drop_reason[i] is None, form
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
    assert list(plus.drop_reason) == list(minus.drop_reason)


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
            want = [min(flank_edit_distance(flank, r, b, upstream=upstream), 2)
                    for r, b in rows]
            assert list(np.minimum(got, 2)) == want


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
            want = [min(flank_edit_distance(flank, r, int(b), upstream=upstream,
                                            masked=masked), 2)
                    for r, b in zip(reads, run_side, strict=True)]
            assert list(np.minimum(got, 2)) == want


def test_every_path_is_used_on_both_sides():
    used = {"upstream": set(), "downstream": set()}
    for case in CASES:
        for orientation in "+-":
            result = _place(case, orientation)
            used["upstream"] |= set(result.upstream_path) - {None}
            used["downstream"] |= set(result.downstream_path) - {None}
    assert used["upstream"] == {"fast", "cigar", "local"}
    assert used["downstream"] == {"fast", "cigar", "local"}


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


def test_tract_qualities_follow_the_tract():
    case = next(c for c in CASES if c.case_id == "invariant_side")
    for orientation in "+-":
        records = [rec for _, rec in all_records(case, orientation)]
        quals = [np.arange(len(r.query_sequence), dtype=np.uint8) for r in records]
        result = place_flanks(
            [r.query_sequence for r in records],
            [r.cigartuples for r in records],
            np.array([r.reference_start for r in records], dtype=np.int64),
            quals,
            upstream=_flank_record(case, case.up, orientation),
            downstream=_flank_record(case, case.dn, orientation),
            ref_orientation=orientation,
            reference_window=chromosome(case.locus, orientation),
            reference_window_start=0,
        )
        q = quals[0] if orientation == "+" else quals[0][::-1]
        lo = int(result.upstream_run_side[0]) + 1
        assert list(result.tract_qualities[0]) == list(q[lo:lo + len(case.true_tract)])


def test_loci_table_is_polyg_oriented():
    """Every locus reads as Gs between its edge bases."""
    for locus in LOCI.values():
        assert set(locus.tract) == {"G"}
        assert locus.upstream[-1] != "G"
        assert locus.downstream[0] != "G"
