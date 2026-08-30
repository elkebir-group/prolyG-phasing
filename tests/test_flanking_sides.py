"""Tests for the flanking-side helpers in prolyg_phasing.io.bam.

A flanking side is a ``(start_distance, bases)`` run. The contract it
has to keep is the per-aligned-position one:

    for every aligned (query, reference) position pair of the alignment
    whose base quality clears ``min_base_q``, the base at that reference
    position appears at its canonical distance from the anchor; every
    other distance reads ``"."``.

`reference_side` below states exactly that, walking pysam's own
``get_aligned_pairs`` — an independent walk of the alignment, not the
block walk under test. The extraction fixture in ``test_io_bam.py``
builds all-M alignments only, so the CIGAR cases that actually stress
the block walk (insertions, deletions, soft and hard clips, reference
skips, padding) are covered here.
"""

from __future__ import annotations

import random

import pysam
import pytest

from prolyg_phasing.io.bam import (
    FLANKING_SEQ_LEN,
    _alignment_flanking,
    _assemble_side,
    _consensus,
    _merge_pair_flanking,
    _strip_side,
)

COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")

# Every CIGAR op, by number: M I D N S H P = X.
ALL_OPS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# Ops that advance pysam's query cursor in get_aligned_pairs. P is here
# because pysam advances across it, which is the behavior this package
# matches.
QUERY_CONSUMING = {0, 1, 4, 6, 7, 8}


@pytest.fixture
def header():
    return pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": "synth", "LN": 100000}]}
    )


def make_alignment(header, ref_start, cigar, seq=None, quality_str=None):
    """A synthetic alignment with the given CIGAR, sized to match it."""
    query_len = sum(ln for op, ln in cigar if op in QUERY_CONSUMING)
    seg = pysam.AlignedSegment(header)
    seg.query_name = "q"
    seg.query_sequence = seq if seq is not None else "ACGT" * (query_len // 4 + 1)
    seg.query_sequence = seg.query_sequence[:query_len]
    seg.query_qualities = pysam.qualitystring_to_array(
        quality_str if quality_str is not None else "I" * query_len
    )
    seg.flag = 0
    seg.reference_id = 0
    seg.reference_start = ref_start
    seg.mapping_quality = 60
    seg.cigartuples = cigar
    return seg


def side_to_map(side) -> dict[int, str]:
    """``{distance: base}`` for the covered distances of a side."""
    start, bases = side
    return {start + i: c for i, c in enumerate(bases) if c != "."}


def reference_side(r, *, bed_start, bed_end, g_walk_up, g_walk_dn, anchor_k,
                   ref_orientation, min_base_q):
    """The per-position contract, walked with pysam's get_aligned_pairs."""
    seq = r.query_sequence.upper()
    quals = r.query_qualities
    up: dict[int, str] = {}
    dn: dict[int, str] = {}
    if ref_orientation == "+":
        up_bound = bed_start - g_walk_up - anchor_k
        dn_bound = bed_end + g_walk_dn + anchor_k
    else:
        up_bound = bed_end + g_walk_up + anchor_k
        dn_bound = bed_start - g_walk_dn - anchor_k
    for qpos, refpos in r.get_aligned_pairs(with_seq=False):
        if qpos is None or refpos is None:
            continue
        if quals[qpos] < min_base_q:
            continue
        base = seq[qpos] if ref_orientation == "+" else seq[qpos].translate(COMP)
        if ref_orientation == "+":
            if refpos < up_bound:
                up[up_bound - 1 - refpos] = base
            elif refpos >= dn_bound:
                dn[refpos - dn_bound] = base
        else:
            if refpos >= up_bound:
                up[refpos - up_bound] = base
            elif refpos < dn_bound:
                dn[dn_bound - 1 - refpos] = base
    return up, dn


def reference_consensus(a: dict[int, str], b: dict[int, str]) -> dict[int, str]:
    """Agree keeps, disagree drops, covered-by-one keeps."""
    out = {}
    for pos in set(a) | set(b):
        ba, bb = a.get(pos), b.get(pos)
        if ba is None:
            out[pos] = bb
        elif bb is None:
            out[pos] = ba
        elif ba == bb:
            out[pos] = ba
    return out


# ---------------------------------------------------------------------------
# Side normalization
# ---------------------------------------------------------------------------


def test_strip_side_drops_leading_and_trailing_dots():
    assert _strip_side(3, "..AC.G..") == (5, "AC.G")


def test_strip_side_all_dots_is_the_empty_side():
    assert _strip_side(7, "....") == (0, "")


def test_assemble_side_fills_gaps_between_runs():
    # Two runs separated by a two-distance gap the read does not cover.
    assert _assemble_side(
        [(0, "AC"), (4, "GT")], outward_first=False, may_have_dots=True
    ) == (0, "AC..GT")


def test_assemble_side_reverses_run_order_for_the_outward_first_side():
    # The upstream side's runs arrive in decreasing distance order.
    assert _assemble_side(
        [(4, "GT"), (0, "AC")], outward_first=True, may_have_dots=True
    ) == (0, "AC..GT")


def test_assemble_side_empty_run_is_the_empty_side():
    assert _assemble_side([(9, "")], outward_first=False, may_have_dots=False) == (0, "")


# ---------------------------------------------------------------------------
# _alignment_flanking against the per-position contract
# ---------------------------------------------------------------------------


CIGAR_CASES = [
    ("all match", [(0, 60)]),
    ("soft clip both ends", [(4, 5), (0, 50), (4, 5)]),
    ("hard clip both ends", [(5, 5), (0, 60), (5, 5)]),
    ("insertion mid-read", [(0, 25), (1, 4), (0, 31)]),
    ("deletion mid-read", [(0, 25), (2, 6), (0, 35)]),
    ("reference skip mid-read", [(0, 25), (3, 30), (0, 35)]),
    ("padding mid-read", [(0, 25), (6, 3), (0, 32)]),
    ("equal and diff ops", [(7, 20), (8, 5), (7, 35)]),
    ("deletion straddling the upstream bound", [(0, 8), (2, 5), (0, 52)]),
    ("clip, insert, delete together", [(4, 3), (0, 20), (1, 2), (0, 15), (2, 4), (0, 20)]),
]


@pytest.mark.parametrize("label,cigar", CIGAR_CASES, ids=[c[0] for c in CIGAR_CASES])
@pytest.mark.parametrize("ref_orientation", ["+", "-"])
def test_alignment_flanking_matches_per_position_contract(
    header, label, cigar, ref_orientation
):
    bed_start, bed_end = 300, 314
    # Start the alignment well upstream so it straddles the bed and both
    # anchors, and so both sides get covered.
    seg = make_alignment(header, 260, cigar)
    kw = dict(bed_start=bed_start, bed_end=bed_end, g_walk_up=2, g_walk_dn=1,
              anchor_k=FLANKING_SEQ_LEN, ref_orientation=ref_orientation, min_base_q=20)
    up, dn = _alignment_flanking(seg, **kw)
    assert (side_to_map(up), side_to_map(dn)) == reference_side(seg, **kw)


@pytest.mark.parametrize("ref_orientation", ["+", "-"])
def test_alignment_flanking_masks_sub_threshold_bases(header, ref_orientation):
    cigar = [(0, 60)]
    # Q40 everywhere except a run of Q2 in the middle of the upstream side.
    quality = list("I" * 60)
    for i in range(4, 9):
        quality[i] = "#"
    seg = make_alignment(header, 260, cigar, quality_str="".join(quality))
    kw = dict(bed_start=300, bed_end=314, g_walk_up=2, g_walk_dn=1,
              anchor_k=FLANKING_SEQ_LEN, ref_orientation=ref_orientation, min_base_q=20)
    up, dn = _alignment_flanking(seg, **kw)
    assert (side_to_map(up), side_to_map(dn)) == reference_side(seg, **kw)
    # The masked positions really are absent, not merely re-ordered.
    assert sum(c == "." for c in up[1] + dn[1]) == 5


def test_alignment_flanking_without_cigar_is_two_empty_sides(header):
    seg = pysam.AlignedSegment(header)
    seg.query_name = "q"
    seg.query_sequence = "ACGT" * 15
    seg.query_qualities = pysam.qualitystring_to_array("I" * 60)
    seg.flag = 0
    seg.reference_id = 0
    seg.reference_start = 260
    seg.mapping_quality = 60
    up, dn = _alignment_flanking(
        seg, bed_start=300, bed_end=314, g_walk_up=0, g_walk_dn=0,
        anchor_k=FLANKING_SEQ_LEN, ref_orientation="+", min_base_q=20,
    )
    assert (up, dn) == ((0, ""), (0, ""))


@pytest.mark.parametrize("seed", range(8))
def test_alignment_flanking_matches_contract_on_random_cigars(header, seed):
    rng = random.Random(seed)
    for _ in range(300):
        cigar = [(rng.choice(ALL_OPS), rng.randint(1, 12)) for _ in range(rng.randint(1, 6))]
        if not any(op in QUERY_CONSUMING for op, _ in cigar):
            cigar.append((0, 8))
        query_len = sum(ln for op, ln in cigar if op in QUERY_CONSUMING)
        seg = make_alignment(
            header,
            rng.randint(240, 360),
            cigar,
            seq="".join(rng.choice("ACGTNacgtn") for _ in range(query_len)),
            quality_str="".join(chr(33 + rng.randint(0, 41)) for _ in range(query_len)),
        )
        kw = dict(bed_start=300, bed_end=300 + rng.randint(1, 20),
                  g_walk_up=rng.randint(0, 5), g_walk_dn=rng.randint(0, 5),
                  anchor_k=FLANKING_SEQ_LEN,
                  ref_orientation=rng.choice(["+", "-"]),
                  min_base_q=rng.choice([0, 20, 40]))
        up, dn = _alignment_flanking(seg, **kw)
        assert (side_to_map(up), side_to_map(dn)) == reference_side(seg, **kw)


# ---------------------------------------------------------------------------
# _consensus / _merge_pair_flanking
# ---------------------------------------------------------------------------


def test_consensus_keeps_agreement_and_drops_disagreement():
    assert _consensus((0, "ACGT"), (0, "ACTT")) == (0, "AC.T")


def test_consensus_keeps_a_position_only_one_side_covers():
    assert _consensus((0, "AC"), (2, "GT")) == (0, "ACGT")


def test_consensus_gaps_between_disjoint_sides():
    assert _consensus((0, "AC"), (5, "GT")) == (0, "AC...GT")


def test_consensus_with_an_empty_side_returns_the_other():
    assert _consensus((0, ""), (3, "ACGT")) == (3, "ACGT")
    assert _consensus((3, "ACGT"), (0, "")) == (3, "ACGT")


def test_consensus_disagreement_at_the_edge_shrinks_the_extent():
    # Both sides start at 0; they disagree there, so the merged side
    # must not claim distance 0 as covered.
    assert _consensus((0, "AC"), (0, "GC")) == (1, "C")


def test_consensus_is_symmetric_and_matches_the_dict_contract():
    rng = random.Random(31)
    for _ in range(5000):
        sides = []
        for _ in range(2):
            start = rng.randint(0, 8)
            text = "".join(rng.choice("ACGT.") for _ in range(rng.randint(0, 10)))
            sides.append(_strip_side(start, text))
        forward = _consensus(sides[0], sides[1])
        backward = _consensus(sides[1], sides[0])
        assert forward == backward
        assert side_to_map(forward) == reference_consensus(
            side_to_map(sides[0]), side_to_map(sides[1])
        )
        # A side is always normalized: no leading or trailing ".".
        assert not forward[1].startswith(".") and not forward[1].endswith(".")


def test_merge_pair_flanking_passes_a_singleton_through():
    mate = ((1, "AC"), (0, "GT"))
    assert _merge_pair_flanking([mate]) == mate


def test_merge_pair_flanking_of_nothing_is_two_empty_sides():
    assert _merge_pair_flanking([]) == ((0, ""), (0, ""))
