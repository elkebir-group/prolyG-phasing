"""The flank match, and each read's tract under one pair of flanks.

A locus has one flank on each side of the repeat: a 10-base string ``u``
upstream and ``v`` downstream, in polyG orientation. The matched normal's
reads choose the pair (see :mod:`prolyg_phasing.io._flank_construction`).
This module states what the pair does to one read. The tract depends only
on the read sequence, not on the alignment. The search for the pair and
the extraction use this one definition. The alignments give only the
reference coordinates of each flank (`FlankRecord.ref_positions`).

**Match.** A flank matches a read substring if the substring differs from
it by nothing, by one substitution, or by one gap of 1 or 2 bases. A gap
lies between two inner bases of the flank, so the first and last flank
bases align to the first and last substring bases. A gap and a
substitution are never allowed together. A read N differs from every
flank base.

**Placement.** A match of ``u`` whose substring ends at read position
``a``, and a match of ``v`` whose substring starts at a later position
``b``, where ``read[a]`` and ``read[b]`` are not G and ``read[a + 1]`` and
``read[b - 1]`` are G. The tract is ``read[a + 1:b]``; it starts and ends
with G. Only positions with that shape are tract edges, so a match is
looked for only there. The search covers the whole read.

**Value.** All values are in half units, so every value is an integer. A
G in the tract scores +2, any other base -4. A substitution costs 2, and a
gap of ``g`` bases costs ``3 + g``. The value of a placement is the tract
score minus the two match costs. In the units of the decision record this
is #G - 2 #nonG minus match costs of 1, 2 or 2.5.

**A read's tract** is the tract of its highest-value placement. When
several placements have the top value, the read keeps the shortest tract:
the largest ``a`` and, among those, the smallest ``b``. A piece that does
not raise the value does not join the tract. At a flank of a repeated
unit such as TGG, each extra GGT adds 0, so this rule keeps those copies
in the flank. A read with no placement, or whose best value is negative,
is dropped and counted.
"""

from __future__ import annotations

import csv
import dataclasses
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from prolyg_phasing.io._anchors import COMP

FLANK_LEN = 10

# The longest gap a match allows, in bases.
MAX_GAP = 2

# Half-unit scores: a tract G, any other tract base, one substitution, and
# the fixed part of a gap (a gap of g bases costs GAP_OPEN + g).
G_SCORE, NON_G_SCORE = 2, -4
SUBSTITUTION_COST = 2
GAP_OPEN = 3

# Shortest and longest read substring a flank can match.
MIN_MATCH_LEN = FLANK_LEN - MAX_GAP
MAX_MATCH_LEN = FLANK_LEN + MAX_GAP

# Base codes: A, C, G, T are 0 to 3. N stands for every other symbol. PAD marks a
# position outside the read.
G, N, PAD = 2, 4, 5
BASE_CODES = np.full(256, N, dtype=np.uint8)
for _code, _base in enumerate("ACGT"):
    BASE_CODES[ord(_base)] = BASE_CODES[ord(_base.lower())] = _code

# 2-bit place values of the 10 flank bases.
FLANK_PLACE_VALUES = 4 ** np.arange(FLANK_LEN - 1, -1, -1, dtype=np.int64)

# 3-bit place values of a window of `MAX_MATCH_LEN` base codes.
WINDOW_PLACE_VALUES = 8 ** np.arange(MAX_MATCH_LEN - 1, -1, -1, dtype=np.int64)

# Sentinel cost of a window the flank does not match.
NO_MATCH = -1


def flank_codes(flank: str) -> np.ndarray:
    """The base codes of a 10-base A/C/G/T flank."""
    codes = BASE_CODES[np.frombuffer(flank.encode("ascii"), np.uint8)]
    if len(codes) != FLANK_LEN or (codes == N).any():
        raise ValueError(f"a flank is {FLANK_LEN} bases of A, C, G, T, not {flank!r}")
    return codes


def decode(code: int) -> str:
    """The 10-base string of a 2-bit code."""
    return "".join("ACGT"[(code >> (2 * (FLANK_LEN - 1 - j))) & 3] for j in range(FLANK_LEN))


# ---------------------------------------------------------------------------
# Match shapes
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MatchShape:
    """How the 10 flank bases align to a read substring of `length` bases.

    `kind` is "none" (no gap), "insertion" (`gap` extra read bases before
    flank base `at`) or "deletion" (flank bases ``at .. at + gap - 1`` are
    missing from the read). `read_slots` and `flank_slots` are the aligned
    substring and flank indices, in order.
    """

    length: int
    gap: int
    at: int
    kind: str
    read_slots: tuple[int, ...]
    flank_slots: tuple[int, ...]

    @property
    def cost(self) -> int:
        """The cost of the shape with no substitution."""
        return 0 if self.kind == "none" else GAP_OPEN + self.gap


def _shapes() -> tuple[MatchShape, ...]:
    every = tuple(range(FLANK_LEN))
    out = [MatchShape(FLANK_LEN, 0, 0, "none", every, every)]
    for g in range(1, MAX_GAP + 1):
        for i in range(1, FLANK_LEN):
            slots = tuple(range(i)) + tuple(range(i + g, FLANK_LEN + g))
            out.append(MatchShape(FLANK_LEN + g, g, i, "insertion", slots, every))
        for i in range(1, FLANK_LEN - g):
            kept = tuple(range(i)) + tuple(range(i + g, FLANK_LEN))
            out.append(MatchShape(FLANK_LEN - g, g, i, "deletion",
                                  tuple(range(FLANK_LEN - g)), kept))
    return tuple(out)


# Every alignment shape of the match, the gapless one first.
SHAPES = _shapes()


def window_costs(windows: np.ndarray, flank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lowest match cost of `flank` against the substring that ends each window.

    Parameters
    ----------
    windows : ndarray of uint8, shape (n, MAX_MATCH_LEN)
        Base codes; the last column is the last base of the substring. PAD
        marks a position outside the read.
    flank : ndarray of uint8, shape (FLANK_LEN,)
        The flank's base codes, in the same orientation as the windows.

    Returns
    -------
    cost : ndarray of int64
        The lowest cost, or `NO_MATCH`.
    length : ndarray of int64
        The substring length of the lowest-cost match: the first such
        shape in `SHAPES` order. 0 where there is no match.
    """
    n = len(windows)
    cost = np.full(n, NO_MATCH, dtype=np.int64)
    length = np.zeros(n, dtype=np.int64)
    for shape in SHAPES:
        sub = windows[:, MAX_MATCH_LEN - shape.length:]
        inside = (sub != PAD).all(axis=1)
        mismatches = (sub[:, list(shape.read_slots)]
                      != flank[list(shape.flank_slots)]).sum(axis=1)
        if shape.kind == "none":
            ok = inside & (mismatches <= 1)
            c = SUBSTITUTION_COST * mismatches.astype(np.int64)
        else:
            ok = inside & (mismatches == 0)
            c = np.full(n, shape.cost, dtype=np.int64)
        better = ok & ((cost == NO_MATCH) | (c < cost))
        cost[better] = c[better]
        length[better] = shape.length
    return cost, length


# ---------------------------------------------------------------------------
# Reads as code matrices
# ---------------------------------------------------------------------------


def code_matrix(reads: Sequence[str], *, reverse: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Reads as an (n, width) base-code matrix padded with PAD, and their lengths.

    With `reverse`, each read is reversed within its own length.
    """
    lengths = np.fromiter(map(len, reads), dtype=np.int64, count=len(reads))
    width = max(int(lengths.max()) if len(reads) else 0, 1)
    if reverse:
        reads = [s[::-1] for s in reads]
    flat = "".join(s.ljust(width, "\0") for s in reads).encode("ascii")
    matrix = BASE_CODES[np.frombuffer(flat, dtype=np.uint8)].reshape(len(reads), width)
    matrix[np.arange(width)[None, :] >= lengths[:, None]] = PAD
    return matrix, lengths


def tract_edges(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Read and position of every upstream tract edge: a non-G base before a G."""
    edge = np.zeros(matrix.shape, dtype=bool)
    edge[:, :-1] = (matrix[:, :-1] != G) & (matrix[:, :-1] != PAD) & (matrix[:, 1:] == G)
    return np.nonzero(edge)


def edge_windows(matrix: np.ndarray, rows: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """The `MAX_MATCH_LEN` bases that end at each (row, end), PAD before the read."""
    cols = ends[:, None] + np.arange(MAX_MATCH_LEN)[None, :] - (MAX_MATCH_LEN - 1)
    return np.where(cols >= 0, matrix[rows[:, None], np.maximum(cols, 0)], PAD).astype(np.uint8)


def _side_matches(matrix: np.ndarray, flank: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(read, edge position, cost) of every match of `flank` at an upstream tract edge.

    The cost is computed once per distinct window, not once per position.
    """
    rows, ends = tract_edges(matrix)
    if rows.size == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    windows = edge_windows(matrix, rows, ends)
    _, first, inverse = np.unique(windows.astype(np.int64) @ WINDOW_PLACE_VALUES,
                                  return_index=True, return_inverse=True)
    cost = window_costs(windows[first], flank)[0][inverse.ravel()]
    hit = cost != NO_MATCH
    return rows[hit].astype(np.int64), ends[hit].astype(np.int64), cost[hit]


def flank_matches(reads: Sequence[str], upstream: str, downstream: str
                  ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Every match of each flank at a tract edge of each read.

    Returns ``((r, a, cost), (r, b, cost))``: the upstream flank's matches
    by the position ``a`` its substring ends at, and the downstream
    flank's by the position ``b`` its substring starts at. Reads are in
    polyG orientation.
    """
    forward, _ = code_matrix(reads)
    up = _side_matches(forward, flank_codes(upstream))
    backward, lengths = code_matrix(reads, reverse=True)
    r, a_rev, cost = _side_matches(backward, flank_codes(downstream)[::-1].copy())
    return up, (r, lengths[r] - 1 - a_rev, cost)


# ---------------------------------------------------------------------------
# Best placement per read
# ---------------------------------------------------------------------------

# Larger than any value a read can reach, and small enough that sums of two
# stay exact in int64.
_FAR = 1 << 40

# Rows of the dense per-read matrices handled at once.
_PLACEMENT_BLOCK = 4096


def tract_scores(matrix: np.ndarray) -> np.ndarray:
    """Cumulative tract score: ``out[r, j]`` is the score of ``read[r][:j]``."""
    score = np.where(matrix == G, G_SCORE, NON_G_SCORE).astype(np.int64)
    out = np.zeros((matrix.shape[0], matrix.shape[1] + 1), dtype=np.int64)
    np.cumsum(score, axis=1, out=out[:, 1:])
    return out


@dataclasses.dataclass(frozen=True)
class BestPlacements:
    """Each read's highest-value placement under one pair of flanks.

    Attributes
    ----------
    placed : ndarray of bool
        The read has at least one placement.
    value : ndarray of int64
        The best value in half units; 0 where the read has no placement.
    upstream_end, downstream_start : ndarray of int64
        ``a`` and ``b`` of the best placement, or -1. The tract is
        ``read[a + 1:b]``.
    """

    placed: np.ndarray
    value: np.ndarray
    upstream_end: np.ndarray
    downstream_start: np.ndarray


def best_placements(scores: np.ndarray, upstream: Sequence[np.ndarray],
                    downstream: Sequence[np.ndarray]) -> BestPlacements:
    """Each read's highest-value placement, from the match lists of two flanks.

    Parameters
    ----------
    scores : ndarray of int64, shape (n_reads, width + 1)
        :func:`tract_scores` of the reads.
    upstream, downstream : sequence of ndarray
        ``(r, a, cost)`` and ``(r, b, cost)`` as :func:`flank_matches`
        returns them. A position may repeat; its lowest cost counts.

    Notes
    -----
    With ``open_cost[a] = scores[a + 1] + cost_u(a)`` and
    ``close_value[b] = scores[b] - cost_v(b)``, the value of ``(a, b)`` is
    ``close_value[b] - open_cost[a]``. The best value is the largest
    ``close_value[b]`` minus the smallest ``open_cost[a]`` with ``a < b``.
    Among the placements with that value, the largest ``a`` wins, and then
    the smallest ``b``: the shortest tract.
    """
    n_reads, width = scores.shape[0], scores.shape[1] - 1
    placed = np.zeros(n_reads, dtype=bool)
    value = np.zeros(n_reads, dtype=np.int64)
    up_end = np.full(n_reads, -1, dtype=np.int64)
    dn_start = np.full(n_reads, -1, dtype=np.int64)
    ru, au, cu = (np.asarray(x, dtype=np.int64) for x in upstream)
    rv, bv, cv = (np.asarray(x, dtype=np.int64) for x in downstream)
    for lo in range(0, n_reads, _PLACEMENT_BLOCK):
        hi = min(n_reads, lo + _PLACEMENT_BLOCK)
        mu = (ru >= lo) & (ru < hi)
        mv = (rv >= lo) & (rv < hi)
        if not mu.any() or not mv.any():
            continue
        open_cost = np.full((hi - lo, width), _FAR, dtype=np.int64)
        np.minimum.at(open_cost, (ru[mu] - lo, au[mu]), scores[ru[mu], au[mu] + 1] + cu[mu])
        close_value = np.full((hi - lo, width), -_FAR, dtype=np.int64)
        np.maximum.at(close_value, (rv[mv] - lo, bv[mv]), scores[rv[mv], bv[mv]] - cv[mv])
        # The smallest open cost before each b, and the largest close value after each a.
        before = np.full_like(open_cost, _FAR)
        before[:, 1:] = np.minimum.accumulate(open_cost, axis=1)[:, :-1]
        after = np.full_like(close_value, -_FAR)
        after[:, :-1] = np.maximum.accumulate(close_value[:, ::-1], axis=1)[:, ::-1][:, 1:]
        ok_b = (close_value > -_FAR) & (before < _FAR)
        has = ok_b.any(axis=1)
        if not has.any():
            continue
        best = np.where(ok_b, close_value - before, -4 * _FAR).max(axis=1)
        ok_a = (open_cost < _FAR) & (after > -_FAR) & (after - open_cost == best[:, None])
        a = width - 1 - ok_a[:, ::-1].argmax(axis=1)
        open_at_a = open_cost[np.arange(hi - lo), a]
        cols = np.arange(width)[None, :]
        ok_ab = ((cols > a[:, None]) & (close_value > -_FAR)
                 & (close_value - open_at_a[:, None] == best[:, None]))
        b = ok_ab.argmax(axis=1)
        rows = np.arange(lo, hi)[has]
        placed[rows] = True
        value[rows] = best[has]
        up_end[rows] = a[has]
        dn_start[rows] = b[has]
    return BestPlacements(placed, value, up_end, dn_start)


# ---------------------------------------------------------------------------
# Flank records and the flank table
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FlankRecord:
    """One locus side's flank.

    Attributes
    ----------
    bases : str
        The flank, 10 bases in polyG orientation, in read order. It comes
        from the matched normal's reads and can differ from the reference.
    ref_positions : tuple of int
        The 0-based reference coordinates from the flank's outer base to
        its run-side base, contiguous and in polyG read order. The matched
        normal's read alignments give the two end coordinates (see
        :mod:`prolyg_phasing.io._flank_construction`).
    run_side_offset : int
        Offset in `ref_positions` of the base next to the repeat: the last
        offset for the upstream flank, 0 for the downstream flank.
    """

    bases: str
    ref_positions: tuple[int, ...]
    run_side_offset: int

    def __post_init__(self) -> None:
        if len(self.ref_positions) < 2:
            raise ValueError("a flank matches at least two reference bases")
        if self.run_side_offset not in (0, len(self.ref_positions) - 1):
            raise ValueError("the run-side base must be the first or the last matched base")

    @property
    def is_upstream(self) -> bool:
        return self.run_side_offset == len(self.ref_positions) - 1

    @property
    def run_side_position(self) -> int:
        """Reference coordinate of the matched base next to the repeat."""
        return self.ref_positions[self.run_side_offset]

    @property
    def outer_position(self) -> int:
        """Reference coordinate of the matched base farthest from the repeat."""
        return self.ref_positions[0 if self.is_upstream else -1]


# The reference window status of a locus side in the flank table and of a
# locus in the panel: a read rebuilt the window, or none did and the side
# has no flank.
WINDOW_RECOVERED = "recovered"
WINDOW_UNRECOVERABLE = "ref_window_unrecoverable"

# Side-table fields that a side with no reference window leaves empty.
_FLANK_FIELDS = ("flank_bases", "flank_ref_positions")


@dataclasses.dataclass(frozen=True)
class FlankTable:
    """The per-side flank table of one matched normal.

    Every BED locus is in exactly one of `records` and
    `no_reference_window`. A locus with no reference window has no flank
    record, so it cannot be placed by mistake.

    Attributes
    ----------
    sample : str
        The sample whose reads built the flanks.
    records : dict of str to (FlankRecord, FlankRecord)
        Upstream and downstream flank per locus with a reference window.
    no_reference_window : frozenset of str
        Loci whose reference window no read of the sample rebuilt.
    """

    sample: str
    records: dict[str, tuple[FlankRecord, FlankRecord]]
    no_reference_window: frozenset[str]

    @property
    def loci(self) -> set[str]:
        """Every locus of the table."""
        return set(self.records) | self.no_reference_window


def read_flank_table(path: str | Path) -> FlankTable:
    """The flank records and the sample of a side TSV that `build-flanks` writes.

    Raises ValueError when the table has no ``sample`` or
    ``reference_window_status`` column, names more than one sample, lacks
    one side of a locus, gives the two sides of a locus different
    statuses, or has flank fields on a side with no reference window.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for column in ("sample", "reference_window_status"):
            if column not in (reader.fieldnames or ()):
                raise ValueError(f"{path}: the flank table has no {column} column; "
                                 "rebuild it with build-flanks")
        rows = list(reader)
    samples = {row["sample"] for row in rows}
    if len(samples) != 1:
        raise ValueError(f"{path}: the flank table must name one sample, not "
                         f"{sorted(samples)}")
    by_locus: dict[str, dict[str, FlankRecord | None]] = {}
    for row in rows:
        status = row["reference_window_status"]
        if status == WINDOW_UNRECOVERABLE:
            if any(row[field] for field in _FLANK_FIELDS):
                raise ValueError(f"{path}: {row['locus']} {row['side']} has no reference "
                                 "window but carries flank fields")
            by_locus.setdefault(row["locus"], {})[row["side"]] = None
            continue
        if status != WINDOW_RECOVERED:
            raise ValueError(f"{path}: {row['locus']} {row['side']} has the unknown "
                             f"reference window status {status!r}")
        positions = tuple(int(p) for p in row["flank_ref_positions"].split(","))
        record = FlankRecord(
            bases=row["flank_bases"],
            ref_positions=positions,
            run_side_offset=len(positions) - 1 if row["side"] == "upstream" else 0,
        )
        by_locus.setdefault(row["locus"], {})[row["side"]] = record
    incomplete = sorted(locus for locus, pair in by_locus.items()
                        if set(pair) != {"upstream", "downstream"})
    if incomplete:
        raise ValueError(f"{path}: loci without both sides: {incomplete}")
    mixed = sorted(locus for locus, pair in by_locus.items()
                   if (pair["upstream"] is None) != (pair["downstream"] is None))
    if mixed:
        raise ValueError(f"{path}: loci whose two sides differ in reference window "
                         f"status: {mixed}")
    records = {locus: (pair["upstream"], pair["downstream"])
               for locus, pair in by_locus.items() if pair["upstream"] is not None}
    no_window = frozenset(locus for locus, pair in by_locus.items()
                          if pair["upstream"] is None)
    return FlankTable(sample=samples.pop(), records=records, no_reference_window=no_window)


# ---------------------------------------------------------------------------
# Window coordinates
# ---------------------------------------------------------------------------


def window_index(position: int, *, ref_orientation: str, window_start: int,
                 window_len: int) -> int:
    """Index of reference coordinate `position` in the polyG-orientation window.

    The window is the reference on ``[window_start, window_start +
    window_len)``, reverse-complemented at a "-" locus.
    """
    if ref_orientation == "-":
        return window_start + window_len - 1 - position
    return position - window_start


# ---------------------------------------------------------------------------
# Placement of a block of reads
# ---------------------------------------------------------------------------

# Byte table from a stored read base to its upper-case complement. With a
# reversal it gives the same bases as `revcomp` of the upper-cased read.
_COMPLEMENT_BYTES = bytearray(range(256))
for _base, _complement in zip("ACGTN", "ACGTN".translate(COMP), strict=True):
    _COMPLEMENT_BYTES[ord(_base)] = _COMPLEMENT_BYTES[ord(_base.lower())] = ord(_complement)
_COMPLEMENT_BYTES = bytes(_COMPLEMENT_BYTES)


def polyg_reads(read_seqs: Sequence[str], ref_orientation: str) -> list[str]:
    """Reads in polyG orientation, upper-case: reverse-complemented at a "-" locus."""
    if ref_orientation == "-":
        return [s.encode("ascii").translate(_COMPLEMENT_BYTES).decode("ascii").upper()[::-1]
                for s in read_seqs]
    return [s.upper() for s in read_seqs]


@dataclasses.dataclass(frozen=True)
class FlankPlacement:
    """Each read's tract under one pair of flanks, one entry per read.

    Attributes
    ----------
    tracts : ndarray of object
        The tract in polyG orientation for a kept read, else None.
    kept : ndarray of bool
        The read has a placement and its best value is not negative.
    placed : ndarray of bool
        The read has at least one placement, of any value.
    upstream_matched, downstream_matched : ndarray of bool
        The flank matches the read at some tract edge.
    no_placement : ndarray of bool
        Both flanks match the read, but no placement exists: no upstream
        match ends before a downstream match starts.
    negative_value : ndarray of bool
        The read's best placement has a negative value.
    value : ndarray of int64
        The best value in half units; 0 where the read has no placement.
    upstream_run_side, downstream_run_side : ndarray of int64
        ``a`` and ``b`` of the best placement, read indices in polyG
        orientation, or -1. The tract of a kept read is the read on
        ``[upstream_run_side + 1, downstream_run_side)``.
    """

    tracts: np.ndarray
    kept: np.ndarray
    placed: np.ndarray
    upstream_matched: np.ndarray
    downstream_matched: np.ndarray
    no_placement: np.ndarray
    negative_value: np.ndarray
    value: np.ndarray
    upstream_run_side: np.ndarray
    downstream_run_side: np.ndarray


def place_polyg_reads(reads: Sequence[str], upstream: str, downstream: str
                      ) -> FlankPlacement:
    """Each read's tract under the flanks `upstream` and `downstream`.

    Reads are in polyG orientation and upper-case. A read string is placed
    once, however often it occurs.
    """
    n = len(reads)
    tracts = np.full(n, None, dtype=object)
    if not n:
        no = np.zeros(0, dtype=bool)
        z = np.zeros(0, dtype=np.int64)
        return FlankPlacement(tracts, no, no, no, no, no, no, z, z, z)
    distinct, inverse = np.unique(np.asarray(reads, dtype=object), return_inverse=True)
    distinct = list(distinct)
    inverse = inverse.ravel()
    up, dn = flank_matches(distinct, upstream, downstream)
    best = best_placements(tract_scores(code_matrix(distinct)[0]), up, dn)
    up_any = np.zeros(len(distinct), dtype=bool)
    dn_any = np.zeros(len(distinct), dtype=bool)
    up_any[up[0]] = True
    dn_any[dn[0]] = True

    placed = best.placed[inverse]
    value = best.value[inverse]
    upstream_matched = up_any[inverse]
    downstream_matched = dn_any[inverse]
    negative = placed & (value < 0)
    kept = placed & ~negative
    a = best.upstream_end[inverse]
    b = best.downstream_start[inverse]
    rows = np.flatnonzero(kept).tolist()
    tracts[rows] = [reads[i][a[i] + 1:b[i]] for i in rows]
    return FlankPlacement(
        tracts=tracts,
        kept=kept,
        placed=placed,
        upstream_matched=upstream_matched,
        downstream_matched=downstream_matched,
        no_placement=upstream_matched & downstream_matched & ~placed,
        negative_value=negative,
        value=value,
        upstream_run_side=a,
        downstream_run_side=b,
    )


def place_flanks(read_seqs: Sequence[str], *, upstream: FlankRecord,
                 downstream: FlankRecord, ref_orientation: str) -> FlankPlacement:
    """Each read's tract under one locus's two flanks.

    Parameters
    ----------
    read_seqs : sequence of str
        Read sequences as stored in the BAM (reference orientation).
    upstream, downstream : FlankRecord
        The two flanks of the locus.
    ref_orientation : str
        "+" if the repeat reads as Gs on the reference, "-" otherwise.
    """
    return place_polyg_reads(polyg_reads(read_seqs, ref_orientation), upstream.bases,
                             downstream.bases)


# ---------------------------------------------------------------------------
# The flank's read bases at the best placement
# ---------------------------------------------------------------------------


def flank_outer_bases(reads: Sequence[str], placement: FlankPlacement, upstream: str,
                      downstream: str) -> tuple[np.ndarray, np.ndarray]:
    """Read index of each flank's outer base at each kept read's best placement.

    The upstream flank's substring ends at ``a`` and the downstream flank's
    substring starts at ``b`` (see :class:`FlankPlacement`). Its length is
    the length of the lowest-cost match there: 8 to 12 read bases. Of two
    shapes with the same cost, the first in `SHAPES` order gives the
    length, as in :func:`window_costs`.

    Parameters
    ----------
    reads : sequence of str
        The reads that `placement` placed, in polyG orientation.
    placement : FlankPlacement
        The placement of `reads` under `upstream` and `downstream`.
    upstream, downstream : str
        The two flanks.

    Returns
    -------
    upstream_outer, downstream_outer : ndarray of int64
        Read indices in polyG orientation of the base farthest from the
        repeat; -1 where the read is not kept.
    """
    n = len(reads)
    up_outer = np.full(n, -1, dtype=np.int64)
    dn_outer = np.full(n, -1, dtype=np.int64)
    rows = np.flatnonzero(placement.kept)
    if rows.size == 0:
        return up_outer, dn_outer
    kept_reads = [reads[i] for i in rows]
    local = np.arange(rows.size)
    a = placement.upstream_run_side[rows]
    b = placement.downstream_run_side[rows]
    forward, _ = code_matrix(kept_reads)
    _, up_len = window_costs(edge_windows(forward, local, a), flank_codes(upstream))
    backward, lengths = code_matrix(kept_reads, reverse=True)
    _, dn_len = window_costs(edge_windows(backward, local, lengths - 1 - b),
                             flank_codes(downstream)[::-1].copy())
    if (up_len == 0).any() or (dn_len == 0).any():
        raise RuntimeError("a kept read has no flank match at its best placement")
    up_outer[rows] = a - up_len + 1
    dn_outer[rows] = b + dn_len - 1
    return up_outer, dn_outer
