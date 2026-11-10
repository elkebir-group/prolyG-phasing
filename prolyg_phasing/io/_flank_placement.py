"""Per-read flank placement: find each flank in each read and cut out the tract.

A locus has one flank on each side of the repeat. A flank is 10 reference
bases. Its run-side base is the flank base next to the repeat; it is a
non-G base. The tract is the read sequence strictly between the two placed
run-side bases. All work is in polyG orientation: at a "-" locus the read
and its alignment are reverse-complemented first.

The rule that places one flank in one read:

1. Move each indel of the alignment away from the run, as far as the read
   and reference bases let it move without a change to the alignment. An
   indel upstream of the locus middle moves upstream; an indel downstream
   of it moves downstream. Then every alignment of one read string that
   differs only in where a gap sits gives the same result.
2. Project the flank's run-side base through that alignment onto the read.
   Into a soft clip, extend the projection linearly.
3. Score each read position within `_LOCAL_WINDOW` bases of the projection
   as a run-side position. The score is the number of flank edits there
   plus 1 if the position is not the projection. Flank edits are counted
   by a banded edit distance with band 1 over the other nine flank bases.
   A masked flank position matches any base. The run-side base itself is
   not counted. Only positions with at most one flank edit are kept.
4. The lowest score wins. On a tie, a non-G run-side base wins over a G.
   On a further tie, the position nearest the run wins.
5. If the winning run-side base is a G, the read is dropped as
   ``"g_at_run_side"``. If no position is kept, the read is dropped as
   ``"flank_not_covered"`` when the projected flank is not inside the read,
   and as ``"flank_not_placed"`` otherwise. If the downstream run-side base
   is not after the upstream one, the read is dropped as
   ``"flanks_crossed"``.

The projection costs nothing extra when it is exact, so the rule reports
three paths. "fast": the alignment covers the flank with M operations and
the read equals the flank at every unmasked position. "cigar": the
alignment has an indel inside the flank, and the projection wins. "local":
any other placement. A fast placement has score 0, and no other position
can score 0, so a fast read needs no window search.

A read that carries an extra or a missing copy of the run-side base next
to the run is the same string as a read from a molecule where that base
belongs to the tract. The rule puts the extra or missing base in the flank.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from prolyg_phasing.io._anchors import revcomp

# Read positions on each side of the projected run-side base that the
# window search scores. It must cover the largest change of tract length
# that the aligner shows as mismatches or a soft clip instead of a gap.
_LOCAL_WINDOW = 8

# Distances above one edit are not told apart.
_FAR = 2

_G = ord("G")

DROP_REASONS = ("g_at_run_side", "flank_not_placed", "flank_not_covered", "flanks_crossed")
PATHS = ("fast", "cigar", "local")

# CIGAR operation codes.
_M_OPS = (0, 7, 8)          # M, =, X: consume read and reference
_QUERY_OPS = (1, 4, 6)      # I, S, P: consume read only (P as pysam walks it)
_REF_OPS = (2, 3)           # D, N: consume reference only
_INS, _DEL, _SOFT = 1, 2, 4


@dataclasses.dataclass(frozen=True)
class FlankRecord:
    """One locus side's flank.

    Attributes
    ----------
    bases : str
        The flank bases in polyG orientation, in read order.
    ref_positions : tuple of int
        The 0-based reference coordinate of each base in `bases`.
    run_side_offset : int
        Offset in `bases` of the base next to the repeat: the last offset
        for the upstream flank, 0 for the downstream flank.
    masked_offsets : tuple of int
        Offsets where the donor has a G. A masked position matches any base.
    """

    bases: str
    ref_positions: tuple[int, ...]
    run_side_offset: int
    masked_offsets: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if len(self.bases) != len(self.ref_positions):
            raise ValueError("bases and ref_positions differ in length")
        if self.run_side_offset not in (0, len(self.bases) - 1):
            raise ValueError("the run-side base must be the first or the last flank base")
        if any(not 0 <= m < len(self.bases) for m in self.masked_offsets):
            raise ValueError("masked offset outside the flank")

    @property
    def is_upstream(self) -> bool:
        return self.run_side_offset == len(self.bases) - 1


@dataclasses.dataclass(frozen=True)
class FlankPlacement:
    """Output of :func:`place_flanks`, one entry per read.

    `tracts` and `tract_qualities` are in polyG orientation and are None
    for a dropped read. `drop_reason` is one of `DROP_REASONS` or None, and
    `drop_side` names the side that caused it ("upstream", "downstream",
    or None for a kept read or for "flanks_crossed"). Each side's path is
    one of `PATHS`, or None when that side is not placed. The run-side
    arrays hold the placed run-side read index in polyG orientation, or -1.
    """

    tracts: list[str | None]
    tract_qualities: list[np.ndarray | None]
    drop_reason: list[str | None]
    drop_side: list[str | None]
    upstream_path: list[str | None]
    downstream_path: list[str | None]
    upstream_run_side: np.ndarray
    downstream_run_side: np.ndarray


# ---------------------------------------------------------------------------
# Band-1 flank distance
# ---------------------------------------------------------------------------


def _pack(reads: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad reads with NUL into an (n, width) uint8 matrix; also return lengths."""
    lengths = np.fromiter((len(s) for s in reads), dtype=np.int64, count=len(reads))
    width = int(lengths.max()) if len(reads) else 0
    width = max(width, 1)
    flat = "".join(s.ljust(width, "\0") for s in reads).encode("ascii")
    return np.frombuffer(flat, dtype=np.uint8).reshape(len(reads), width), lengths


def _pattern(flank: str, upstream: bool, masked_offsets: Sequence[int]
             ) -> tuple[np.ndarray, np.ndarray]:
    """The flank bases other than the run-side base, from the run side outward."""
    k = len(flank)
    if upstream:
        pattern = flank[:-1][::-1]
        offsets = [k - 2 - i for i in range(k - 1)]
    else:
        pattern = flank[1:]
        offsets = [i + 1 for i in range(k - 1)]
    masked = set(masked_offsets)
    mask = np.array([o in masked for o in offsets], dtype=bool)
    return np.frombuffer(pattern.encode("ascii"), dtype=np.uint8), mask


def _banded_distance(matrix: np.ndarray, lengths: np.ndarray, rows: np.ndarray,
                     run_side: np.ndarray, pattern: np.ndarray, mask: np.ndarray,
                     upstream: bool) -> np.ndarray:
    """Band-1 distance of `pattern` placed next to `run_side` in each of `rows`.

    The pattern starts next to the run-side base and runs outward; its outer
    end is free in the read. Returns 0, 1, or `_FAR` for two or more edits.
    A run-side index outside the read scores `_FAR`.
    """
    n = len(pattern)
    m = len(rows)
    if m == 0:
        return np.zeros(0, dtype=np.int64)
    steps = np.arange(n + 1)
    if upstream:
        idx = run_side[:, None] - 1 - steps
    else:
        idx = run_side[:, None] + 1 + steps
    row_len = lengths[rows][:, None]
    valid = (idx >= 0) & (idx < row_len)
    text = matrix[rows[:, None], np.clip(idx, 0, matrix.shape[1] - 1)]

    def match(text_part, valid_part, pat, pat_mask):
        return ((text_part == pat) | pat_mask) & valid_part

    eq0 = match(text[:, :n], valid[:, :n], pattern, mask)
    # Text base k against pattern base k + 1: a pattern base was deleted before k.
    eq_skip_pattern = match(text[:, :n - 1], valid[:, :n - 1], pattern[1:], mask[1:])
    # Text base k + 1 against pattern base k: a read base was inserted before k.
    eq_skip_text = match(text[:, 1:], valid[:, 1:], pattern, mask)

    ones = np.ones((m, 1), dtype=bool)
    prefix = np.concatenate([ones, np.cumprod(eq0, axis=1, dtype=bool)], axis=1)

    def suffix_all(a):
        # suffix[:, i] = all(a[:, i:]), for i = 0..a.shape[1]
        rev = np.cumprod(a[:, ::-1], axis=1, dtype=bool)[:, ::-1]
        return np.concatenate([rev, ones], axis=1)

    deletion = (prefix[:, :n] & suffix_all(eq_skip_pattern)).any(axis=1)
    insertion = (prefix[:, :n] & suffix_all(eq_skip_text)[:, :n]).any(axis=1)
    mismatches = n - eq0.sum(axis=1)

    dist = np.where(mismatches == 0, 0,
                    np.where((mismatches == 1) | deletion | insertion, 1, _FAR))
    inside = (run_side >= 0) & (run_side < lengths[rows])
    return np.where(inside, dist, _FAR).astype(np.int64)


def banded_flank_distance(flank: str, reads: Sequence[str], run_side_indices: np.ndarray,
                          *, upstream: bool, masked_offsets: Sequence[int]) -> np.ndarray:
    """Flank edits when the flank's run-side base sits on ``reads[i][run_side_indices[i]]``.

    The run-side base is not counted. The other flank bases align to the
    read bases on the far side of it, with the outer end free, by edit
    distance with band 1. A masked offset matches any base.

    Parameters
    ----------
    flank : str
        Flank bases in read order (polyG orientation).
    reads : sequence of str
        Reads in polyG orientation, upper-case.
    run_side_indices : ndarray of int
        One read index per read.
    upstream : bool
        True for the upstream flank (run-side base last), False for the
        downstream flank (run-side base first).
    masked_offsets : sequence of int
        Flank offsets that match any base.

    Returns
    -------
    ndarray of int64
        0 or 1 edits, or 2 for two or more.
    """
    if not len(reads):
        return np.zeros(0, dtype=np.int64)
    matrix, lengths = _pack(reads)
    pattern, mask = _pattern(flank, upstream, masked_offsets)
    rows = np.arange(len(reads))
    return _banded_distance(matrix, lengths, rows, np.asarray(run_side_indices, dtype=np.int64),
                            pattern, mask, upstream)


# ---------------------------------------------------------------------------
# Alignment in polyG orientation
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


def window_reference_position(index: int, *, ref_orientation: str, window_start: int,
                              window_len: int) -> int:
    """Reference coordinate of index `index` in the polyG-orientation window."""
    if ref_orientation == "-":
        return window_start + window_len - 1 - index
    return window_start + index


def polyg_alignments(
    read_seqs: Sequence[str],
    cigars: Sequence[Sequence[tuple[int, int]]],
    reference_starts: Sequence[int] | np.ndarray,
    *,
    ref_orientation: str,
    window_start: int,
    window_len: int,
) -> tuple[list[str], list[list[list[int]]], np.ndarray]:
    """Reads, CIGARs and alignment starts in polyG orientation.

    Reads are upper-cased. Each CIGAR is a list of mutable ``[op, length]``
    pairs without hard clips. Each start is the polyG-orientation window
    index (see :func:`window_index`) of the first aligned reference base.
    At a "-" locus the read is reverse-complemented and the CIGAR reversed.
    """
    n_reads = len(read_seqs)
    minus = ref_orientation == "-"
    reads: list[str] = []
    polyg_cigars: list[list[list[int]]] = []
    starts = np.empty(n_reads, dtype=np.int64)
    for i in range(n_reads):
        seq = read_seqs[i].upper()
        cig = [[op, length] for op, length in cigars[i] if op != 5]
        start = int(reference_starts[i])
        if minus:
            span = sum(length for op, length in cig if op in _M_OPS or op in _REF_OPS)
            seq = revcomp(seq)
            cig.reverse()
            start = start + span - 1
        starts[i] = window_index(start, ref_orientation=ref_orientation,
                                 window_start=window_start, window_len=window_len)
        reads.append(seq)
        polyg_cigars.append(cig)
    return reads, polyg_cigars, starts


def _shift_indels(cigar: list[list[int]], start: int, read: str, reference: str,
                  middle: float) -> None:
    """Move each indel away from `middle`, in place, while the bases allow it.

    `cigar` and `start` are in polyG orientation and `reference` is indexed
    by the same coordinates. A shift by one base keeps the alignment the
    same when the base that changes sides of the gap equals the gap's base
    at the other end. Reference bases outside `reference` stop the shift.
    """
    moved = True
    while moved:
        moved = False
        q, p = 0, start
        for k, (op, length) in enumerate(cigar):
            if op in (_INS, _DEL) and 0 < k < len(cigar) - 1 \
                    and cigar[k - 1][0] in _M_OPS and cigar[k + 1][0] in _M_OPS:
                if p < middle:
                    if cigar[k - 1][1] > 1:
                        if op == _INS:
                            same = read[q - 1] == read[q + length - 1]
                        else:
                            same = (0 <= p - 1 and p + length - 1 < len(reference)
                                    and reference[p - 1] == reference[p + length - 1])
                        if same:
                            cigar[k - 1][1] -= 1
                            cigar[k + 1][1] += 1
                            moved = True
                            break
                elif cigar[k + 1][1] > 1:
                    if op == _INS:
                        same = q + length < len(read) and read[q] == read[q + length]
                    else:
                        same = (0 <= p and p + length < len(reference)
                                and reference[p] == reference[p + length])
                    if same:
                        cigar[k - 1][1] += 1
                        cigar[k + 1][1] -= 1
                        moved = True
                        break
            if op in _M_OPS:
                q += length
                p += length
            elif op in _QUERY_OPS:
                q += length
            elif op in _REF_OPS:
                p += length


def _project(cigar: list[list[int]], start: int, position: int, read_len: int
             ) -> tuple[int, bool]:
    """Read index aligned to reference `position`, and whether it is an M column.

    A deleted position maps to the next read base. A position outside the
    aligned span maps by linear extension past the alignment's end,
    through any soft clip.
    """
    q, p = 0, start
    for op, length in cigar:
        if op in _M_OPS:
            if p <= position < p + length:
                return q + position - p, True
            q += length
            p += length
        elif op in _QUERY_OPS:
            q += length
        elif op in _REF_OPS:
            if p <= position < p + length:
                return q, False
            p += length
    if position < start:
        lead = cigar[0][1] if cigar[0][0] == _SOFT else 0
        return lead - (start - position), False
    trail = cigar[-1][1] if cigar[-1][0] == _SOFT else 0
    return read_len - trail + (position - p), False


def _flank_events(cigar: list[list[int]], start: int, lo: int, hi: int) -> tuple[bool, bool]:
    """Whether the alignment has an indel, or a soft clip, over reference ``[lo, hi)``."""
    p = start
    indel = clip = False
    last = len(cigar) - 1
    for k, (op, length) in enumerate(cigar):
        if op in _M_OPS:
            p += length
        elif op in _REF_OPS:
            if lo < p + length and p < hi:
                indel = True
            p += length
        elif op == _INS:
            if lo < p < hi:
                indel = True
        elif op == _SOFT:
            if (k == 0 and p > lo) or (k == last and p < hi):
                clip = True
    return indel, clip


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


def place_flanks(
    read_seqs: Sequence[str],
    cigars: Sequence[Sequence[tuple[int, int]]],
    reference_starts: np.ndarray,
    base_qualities: Sequence[np.ndarray | None],
    *,
    upstream: FlankRecord,
    downstream: FlankRecord,
    ref_orientation: str,
    reference_window: str,
    reference_window_start: int,
) -> FlankPlacement:
    """Place both flanks in a block of reads at one locus; cut out each tract.

    Parameters
    ----------
    read_seqs : sequence of str
        Read sequences as stored in the BAM (reference orientation).
    cigars : sequence of sequence of (op, length)
        CIGAR tuples, as `pysam` reports them.
    reference_starts : ndarray of int
        0-based reference start of each alignment.
    base_qualities : sequence of ndarray or None
        Base qualities in stored order; None gives None tract qualities.
    upstream, downstream : FlankRecord
        The two flanks of the locus.
    ref_orientation : str
        "+" if the repeat reads as Gs on the reference, "-" otherwise.
    reference_window : str
        Reference bases (reference orientation) around the locus. They
        decide which indels can move; they must cover both flanks.
    reference_window_start : int
        Reference coordinate of ``reference_window[0]``.

    Returns
    -------
    FlankPlacement
    """
    n_reads = len(read_seqs)
    window_len = len(reference_window)
    minus = ref_orientation == "-"
    polyg_reference = revcomp(reference_window.upper()) if minus else reference_window.upper()

    sides = []
    for flank in (upstream, downstream):
        us = [window_index(r, ref_orientation=ref_orientation,
                           window_start=reference_window_start, window_len=window_len)
              for r in flank.ref_positions]
        if any(b - a != 1 for a, b in zip(us, us[1:], strict=False)):
            raise ValueError("flank ref_positions do not run along the polyG orientation")
        sides.append((flank, us[0], us[flank.run_side_offset]))
    middle = (sides[0][2] + sides[1][2]) / 2

    reads, polyg_cigars, starts = polyg_alignments(
        read_seqs, cigars, reference_starts, ref_orientation=ref_orientation,
        window_start=reference_window_start, window_len=window_len)
    for i in range(n_reads):
        cig = polyg_cigars[i]
        if any(op in (_INS, _DEL) for op, _ in cig):
            _shift_indels(cig, int(starts[i]), reads[i], polyg_reference, middle)

    matrix, lengths = _pack(reads) if n_reads else (np.zeros((0, 1), np.uint8),
                                                     np.zeros(0, np.int64))

    run_side = []
    side_paths = []
    side_drop = []
    for flank, lo, rs in sides:
        is_up = flank.is_upstream
        k = len(flank.bases)
        b0 = np.empty(n_reads, dtype=np.int64)
        on_m = np.empty(n_reads, dtype=bool)
        clean = np.empty(n_reads, dtype=bool)
        has_indel = np.empty(n_reads, dtype=bool)
        has_clip = np.empty(n_reads, dtype=bool)
        for i in range(n_reads):
            cig = polyg_cigars[i]
            b0[i], on_m[i] = _project(cig, int(starts[i]), rs, int(lengths[i]))
            indel, clip = _flank_events(cig, int(starts[i]), lo, lo + k)
            has_indel[i] = indel
            has_clip[i] = clip
            clean[i] = on_m[i] and not indel and not clip

        pattern, mask = _pattern(flank.bases, is_up, flank.masked_offsets)
        far_end = b0 - (k - 1) if is_up else b0 + (k - 1)
        covered = (np.minimum(b0, far_end) >= 0) & (np.maximum(b0, far_end) < lengths)

        # Fast path: the projection, with no indel or clip over the flank.
        rows = np.arange(n_reads)
        fast_rows = rows[clean & covered]
        d0 = _banded_distance(matrix, lengths, fast_rows, b0[fast_rows], pattern, mask, is_up)
        g0 = matrix[fast_rows, b0[fast_rows]] == _G
        fast = fast_rows[(d0 == 0) & ~g0]

        chosen = np.full(n_reads, -1, dtype=np.int64)
        path: list[str | None] = [None] * n_reads
        drop: list[str | None] = [None] * n_reads
        chosen[fast] = b0[fast]
        for i in fast.tolist():
            path[i] = "fast"

        # Window search for every other read.
        rest = np.setdiff1d(rows, fast, assume_unique=True)
        if rest.size:
            width = 2 * _LOCAL_WINDOW + 1
            shift = np.arange(-_LOCAL_WINDOW, _LOCAL_WINDOW + 1)
            cand = (b0[rest][:, None] + shift[None, :]).ravel()
            cand_rows = np.repeat(rest, width)
            dist = _banded_distance(matrix, lengths, cand_rows, cand, pattern, mask, is_up)
            inside = (cand >= 0) & (cand < lengths[cand_rows])
            is_g = np.zeros(cand.shape, dtype=bool)
            is_g[inside] = matrix[cand_rows[inside], cand[inside]] == _G
            score = dist + (np.tile(shift, rest.size) != 0)
            nearness = cand if is_up else -cand
            # Order: score, then non-G, then nearest the run. Positions with
            # more than one flank edit cannot win.
            span = 2 * (window_len + int(lengths.max()) + 2 * _LOCAL_WINDOW + 4)
            key = (score * 2 + is_g) * span - (nearness - nearness.min())
            key = np.where(dist <= 1, key, np.iinfo(np.int64).max)
            key = key.reshape(rest.size, width)
            best = key.argmin(axis=1)
            found = key[np.arange(rest.size), best] != np.iinfo(np.int64).max
            best_b = cand.reshape(rest.size, width)[np.arange(rest.size), best]
            best_g = is_g.reshape(rest.size, width)[np.arange(rest.size), best]
            for j, i in enumerate(rest.tolist()):
                if not found[j]:
                    drop[i] = "flank_not_placed" if covered[i] else "flank_not_covered"
                elif best_g[j]:
                    drop[i] = "g_at_run_side"
                else:
                    b = int(best_b[j])
                    chosen[i] = b
                    if b == b0[i] and on_m[i] and has_indel[i] and not has_clip[i]:
                        path[i] = "cigar"
                    else:
                        path[i] = "local"
        run_side.append(chosen)
        side_paths.append(path)
        side_drop.append(drop)

    tracts: list[str | None] = [None] * n_reads
    tract_quals: list[np.ndarray | None] = [None] * n_reads
    reasons: list[str | None] = [None] * n_reads
    drop_side: list[str | None] = [None] * n_reads
    up_b, dn_b = run_side
    for i in range(n_reads):
        if side_drop[0][i] is not None:
            reasons[i], drop_side[i] = side_drop[0][i], "upstream"
            continue
        if side_drop[1][i] is not None:
            reasons[i], drop_side[i] = side_drop[1][i], "downstream"
            continue
        lo, hi = int(up_b[i]) + 1, int(dn_b[i])
        if hi < lo:
            reasons[i] = "flanks_crossed"
            continue
        tracts[i] = reads[i][lo:hi]
        quals = base_qualities[i]
        if quals is not None:
            q = np.asarray(quals)
            if minus:
                q = q[::-1]
            tract_quals[i] = q[lo:hi].copy()

    return FlankPlacement(
        tracts=tracts,
        tract_qualities=tract_quals,
        drop_reason=reasons,
        drop_side=drop_side,
        upstream_path=side_paths[0],
        downstream_path=side_paths[1],
        upstream_run_side=up_b,
        downstream_run_side=dn_b,
    )
