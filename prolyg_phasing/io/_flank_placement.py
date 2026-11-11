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
   as a run-side position. The score is the cost of the flank edit there
   plus 1 if the position is not the projection. A flank mismatch costs
   `_MISMATCH_COST` (1) and a 1-base flank indel costs `_INDEL_COST` (2).
   The edit is found by a banded edit distance with band 1 over the other
   nine flank bases. A masked flank position matches any base. The
   run-side base itself is not counted. Only positions with at most one
   flank edit, one mismatch or one 1-base indel, are kept.
4. The lowest score wins. On a tie, a non-G run-side base wins over a G.
   On a further tie, the position nearest the run wins.
5. If the winning run-side base is a G, the side drops as
   ``"g_at_run_side"``. If no position is kept, the side drops as
   ``"flank_not_covered"`` when the projected flank is not inside the
   read, and as ``"flank_not_placed"`` otherwise. Both sides are placed
   in every read, so a read can drop on both sides. If both sides are
   placed and the downstream run-side base is not after the upstream one,
   the read drops as ``"flanks_crossed"``. That reason belongs to the
   read, not to a side.

The projection costs nothing extra when it is exact, so the rule reports
three paths. "fast": the alignment covers the flank with M operations and
the read equals the flank at every unmasked position. "cigar": the
alignment has an indel inside the flank, and the projection wins. "local":
any other placement. A fast placement has score 0. Every other position
pays 1 for leaving the projection, so no other position can score 0, and a
fast read needs no window search.

The CIGARs of a block of reads are walked once for both sides, as arrays
over the whole block. A read whose CIGAR is one M operation has no indel to
move, and its projection is the run-side position minus the alignment
start. A read with one indel moves it over the whole block; only a read
with more than one indel moves them one read at a time.

A read with an extra non-G base right after the run, or with a missing copy
of the run-side base next to the run, is the same string as a read from a
molecule with another tract. Read as the run-side base, an extra base
scores 1 for leaving the projection; read as a flank insertion, it scores
2. So an extra base goes to the tract, as a boundary zero run. A missing
copy of the run-side base goes to the flank.
"""

from __future__ import annotations

import csv
import dataclasses
import itertools
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from prolyg_phasing.io._anchors import COMP, revcomp

# Read positions on each side of the projected run-side base that the
# window search scores. It must cover the largest change of tract length
# that the aligner shows as mismatches or a soft clip instead of a gap.
_LOCAL_WINDOW = 8

# Cost of one flank mismatch in the window score. Substitutions outnumber
# small indels in Illumina reads outside homopolymers.
_MISMATCH_COST = 1

# Cost of one 1-base flank indel in the window score; twice a mismatch,
# since substitutions outnumber small indels in Illumina reads outside
# homopolymers.
_INDEL_COST = 2

# The cost of a placement with two or more flank edits. It is above every
# allowed cost, so such a placement never wins.
_FAR = _INDEL_COST + 1

_G = ord("G")

PATHS = ("fast", "cigar", "local")
SIDE_DROP_REASONS = ("g_at_run_side", "flank_not_placed", "flank_not_covered")
DROP_REASONS = (*SIDE_DROP_REASONS, "flanks_crossed")

# Codes of the per-side outputs; 0 is "none".
_FAST, _CIGAR, _LOCAL = 1, 2, 3
_G_AT_RUN_SIDE, _NOT_PLACED, _NOT_COVERED = 1, 2, 3
_PATH_NAMES = np.array([None, *PATHS], dtype=object)
_SIDE_DROP_NAMES = np.array([None, *SIDE_DROP_REASONS], dtype=object)

# CIGAR operation codes.
_M_OPS = (0, 7, 8)          # M, =, X: consume read and reference
_QUERY_OPS = (1, 4, 6)      # I, S, P: consume read only (P as pysam walks it)
_REF_OPS = (2, 3)           # D, N: consume reference only
_INS, _DEL, _SOFT, _HARD = 1, 2, 4, 5

# Byte tables from a stored read base to its polyG-orientation base: the
# upper case at a "+" locus, the complement of the upper case at a "-"
# locus. They give the same bases as `str.upper` and `revcomp`.
_LOWER = bytes(range(ord("a"), ord("z") + 1))
_UPPER_BYTES = bytes.maketrans(_LOWER, _LOWER.upper())
_COMPLEMENT_BYTES = bytearray(_UPPER_BYTES)
for _base, _complement in zip("ACGTN", "ACGTN".translate(COMP), strict=True):
    _COMPLEMENT_BYTES[ord(_base)] = _COMPLEMENT_BYTES[ord(_base.lower())] = ord(_complement)
_COMPLEMENT_BYTES = bytes(_COMPLEMENT_BYTES)


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

    @property
    def run_side_position(self) -> int:
        """Reference coordinate of the run-side base."""
        return self.ref_positions[self.run_side_offset]

    @property
    def outer_position(self) -> int:
        """Reference coordinate of the flank base farthest from the repeat."""
        return self.ref_positions[0 if self.is_upstream else -1]


# The reference window status of a locus side in the flank table and of a
# locus in the panel: a read rebuilt the window, or none did and the side
# has no flank.
WINDOW_RECOVERED = "recovered"
WINDOW_UNRECOVERABLE = "ref_window_unrecoverable"

# Side-table fields that a side with no reference window leaves empty.
_FLANK_FIELDS = ("flank_bases", "flank_ref_positions", "masked_offsets")


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
        masked = tuple(int(m) for m in row["masked_offsets"].split(",")) \
            if row["masked_offsets"] else ()
        record = FlankRecord(
            bases=row["flank_bases"],
            ref_positions=positions,
            run_side_offset=len(positions) - 1 if row["side"] == "upstream" else 0,
            masked_offsets=masked,
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


@dataclasses.dataclass(frozen=True)
class FlankPlacement:
    """Output of :func:`place_flanks`, one entry per read.

    Attributes
    ----------
    tracts : ndarray of object
        The tract in polyG orientation for a kept read, else None.
    kept : ndarray of bool
        Both sides placed and not crossed.
    upstream_path, downstream_path : ndarray of object
        Each side's path, one of `PATHS`, or None when the side is not
        placed.
    upstream_drop, downstream_drop : ndarray of object
        Each side's drop reason, one of `SIDE_DROP_REASONS`, or None when
        the side is placed.
    flanks_crossed : ndarray of bool
        Both sides placed, and the downstream run-side base is not after
        the upstream one.
    upstream_run_side, downstream_run_side : ndarray of int64
        The placed run-side read index in polyG orientation, or -1. The
        tract of a kept read is the polyG-orientation read on
        ``[upstream_run_side + 1, downstream_run_side)``.
    """

    tracts: np.ndarray
    kept: np.ndarray
    upstream_path: np.ndarray
    downstream_path: np.ndarray
    upstream_drop: np.ndarray
    downstream_drop: np.ndarray
    flanks_crossed: np.ndarray
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


def _outward_bases(matrix: np.ndarray, lengths: np.ndarray, rows: np.ndarray,
                   run_side: np.ndarray, n_bases: int, upstream: bool
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Read bases outward from each run-side index, and whether each is inside the read."""
    steps = np.arange(n_bases)
    if upstream:
        idx = run_side[:, None] - 1 - steps
    else:
        idx = run_side[:, None] + 1 + steps
    valid = (idx >= 0) & (idx < lengths[rows][:, None])
    text = matrix[rows[:, None], np.clip(idx, 0, matrix.shape[1] - 1)]
    return text, valid


def _banded_distance(matrix: np.ndarray, lengths: np.ndarray, rows: np.ndarray,
                     run_side: np.ndarray, pattern: np.ndarray, mask: np.ndarray,
                     upstream: bool) -> np.ndarray:
    """Band-1 distance of `pattern` placed next to `run_side` in each of `rows`.

    The pattern starts next to the run-side base and runs outward; its outer
    end is free in the read. Returns 0 for no edit, `_MISMATCH_COST` for one
    mismatch, `_INDEL_COST` for one 1-base indel, and `_FAR` for two or more
    edits. A run-side index outside the read scores `_FAR`.
    """
    n = len(pattern)
    m = len(rows)
    if m == 0:
        return np.zeros(0, dtype=np.int64)
    text, valid = _outward_bases(matrix, lengths, rows, run_side, n + 1, upstream)

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
                    np.where(mismatches == 1, _MISMATCH_COST,
                             np.where(deletion | insertion, _INDEL_COST, _FAR)))
    inside = (run_side >= 0) & (run_side < lengths[rows])
    return np.where(inside, dist, _FAR).astype(np.int64)


def _exact_match(matrix: np.ndarray, lengths: np.ndarray, rows: np.ndarray,
                 run_side: np.ndarray, pattern: np.ndarray, mask: np.ndarray,
                 upstream: bool) -> np.ndarray:
    """Whether the read equals `pattern` at every unmasked flank position.

    For a flank inside the read this is a band-1 cost of 0, without the
    indel terms.
    """
    if len(rows) == 0:
        return np.zeros(0, dtype=bool)
    text, valid = _outward_bases(matrix, lengths, rows, run_side, len(pattern), upstream)
    return (((text == pattern) | mask) & valid).all(axis=1)


def banded_flank_distance(flank: str, reads: Sequence[str], run_side_indices: np.ndarray,
                          *, upstream: bool, masked_offsets: Sequence[int]) -> np.ndarray:
    """Flank edit cost when the flank's run-side base sits on ``reads[i][run_side_indices[i]]``.

    The run-side base is not counted. The other flank bases align to the
    read bases on the far side of it, with the outer end free, by edit
    distance with band 1. A masked offset matches any base. One mismatch
    costs `_MISMATCH_COST` and one 1-base indel `_INDEL_COST`.

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
        0 for no edit, 1 for one mismatch, 2 for one 1-base indel, and 3
        (`_FAR`) for two or more edits.
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


def _polyg_cigar(cigar: Sequence[tuple[int, int]], reference_start: int, *,
                 ref_orientation: str, window_start: int, window_len: int
                 ) -> tuple[list[list[int]], int]:
    """One CIGAR without hard clips, and its start, in polyG orientation.

    The CIGAR is a list of mutable ``[op, length]`` pairs, reversed at a
    "-" locus. The start is the polyG-orientation window index of the
    first aligned reference base.
    """
    cig = [[op, length] for op, length in cigar if op != _HARD]
    start = reference_start
    if ref_orientation == "-":
        span = sum(length for op, length in cig if op in _M_OPS or op in _REF_OPS)
        cig.reverse()
        start = start + span - 1
    return cig, window_index(start, ref_orientation=ref_orientation,
                             window_start=window_start, window_len=window_len)


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
        cig, starts[i] = _polyg_cigar(cigars[i], int(reference_starts[i]),
                                      ref_orientation=ref_orientation,
                                      window_start=window_start, window_len=window_len)
        reads.append(revcomp(seq) if minus else seq)
        polyg_cigars.append(cig)
    return reads, polyg_cigars, starts


def _polyg_matrix(read_seqs: Sequence[str], minus: bool) -> tuple[np.ndarray, np.ndarray]:
    """Reads in polyG orientation as a right-padded (n, width) uint8 matrix, and lengths.

    The same bases as :func:`polyg_alignments` gives. At a "-" locus each
    read is padded on the left, complemented as one byte string, and the
    matrix columns are reversed.
    """
    n_reads = len(read_seqs)
    lengths = np.fromiter(map(len, read_seqs), dtype=np.int64, count=n_reads)
    width = max(int(lengths.max()) if n_reads else 0, 1)
    if minus:
        flat = "".join([s.rjust(width, "\0") for s in read_seqs]).encode("ascii")
        matrix = np.frombuffer(flat.translate(_COMPLEMENT_BYTES), dtype=np.uint8)
        return np.ascontiguousarray(matrix.reshape(n_reads, width)[:, ::-1]), lengths
    flat = "".join([s.ljust(width, "\0") for s in read_seqs]).encode("ascii")
    matrix = np.frombuffer(flat.translate(_UPPER_BYTES), dtype=np.uint8)
    return matrix.reshape(n_reads, width), lengths


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
    through any soft clip. :func:`_alignment_geometry` computes the same
    for a block of reads.
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


@dataclasses.dataclass
class _Alignments:
    """A block of polyG-orientation CIGARs as (n_reads, n_ops) arrays.

    Hard clips are removed and, at a "-" locus, each CIGAR is reversed.
    Each row holds its `n_ops` operations first; the padding has op -1 and
    length 0. `start` is the polyG-orientation window index of the first
    aligned reference base, and `span` the reference bases the alignment
    covers.
    """

    ops: np.ndarray
    lengths: np.ndarray
    n_ops: np.ndarray
    start: np.ndarray
    span: np.ndarray

    def offsets(self) -> tuple[np.ndarray, np.ndarray]:
        """Read index and window index at which each operation starts."""
        query = np.isin(self.ops, _M_OPS + _QUERY_OPS)
        reference = np.isin(self.ops, _M_OPS + _REF_OPS)
        q = np.cumsum(self.lengths * query, axis=1) - self.lengths * query
        p = np.cumsum(self.lengths * reference, axis=1) - self.lengths * reference
        return q, p + self.start[:, None]


def _polyg_cigar_arrays(cigars: Sequence[Sequence[tuple[int, int]]],
                        reference_starts: np.ndarray, *, ref_orientation: str,
                        window_start: int, window_len: int) -> _Alignments:
    """The CIGARs of a block of reads in polyG orientation, as arrays.

    The same operations and starts as :func:`_polyg_cigar` gives one read
    at a time.
    """
    n_reads = len(cigars)
    n_ops = np.fromiter((len(c) for c in cigars), dtype=np.int64, count=n_reads)
    total = int(n_ops.sum())
    flat = np.fromiter(itertools.chain.from_iterable(cigars),
                       dtype=np.dtype((np.int64, 2)), count=total)
    width = max(int(n_ops.max()) if n_reads else 0, 1)
    rows = np.repeat(np.arange(n_reads), n_ops)
    cols = np.arange(total) - np.repeat(np.cumsum(n_ops) - n_ops, n_ops)
    ops = np.full((n_reads, width), -1, dtype=np.int64)
    lengths = np.zeros((n_reads, width), dtype=np.int64)
    ops[rows, cols] = flat[:, 0]
    lengths[rows, cols] = flat[:, 1]

    # Drop hard clips: a stable sort moves the kept operations first.
    kept = (ops != -1) & (ops != _HARD)
    if not kept.sum(axis=1).tolist() == n_ops.tolist():
        order = np.argsort(~kept, axis=1, kind="stable")
        ops = np.take_along_axis(ops, order, axis=1)
        lengths = np.take_along_axis(lengths, order, axis=1)
        n_ops = kept.sum(axis=1)
        pad = np.arange(width)[None, :] >= n_ops[:, None]
        ops[pad] = -1
        lengths[pad] = 0
    span = (lengths * np.isin(ops, _M_OPS + _REF_OPS)).sum(axis=1)
    starts = np.asarray(reference_starts, dtype=np.int64)
    if ref_orientation == "-":
        index = n_ops[:, None] - 1 - np.arange(width)[None, :]
        inside = index >= 0
        index = np.where(inside, index, 0)
        ops = np.where(inside, np.take_along_axis(ops, index, axis=1), -1)
        lengths = np.where(inside, np.take_along_axis(lengths, index, axis=1), 0)
        start = window_start + window_len - 1 - (starts + span - 1)
    else:
        start = starts - window_start
    return _Alignments(ops, lengths, n_ops, start, span)


def _shift_single_indels(aln: _Alignments, rows: np.ndarray, matrix: np.ndarray,
                         read_lengths: np.ndarray, reference: np.ndarray,
                         middle: float) -> None:
    """:func:`_shift_indels` for reads with exactly one indel, over the block, in place.

    With one indel the order of the one-base moves does not matter, so
    each read moves its indel step by step until the bases stop it.
    """
    is_indel = (aln.ops[rows] == _INS) | (aln.ops[rows] == _DEL)
    k = is_indel.argmax(axis=1)
    n_ops = aln.n_ops[rows]
    movable = (k > 0) & (k < n_ops - 1)
    left_op = aln.ops[rows, np.maximum(k - 1, 0)]
    right_op = aln.ops[rows, np.minimum(k + 1, aln.ops.shape[1] - 1)]
    movable &= np.isin(left_op, _M_OPS) & np.isin(right_op, _M_OPS)
    rows, k = rows[movable], k[movable]
    if rows.size == 0:
        return
    q_start, p_start = aln.offsets()
    q = q_start[rows, k]
    p = p_start[rows, k]
    op = aln.ops[rows, k]
    size = aln.lengths[rows, k]
    a = aln.lengths[rows, k - 1].copy()
    b = aln.lengths[rows, k + 1].copy()
    leftward = p < middle
    ins = op == _INS
    n_ref = len(reference)
    read_len = read_lengths[rows]
    width = matrix.shape[1]

    def read_at(i):
        return matrix[rows, np.clip(i, 0, width - 1)]

    def ref_at(i):
        return reference[np.clip(i, 0, n_ref - 1)]

    active = np.ones(rows.size, dtype=bool)
    while active.any():
        # A move to the left keeps the alignment when the base that changes
        # sides of the gap equals the gap's base at the other end.
        same_left = np.where(
            ins, read_at(q - 1) == read_at(q + size - 1),
            (p - 1 >= 0) & (p + size - 1 < n_ref) & (ref_at(p - 1) == ref_at(p + size - 1)))
        same_right = np.where(
            ins, (q + size < read_len) & (read_at(q) == read_at(q + size)),
            (p >= 0) & (p + size < n_ref) & (ref_at(p) == ref_at(p + size)))
        go_left = active & leftward & (a > 1) & same_left
        go_right = active & ~leftward & (b > 1) & same_right
        step = go_right.astype(np.int64) - go_left.astype(np.int64)
        q += step
        p += step
        a += step
        b -= step
        active = go_left | go_right
    aln.lengths[rows, k - 1] = a
    aln.lengths[rows, k + 1] = b


def _shift_block_indels(aln: _Alignments, matrix: np.ndarray, read_lengths: np.ndarray,
                        reference: str, middle: float) -> None:
    """:func:`_shift_indels` for every read of a block, in place.

    A read with one indel moves it over the whole block. A read with more
    than one indel moves them one read at a time, because one indel can
    stop another.
    """
    n_indels = ((aln.ops == _INS) | (aln.ops == _DEL)).sum(axis=1)
    reference_bases = np.frombuffer(reference.encode("ascii"), dtype=np.uint8)
    _shift_single_indels(aln, np.flatnonzero(n_indels == 1), matrix, read_lengths,
                         reference_bases, middle)
    several = np.flatnonzero(n_indels > 1).tolist()
    if not several:
        return
    width = matrix.shape[1]
    text = matrix.tobytes().decode("latin-1")
    for i in several:
        n_ops = int(aln.n_ops[i])
        cig = [[op, length] for op, length in zip(aln.ops[i, :n_ops].tolist(),
                                                  aln.lengths[i, :n_ops].tolist(),
                                                  strict=True)]
        _shift_indels(cig, int(aln.start[i]), text[i * width:i * width + int(read_lengths[i])],
                      reference, middle)
        aln.lengths[i, :n_ops] = [length for _, length in cig]


def _alignment_geometry(aln: _Alignments, read_lengths: np.ndarray,
                        positions: Sequence[int], intervals: Sequence[tuple[int, int]]
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Projection of positions, and indel or clip over intervals, for a block of reads.

    For each reference position: the read index aligned to it, and whether
    an M operation covers it (see :func:`_project`). For each reference
    interval ``[lo, hi)``: whether the alignment has an indel over it, and
    whether a soft clip reaches into it. Each output has one row per
    position or interval and one column per read.
    """
    n_reads = len(read_lengths)
    q_start, p_start = aln.offsets()
    ops, lengths = aln.ops, aln.lengths
    is_m = np.isin(ops, _M_OPS)
    on_reference = np.isin(ops, _M_OPS + _REF_OPS)
    pick = np.arange(n_reads)
    last = np.maximum(aln.n_ops - 1, 0)
    lead = np.where(ops[:, 0] == _SOFT, lengths[:, 0], 0)
    trail = np.where(ops[pick, last] == _SOFT, lengths[pick, last], 0)
    end = aln.start + aln.span

    projection = np.empty((len(positions), n_reads), dtype=np.int64)
    on_m = np.empty((len(positions), n_reads), dtype=bool)
    for s, position in enumerate(positions):
        covers = on_reference & (p_start <= position) & (position < p_start + lengths)
        found = covers.any(axis=1)
        j = covers.argmax(axis=1)
        m = is_m[pick, j]
        inside = q_start[pick, j] + np.where(m, position - p_start[pick, j], 0)
        outside = np.where(position < aln.start, lead - (aln.start - position),
                           read_lengths - trail + (position - end))
        projection[s] = np.where(found, inside, outside)
        on_m[s] = found & m

    indel = np.empty((len(intervals), n_reads), dtype=bool)
    clip = np.empty((len(intervals), n_reads), dtype=bool)
    gap = on_reference & ~is_m
    for s, (lo, hi) in enumerate(intervals):
        over_gap = gap & (lo < p_start + lengths) & (p_start < hi)
        over_insertion = (ops == _INS) & (lo < p_start) & (p_start < hi)
        indel[s] = (over_gap | over_insertion).any(axis=1)
        clip[s] = (((ops[:, 0] == _SOFT) & (aln.start > lo))
                   | ((ops[pick, last] == _SOFT) & (p_start[pick, last] < hi)))
    return projection, on_m, indel, clip


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


def _place_side(matrix: np.ndarray, lengths: np.ndarray, flank: FlankRecord,
                projection: np.ndarray, on_m: np.ndarray, indel: np.ndarray,
                clip: np.ndarray, window_len: int
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place one flank in every read: run-side index (-1), path code, drop code."""
    n_reads = len(lengths)
    is_up = flank.is_upstream
    k = len(flank.bases)
    pattern, mask = _pattern(flank.bases, is_up, flank.masked_offsets)
    far_end = projection - (k - 1) if is_up else projection + (k - 1)
    covered = ((np.minimum(projection, far_end) >= 0)
               & (np.maximum(projection, far_end) < lengths))
    chosen = np.full(n_reads, -1, dtype=np.int64)
    path = np.zeros(n_reads, dtype=np.int8)
    drop = np.zeros(n_reads, dtype=np.int8)

    # Fast path: the projection, with no indel or clip over the flank.
    fast = np.flatnonzero(on_m & ~indel & ~clip & covered)
    b = projection[fast]
    ok = _exact_match(matrix, lengths, fast, b, pattern, mask, is_up)
    ok &= matrix[fast, b] != _G
    fast = fast[ok]
    chosen[fast] = projection[fast]
    path[fast] = _FAST

    # Window search for every other read.
    rest_mask = np.ones(n_reads, dtype=bool)
    rest_mask[fast] = False
    rest = np.flatnonzero(rest_mask)
    if rest.size == 0:
        return chosen, path, drop
    width = 2 * _LOCAL_WINDOW + 1
    shift = np.arange(-_LOCAL_WINDOW, _LOCAL_WINDOW + 1)
    cand = (projection[rest][:, None] + shift[None, :]).ravel()
    cand_rows = np.repeat(rest, width)
    dist = _banded_distance(matrix, lengths, cand_rows, cand, pattern, mask, is_up)
    inside = (cand >= 0) & (cand < lengths[cand_rows])
    is_g = np.zeros(cand.shape, dtype=bool)
    is_g[inside] = matrix[cand_rows[inside], cand[inside]] == _G
    score = dist + (np.tile(shift, rest.size) != 0)
    nearness = cand if is_up else -cand
    # Order: score, then non-G, then nearest the run. Positions with more
    # than one flank edit cannot win.
    span = 2 * (window_len + int(lengths.max()) + 2 * _LOCAL_WINDOW + 4)
    key = (score * 2 + is_g) * span - (nearness - nearness.min())
    key = np.where(dist < _FAR, key, np.iinfo(np.int64).max)
    key = key.reshape(rest.size, width)
    best = key.argmin(axis=1)
    pick = np.arange(rest.size)
    found = key[pick, best] != np.iinfo(np.int64).max
    best_b = cand.reshape(rest.size, width)[pick, best]
    best_g = is_g.reshape(rest.size, width)[pick, best]

    missing = rest[~found]
    drop[missing] = np.where(covered[missing], _NOT_PLACED, _NOT_COVERED)
    drop[rest[found & best_g]] = _G_AT_RUN_SIDE
    placed = found & ~best_g
    rows = rest[placed]
    b = best_b[placed]
    chosen[rows] = b
    cigar_path = (b == projection[rows]) & on_m[rows] & indel[rows] & ~clip[rows]
    path[rows] = np.where(cigar_path, _CIGAR, _LOCAL)
    return chosen, path, drop


def place_flanks(
    read_seqs: Sequence[str],
    cigars: Sequence[Sequence[tuple[int, int]]],
    reference_starts: np.ndarray | Sequence[int],
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
    reference_starts : array of int
        0-based reference start of each alignment.
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

    flanks = (upstream, downstream)
    first_index = []
    run_side_index = []
    for flank in flanks:
        us = [window_index(r, ref_orientation=ref_orientation,
                           window_start=reference_window_start, window_len=window_len)
              for r in flank.ref_positions]
        if any(b - a != 1 for a, b in zip(us, us[1:], strict=False)):
            raise ValueError("flank ref_positions do not run along the polyG orientation")
        first_index.append(us[0])
        run_side_index.append(us[flank.run_side_offset])
    middle = (run_side_index[0] + run_side_index[1]) / 2
    intervals = [(lo, lo + len(f.bases)) for lo, f in zip(first_index, flanks, strict=True)]

    matrix, lengths = _polyg_matrix(read_seqs, minus)
    width = matrix.shape[1]
    starts = np.asarray(reference_starts, dtype=np.int64)

    aln = _polyg_cigar_arrays(cigars, starts, ref_orientation=ref_orientation,
                              window_start=reference_window_start, window_len=window_len)
    _shift_block_indels(aln, matrix, lengths, polyg_reference, middle)
    projection, on_m, indel, clip = _alignment_geometry(aln, lengths, run_side_index,
                                                        intervals)
    text = matrix.tobytes().decode("latin-1")

    up_b, up_path, up_drop = _place_side(matrix, lengths, upstream, projection[0], on_m[0],
                                         indel[0], clip[0], window_len)
    dn_b, dn_path, dn_drop = _place_side(matrix, lengths, downstream, projection[1],
                                         on_m[1], indel[1], clip[1], window_len)

    placed = (up_b >= 0) & (dn_b >= 0)
    crossed = placed & (dn_b <= up_b)
    kept = placed & ~crossed
    tracts = np.full(n_reads, None, dtype=object)
    rows = np.flatnonzero(kept)
    if rows.size:
        lo = (rows * width + up_b[rows] + 1).tolist()
        hi = (rows * width + dn_b[rows]).tolist()
        tracts[rows] = [text[a:b] for a, b in zip(lo, hi, strict=True)]

    return FlankPlacement(
        tracts=tracts,
        kept=kept,
        upstream_path=_PATH_NAMES[up_path],
        downstream_path=_PATH_NAMES[dn_path],
        upstream_drop=_SIDE_DROP_NAMES[up_drop],
        downstream_drop=_SIDE_DROP_NAMES[dn_drop],
        flanks_crossed=crossed,
        upstream_run_side=up_b,
        downstream_run_side=dn_b,
    )
