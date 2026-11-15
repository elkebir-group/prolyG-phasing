"""Per-donor flank construction: the pair of flanks that maximizes one objective.

For each locus, the flanks are the pair of 10-base strings ``(u, v)`` that
maximizes, over the matched normal's reads,

    F(u, v) = sum over reads of max(0, the value of the read's best placement),

where the match, the marks, the placement and the value are those of
:mod:`prolyg_phasing.io._flank_placement`: a placement holds a read base
that the aligner aligns inside the BED interval. A read with no placement,
or whose best value is negative, adds 0. The constraints:

- The last base of ``u`` and the first base of ``v`` are not G.
- ``u`` does not end with three G and then its last base; ``v`` does not
  start with its first base and then three G. Joining such a piece to the
  tract raises each read's value, so the tract must not stop in front of it.
- Each of ``u`` and ``v`` is a candidate: it occurs exactly at a tract edge
  of a placement in at least one read. A ``u`` candidate is the 10 bases
  that end at an upstream tract edge ``a`` (a non-G base followed by a G)
  for which the read has a downstream tract edge ``b`` (a non-G base
  preceded by a G) with room for a downstream match (at least 8 bases from
  ``b`` to the read end) and a marked base between ``a`` and ``b``. A ``v``
  candidate is the same on the other side: the 10 bases that start at a
  downstream tract edge with an upstream tract edge before it, at least 8
  bases from the read start, and a marked base between them. Only these
  edges can hold a flank of a placement.

**The search is exact over the candidates.** All values are integers in
half units. The downstream side runs the upstream code on per-read
reversed reads with reversed strings: ``v`` matches read ``r`` at start
``b`` exactly when ``reverse(v)`` matches ``reverse(r)`` at end
``len(r) - 1 - b``.

1. **Index.** Per side, the 12-base window that ends at each tract edge
   that can hold a flank of a placement (as for the candidates) is stored
   once per distinct window, with the read positions that show it. No
   other edge is part of a placement, so dropping them changes no value.
   Every candidate's matches to the distinct windows, with their lowest
   cost, come from listing each window's neighbors under the match rule.
2. **Bounds.** ``M[r, a]`` is the best tract score that starts after edge
   ``a`` of read ``r`` and ends at a downstream tract edge past the first
   marked base after ``a``. A candidate's position bound sums ``max(0, M -
   cost)`` over its matched positions, weighted by read count. Its
   per-read bound takes the largest such term per read. A pair's joint
   bound sums, per read, the smaller of its two per-read bounds. Each
   bound is at least F of every pair it covers: each covers every
   placement the reads allow, and more.
3. **Search.** A first best value comes from the 40 candidates per side
   with the highest position bound. Then every pair whose bounds reach the
   current best is evaluated exactly. All pairs that reach the optimum are
   kept.

The optimal pair is then placed in every read by
:func:`~prolyg_phasing.io._flank_placement.place_polyg_reads`, the
function extraction uses, and the summed value must equal the optimum.

**Reference coordinates.** The reads' alignments give each flank's place
in the reference; the reference sequence is not searched, so a donor flank
can differ from the reference by any number of edits. In each kept read the
flank occupies known read bases at the best placement. The aligner maps the
flank's outer base and its run-side base to reference coordinates. The
flank's position is the most common (outer, run-side) pair over the kept
reads that give both (see :func:`flank_records_from_alignments`). The
phasing window starts past the outer base. The tract never uses these coordinates.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Iterator, Sequence
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

from prolyg_phasing.io._anchors import build_anchors, get_reference_window, revcomp
from prolyg_phasing.io._flank_placement import (
    FLANK_LEN,
    FLANK_PLACE_VALUES,
    MAX_MATCH_LEN,
    MIN_MATCH_LEN,
    PAD,
    SHAPES,
    SUBSTITUTION_COST,
    WINDOW_PLACE_VALUES,
    WINDOW_RECOVERED,
    WINDOW_UNRECOVERABLE,
    FlankRecord,
    G,
    N,
    bed_marks,
    code_matrix,
    decode,
    distinct_reads,
    edge_windows,
    flank_outer_bases,
    mark_bounds,
    place_polyg_reads,
    polyg_reads,
    polyg_reference_coordinates,
    tract_scores,
)
from prolyg_phasing.io.bam import (
    REF_WINDOW_PAD,
    _load_bed,
    filtered_alignments,
    new_locus_qc,
)

# Candidates per side whose pairs give the first best value.
_SEED_CANDIDATES = 40

# Candidates per batch when match lists are expanded.
_LIST_BATCH = 64

# Stands in for the bound when every match of a window is listed.
_LIST_ALL = 1 << 20

# Largest cell count of one 0/1 matrix block in the joint bound.
_JOINT_BOUND_CELLS = 20_000_000

# Memory one worker process can use at a deep locus, with headroom.
_WORKER_MEMORY_BYTES = 1 << 30

# Share of the available memory that the workers may use together.
_WORKER_MEMORY_SHARE = 0.5

# Share of the usable cores that is left free for other work.
_FREE_CORE_SHARE = 0.125

# The read filters of the extraction, at their defaults.
_FILTERS = {"min_mapq": 40, "drop_chimeric": True, "max_tlen": 1000}

SIDES = ("upstream", "downstream")

# Sentinel tract score: below every real one.
_NEG = -(1 << 30)


# ---------------------------------------------------------------------------
# The candidate index of one side
# ---------------------------------------------------------------------------


def window_neighbors(windows: np.ndarray, bound: np.ndarray
                     ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Every 10-base string that matches the end of a window at a cost below its bound.

    Yields ``(string code, window index, bound - cost)`` arrays; a string
    can repeat for one window, once per shape it matches by. The match
    rule is `SHAPES` with the costs of :mod:`._flank_placement`: the
    gapless shape allows one substitution, a gap shape none. A flank base
    that a deletion shape leaves out of the read takes every base. A read
    N matches no flank base, so a window with an N matches only by a
    substitution there.
    """
    ids = np.arange(len(windows))
    for shape in SHAPES:
        live = bound > shape.cost
        if not live.any():
            continue
        sub = windows[:, MAX_MATCH_LEN - shape.length:]
        inside = (sub != PAD).all(axis=1) & live
        core = sub[:, list(shape.read_slots)]
        flank_slots = list(shape.flank_slots)
        is_n = core == N
        n_count = is_n.sum(axis=1)
        core0 = np.where(is_n, 0, core).astype(np.int64)
        base_code = core0 @ FLANK_PLACE_VALUES[flank_slots]
        missing = list(range(shape.at, shape.at + shape.gap)) if shape.kind == "deletion" else []
        fills = [sum(int(c) * int(FLANK_PLACE_VALUES[p]) for c, p in zip(f, missing, strict=True))
                 for f in np.ndindex(*([4] * len(missing)))] if missing else [0]
        exact = inside & (n_count == 0)
        if exact.any():
            for fill in fills:
                yield base_code[exact] + fill, ids[exact], bound[exact] - shape.cost
        if shape.kind != "none":
            continue
        for j, p in enumerate(flank_slots):
            ok = inside & (bound > SUBSTITUTION_COST) & ((n_count == 0)
                                                        | ((n_count == 1) & is_n[:, j]))
            if not ok.any():
                continue
            for c in range(4):
                other = core[ok, j] != c
                code = base_code[ok] + (c - core0[ok, j]) * FLANK_PLACE_VALUES[p]
                yield code[other], ids[ok][other], (bound[ok] - SUBSTITUTION_COST)[other]


def _split_lists(n: int, k: np.ndarray, r: np.ndarray, p: np.ndarray, c: np.ndarray
                 ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per-list (reads, positions, costs) arrays from rows sorted by list index `k`."""
    bounds = np.searchsorted(k, np.arange(n + 1))
    r, p, c = r.astype(np.int32), p.astype(np.uint8), c.astype(np.uint8)
    return [(r[bounds[i]:bounds[i + 1]], p[bounds[i]:bounds[i + 1]], c[bounds[i]:bounds[i + 1]])
            for i in range(n)]


def _reverse_rows(marks: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Each row of `marks` reversed within its read's length; False past it."""
    index = lengths[:, None] - 1 - np.arange(marks.shape[1])[None, :]
    return np.where(index >= 0, np.take_along_axis(marks, np.maximum(index, 0), axis=1), False)


class _ReadSide:
    """The reads of one side: forward for ``u``, reversed per read for ``v``.

    Attributes
    ----------
    mat : ndarray of uint8, shape (n_reads, width)
        Base codes, PAD past each read.
    lengths : ndarray of int64
    weights : ndarray of int64
        The count of each distinct read.
    scores : ndarray of int64
        :func:`tract_scores` of `mat`.
    best_tract : ndarray of int64
        ``M[r, a]``: the best tract score from edge ``a`` to a closing edge
        with room for a downstream match and a marked base in between, at
        least 0; 0 where ``a`` is not an edge.
    usable : ndarray of bool
        The edges that can hold this side's flank in a placement: an
        opening edge with such a closing edge after it.
    windows : ndarray
        The distinct 12-base edge windows.
    candidates : ndarray of int64
        Sorted candidate codes, in side orientation.
    position_bound : ndarray of int64
        Each candidate's position-summed bound.
    """

    def __init__(self, strings: Sequence[str], marks: np.ndarray, weights: np.ndarray, *,
                 reverse: bool):
        self.mat, self.lengths = code_matrix(strings, reverse=reverse)
        self.marks = _reverse_rows(marks, self.lengths) if reverse else marks
        self.n_reads, self.width = self.mat.shape
        if self.n_reads >= 1 << 24 or self.width >= 256:
            raise ValueError("a locus holds fewer than 2^24 distinct reads of at most "
                             "255 bases")
        self.weights = np.asarray(weights, dtype=np.int64)
        self.scores = tract_scores(self.mat)
        self._bound_tracts()
        self._index()

    def _bound_tracts(self) -> None:
        mat, lengths, width = self.mat, self.lengths, self.width
        is_g = mat == G
        prev_g = np.zeros_like(is_g)
        prev_g[:, 1:] = is_g[:, :-1]
        next_g = np.zeros_like(is_g)
        next_g[:, :-1] = is_g[:, 1:]
        col = np.arange(width)[None, :]
        closing = (~is_g) & (mat != PAD) & prev_g & (col <= lengths[:, None] - MIN_MATCH_LEN)
        # best_close[:, j]: the best closing score at a closing edge >= j.
        best_close = np.full((self.n_reads, width + 1), _NEG, dtype=np.int64)
        best_close[:, :width] = np.maximum.accumulate(
            np.where(closing, self.scores[:, :width], _NEG)[:, ::-1], axis=1)[:, ::-1]
        next_mark, _ = mark_bounds(self.marks)
        gate = np.take_along_axis(best_close, np.minimum(next_mark + 1, width), axis=1)
        bound = np.maximum(gate - self.scores[:, 1:], 0)
        opening = (~is_g) & next_g & (col >= MIN_MATCH_LEN - 1) & (col < lengths[:, None])
        self.usable = opening & (gate > _NEG)
        self.best_tract = np.where(opening, bound, 0)

    def _index(self) -> None:
        r, a = np.nonzero(self.usable)
        windows = edge_windows(self.mat, r, a)
        _, first, inverse = np.unique(windows.astype(np.int64) @ WINDOW_PLACE_VALUES,
                                      return_index=True, return_inverse=True)
        inverse = inverse.ravel()
        n_windows = len(first)
        self.windows = windows[first]
        order = np.argsort(inverse, kind="stable")
        self.position_read = r[order].astype(np.int32)
        self.position_edge = a[order].astype(np.int16)
        self.window_start = np.r_[0, np.cumsum(np.bincount(inverse, minlength=n_windows))]

        # Candidates: the last 10 window bases, all A/C/G/T, and not three G
        # before the edge base.
        last = self.windows[:, MAX_MATCH_LEN - FLANK_LEN:]
        ok = (last < N).all(axis=1) & ~(last[:, -4:-1] == G).all(axis=1)
        self.candidates = np.unique(last[ok].astype(np.int64) @ FLANK_PLACE_VALUES)
        n_cand = len(self.candidates)

        # Every (candidate, window) match with its lowest cost.
        xs, ws, costs = [], [], []
        if n_cand:
            for code, window, rest in window_neighbors(
                    self.windows, np.full(n_windows, _LIST_ALL, dtype=np.int64)):
                j = np.minimum(np.searchsorted(self.candidates, code), n_cand - 1)
                hit = self.candidates[j] == code
                if hit.any():
                    xs.append(j[hit])
                    ws.append(window[hit])
                    costs.append(_LIST_ALL - rest[hit])
        if xs:
            key = np.sort((np.concatenate(xs).astype(np.int64) * n_windows
                           + np.concatenate(ws)) * 16 + np.concatenate(costs))
            pair = key >> 4
            key = key[np.r_[True, pair[1:] != pair[:-1]]]
            self.match_candidate = (key >> 4) // n_windows
            self.match_window = (key >> 4) % n_windows
        else:
            key = np.zeros(0, dtype=np.int64)
            self.match_candidate = self.match_window = key
        self.match_cost = key & 15
        self.candidate_start = np.searchsorted(self.match_candidate, np.arange(n_cand + 1))

        # Position-summed bound: sum over matched positions of w_r * max(0, M - cost).
        best = self.best_tract[self.position_read, self.position_edge].astype(np.int64)
        weight = self.weights[self.position_read]
        window_of_position = np.repeat(np.arange(n_windows), np.diff(self.window_start))
        bound = np.zeros(n_cand, dtype=np.int64)
        for c in np.unique(self.match_cost):
            per_window = np.bincount(window_of_position,
                                     weights=weight * np.maximum(best - c, 0),
                                     minlength=n_windows)
            m = self.match_cost == c
            bound += np.bincount(self.match_candidate[m], weights=per_window[self.match_window[m]],
                                 minlength=n_cand).astype(np.int64)
        self.position_bound = bound

    def match_rows(self, cidx: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(list index, read, edge, cost) for the candidates `cidx`, sorted by the first three."""
        cidx = np.asarray(cidx, dtype=np.int64)
        lo, hi = self.candidate_start[cidx], self.candidate_start[cidx + 1]
        n_rows = hi - lo
        if not n_rows.sum():
            z = np.zeros(0, dtype=np.int64)
            return z, z, z, z
        k = np.repeat(np.arange(len(cidx)), n_rows)
        rows = np.repeat(lo - np.cumsum(n_rows) + n_rows, n_rows) + np.arange(n_rows.sum())
        window, cost = self.match_window[rows], self.match_cost[rows]
        n_pos = self.window_start[window + 1] - self.window_start[window]
        p = (np.repeat(self.window_start[window] - np.cumsum(n_pos) + n_pos, n_pos)
             + np.arange(int(n_pos.sum())))
        k, cost = np.repeat(k, n_pos), np.repeat(cost, n_pos)
        # One packed key, sorted once: list (27 bits) | read (24) | edge (8) | cost (4).
        key = ((((k << 24) | self.position_read[p].astype(np.int64)) << 8
                | self.position_edge[p].astype(np.int64)) << 4) | cost
        key.sort()
        return key >> 36, (key >> 12) & 0xFFFFFF, (key >> 4) & 0xFF, key & 0xF


# ---------------------------------------------------------------------------
# The pair search over one locus
# ---------------------------------------------------------------------------


class _LocusSearch:
    """The two sides of one locus, and F over pairs of their match lists."""

    def __init__(self, strings: Sequence[str], marks: np.ndarray, weights: Sequence[int]):
        self.strings = list(strings)
        self.weights = np.asarray(weights, dtype=np.int64)
        self.u = _ReadSide(self.strings, marks, self.weights, reverse=False)
        self.v = _ReadSide(self.strings, marks, self.weights, reverse=True)
        self.last_mark = mark_bounds(marks)[1]
        self.n_reads = self.u.n_reads
        self.lengths = self.u.lengths
        self.scores = self.u.scores
        self.width = self.u.width

    def u_lists(self, cidx: np.ndarray) -> list:
        k, r, a, c = self.u.match_rows(cidx)
        return _split_lists(len(cidx), k, r, a, c)

    def v_lists(self, cidx: np.ndarray) -> list:
        """Match lists of `v` candidates; positions are forward starts ``b``."""
        k, r, a, c = self.v.match_rows(cidx)
        return _split_lists(len(cidx), k, r, self.lengths[r] - 1 - a, c)

    def per_read_bound(self, lst, side: str) -> int:
        """``sum_r w_r max(0, max over the list's positions in r of M - cost)``."""
        r, p, c = (np.asarray(t, np.int64) for t in lst)
        if not len(r):
            return 0
        if side == "u":
            val = np.maximum(self.u.best_tract[r, p] - c, 0)
        else:
            val = np.maximum(self.v.best_tract[r, self.lengths[r] - 1 - p] - c, 0)
        best = np.zeros(self.n_reads, dtype=np.int64)
        np.maximum.at(best, r, val)
        return int((best * self.weights).sum())

    def objective_block(self, ul, vls) -> np.ndarray:
        """F(u, v) for one ``u`` match list against several ``v`` match lists.

        Per read: max(0, max over ``b`` of ``scores[b] - cost_v(b) -
        min{scores[a + 1] + cost_u(a) : a < last_mark[b]}``), so that a
        marked base lies between ``a`` and ``b``.
        """
        out = np.zeros(len(vls), dtype=np.int64)
        ru, au, cu = (np.asarray(t, np.int64) for t in ul)
        if not len(ru) or not vls:
            return out
        far = 1 << 40
        rows, loc = np.unique(ru, return_inverse=True)
        loc = loc.ravel()
        lut = np.full(self.n_reads, -1, dtype=np.int64)
        lut[rows] = np.arange(len(rows))
        arr = np.full((len(rows), self.width + 1), far, dtype=np.int64)
        np.minimum.at(arr, (loc, au + 1), self.scores[ru, au + 1] + cu)
        prefix_min = np.minimum.accumulate(arr, axis=1)
        vid = np.concatenate([np.full(len(v[0]), i) for i, v in enumerate(vls)])
        rv = np.concatenate([v[0] for v in vls]).astype(np.int64)
        bv = np.concatenate([v[1] for v in vls]).astype(np.int64)
        cv = np.concatenate([v[2] for v in vls]).astype(np.int64)
        lv = lut[rv]
        keep = lv >= 0
        vid, rv, bv, cv, lv = vid[keep], rv[keep], bv[keep], cv[keep], lv[keep]
        gate = self.last_mark[rv, bv]
        pm = np.where(gate >= 0, prefix_min[lv, np.maximum(gate, 0)], far)
        keep = pm < far // 2
        vid, rv, bv, cv, pm = vid[keep], rv[keep], bv[keep], cv[keep], pm[keep]
        if not len(vid):
            return out
        val = self.scores[rv, bv] - cv - pm
        key = vid * self.n_reads + rv
        o = np.argsort(key, kind="stable")
        key, val = key[o], val[o]
        st = np.nonzero(np.r_[True, key[1:] != key[:-1]])[0]
        best = np.maximum(np.maximum.reduceat(val, st), 0)
        k0 = key[st]
        out += np.bincount(k0 // self.n_reads, weights=best * self.weights[k0 % self.n_reads],
                           minlength=len(vls)).astype(np.int64)
        return out


def _collect(search: _LocusSearch, cidx: np.ndarray, side: str, keep: int) -> dict:
    """Match lists of the candidates whose per-read bound reaches `keep`, by index in `cidx`."""
    kept = {}
    for b0 in range(0, len(cidx), _LIST_BATCH):
        part = cidx[b0:b0 + _LIST_BATCH]
        lists = search.u_lists(part) if side == "u" else search.v_lists(part)
        for j, lst in enumerate(lists):
            if search.per_read_bound(lst, side) >= keep:
                kept[b0 + j] = lst
    return kept


def _dedupe(lists: list) -> tuple[list, np.ndarray]:
    """Group candidates with identical match lists; F depends only on the list."""
    signature: dict[bytes, int] = {}
    distinct, inverse = [], []
    for lst in lists:
        key = lst[0].tobytes() + lst[1].tobytes() + lst[2].tobytes()
        if key not in signature:
            signature[key] = len(distinct)
            distinct.append(lst)
        inverse.append(signature[key])
    return distinct, np.array(inverse, dtype=np.int64)


def _read_bounds(search: _LocusSearch, lists: list, side: str) -> np.ndarray:
    """Per-read bound of each list, ``max(0, max over its positions of M - cost)``,
    as a dense (n_lists, n_reads) int16 matrix."""
    out = np.zeros((len(lists), search.n_reads), dtype=np.int16)
    best_tract = search.u.best_tract if side == "u" else search.v.best_tract
    for i, (r, p, c) in enumerate(lists):
        if not len(r):
            continue
        r = r.astype(np.int64)
        a = p.astype(np.int64) if side == "u" else search.lengths[r] - 1 - p.astype(np.int64)
        np.maximum.at(out[i], r,
                      np.maximum(best_tract[r, a] - c.astype(np.int64), 0).astype(np.int16))
    return out


def _merge_columns(bounds: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge identical columns of bounds, summing their weights.

    Columns are grouped by a random linear hash; the grouping is checked
    column by column, so a hash collision raises instead of merging two
    different columns.
    """
    rng = np.random.default_rng(0)
    h = (rng.integers(1, 1 << 62, size=bounds.shape[0], dtype=np.int64)[:, None]
         * bounds.astype(np.int64)).sum(0)
    _, first, inv = np.unique(h, return_index=True, return_inverse=True)
    inv = inv.ravel()
    if not (bounds[:, first[inv]] == bounds).all():
        raise RuntimeError("hash collision while merging read columns")
    return bounds[:, first], np.bincount(inv, weights=w)


def _joint_bound(search: _LocusSearch, ul: list, vl: list) -> np.ndarray:
    """``joint[i, j] = sum_r w_r min(bu_i[r], bv_j[r]) >= F(u_i, v_j)``.

    u_hits read's value is at most its ``u``-side bound and at most its
    ``v``-side bound. Per read column, ``min(a, b)`` is the sum over the
    column's distinct positive values ``t_1 < t_2 < ...`` of ``[a >= t_k]
    [b >= t_k] (t_k - t_{k-1})``. So the bound is one product of 0/1 matrices,
    done in blocks of thresholds. Every term is an integer and the sums
    stay far below 2^53, so the float product is exact.
    """
    nu = len(ul)
    u_bounds, v_bounds = _read_bounds(search, ul, "u"), _read_bounds(search, vl, "v")
    live = (u_bounds.max(0) > 0) & (v_bounds.max(0) > 0)
    joint = np.zeros((nu, len(vl)), dtype=np.float64)
    if not live.any():
        return joint.astype(np.int64)
    bounds, w = _merge_columns(np.vstack([u_bounds[:, live], v_bounds[:, live]]),
                               search.weights[live].astype(np.float64))
    ordered = np.sort(bounds, axis=0)
    new = np.vstack([ordered[:1] > 0, (ordered[1:] != ordered[:-1]) & (ordered[1:] > 0)])
    col, row = np.nonzero(new.T)
    t = ordered[row, col].astype(np.int64)
    prev = np.r_[0, t[:-1]]
    prev[np.r_[True, col[1:] != col[:-1]]] = 0
    d = (t - prev) * w[col]
    step = max(1, _JOINT_BOUND_CELLS // bounds.shape[0])
    for k0 in range(0, len(t), step):
        c_, t_ = col[k0:k0 + step], t[k0:k0 + step]
        u_hits = (bounds[:nu, c_] >= t_) * d[k0:k0 + step]
        v_hits = (bounds[nu:, c_] >= t_).astype(np.float64)
        joint += u_hits @ v_hits.T
    return np.rint(joint).astype(np.int64)


def _best_pairs(search: _LocusSearch, ul: list, vl: list, floor: int
                ) -> tuple[int, list[tuple[int, int]]]:
    """All list pairs with F >= best, where best starts at `floor`.

    A pair is evaluated only if its joint bound reaches the current best.
    """
    if not ul or not vl:
        return floor, []
    joint = _joint_bound(search, ul, vl)
    best, ties = floor, []
    for iu in np.argsort(-joint.max(1), kind="stable"):
        if joint[iu].max() < best:
            break
        vi = np.nonzero(joint[iu] >= best)[0]
        vi = vi[np.argsort(-joint[iu, vi], kind="stable")]
        f = search.objective_block(ul[iu], [vl[j] for j in vi])
        m = int(f.max())
        if m > best:
            best, ties = m, []
        if m >= best:
            ties += [(int(iu), int(vi[j])) for j in np.nonzero(f == best)[0]]
    return best, ties


def _search_pairs(search: _LocusSearch, ui: np.ndarray, vi: np.ndarray, floor: int
                  ) -> tuple[int, list[tuple[int, int]]]:
    """Best F over the candidate pairs ``ui x vi`` with F >= `floor`, and every tied
    pair as candidate indices. A candidate whose per-read bound is below `floor`
    cannot reach it and takes no part."""
    ku = _collect(search, ui, "u", floor)
    kv = _collect(search, vi, "v", floor)
    a, b = sorted(ku), sorted(kv)
    ul, iu = _dedupe([ku[i] for i in a])
    vl, iv = _dedupe([kv[i] for i in b])
    best, ties = _best_pairs(search, ul, vl, floor)
    pairs = set()
    for ru, rv in ties:
        for i in np.nonzero(iu == ru)[0]:
            for j in np.nonzero(iv == rv)[0]:
                pairs.add((int(ui[a[i]]), int(vi[b[j]])))
    return best, sorted(pairs)


@dataclasses.dataclass(frozen=True)
class FlankPair:
    """The optimal flank pair of one locus.

    Attributes
    ----------
    upstream, downstream : str
        The pair, in polyG orientation. Of several optimal pairs, the first
        in lexicographic order of ``(upstream, downstream)``.
    objective_half_units : int
        F of the pair, in half units.
    optimal_pairs : tuple of (str, str)
        Every optimal pair, in lexicographic order.
    n_reads : int
        The reads, with repeats.
    n_reads_placed : int
        The reads with at least one placement under the pair, of any value.
    """

    upstream: str
    downstream: str
    objective_half_units: int
    optimal_pairs: tuple[tuple[str, str], ...]
    n_reads: int
    n_reads_placed: int


def solve_flank_pair(reads: Sequence[str], marks: np.ndarray) -> FlankPair | None:
    """The pair of flanks that maximizes F over `reads` (polyG orientation, upper-case).

    `marks` holds one row per read, in polyG read order, at least as wide
    as the longest read: True at a marked base. Returns None when no pair
    reaches F > 0. Raises RuntimeError when the extraction's placement of
    the optimal pair does not give the optimum: the search and the
    placement would then disagree on a read's tract.
    """
    if not len(reads):
        return None
    strings, distinct_marks, inverse = distinct_reads(reads, marks)
    weights = np.bincount(inverse, minlength=len(strings)).astype(np.int64)
    search = _LocusSearch(strings, distinct_marks, weights)
    if not len(search.u.candidates) or not len(search.v.candidates):
        return None
    ub_u, ub_v = search.u.position_bound, search.v.position_bound
    seed_u = np.argsort(-ub_u, kind="stable")[:_SEED_CANDIDATES]
    seed_v = np.argsort(-ub_v, kind="stable")[:_SEED_CANDIDATES]
    first_best, _ = _search_pairs(search, seed_u, seed_v, 1)
    floor = max(first_best, 1)
    best, pairs = _search_pairs(search, np.nonzero(ub_u >= floor)[0],
                                np.nonzero(ub_v >= floor)[0], floor)
    if not pairs:
        return None
    optimal = sorted((decode(int(search.u.candidates[i])),
                      decode(int(search.v.candidates[j]))[::-1]) for i, j in pairs)
    upstream, downstream = optimal[0]
    placement = place_polyg_reads(strings, upstream, downstream, distinct_marks)
    total = int((weights * np.maximum(placement.value, 0)).sum())
    if total != best:
        raise RuntimeError(f"the placement of the optimal pair ({upstream}, {downstream}) "
                           f"gives F = {total} half units, the search {best}")
    return FlankPair(upstream, downstream, best, tuple(optimal), int(weights.sum()),
                     int(weights[placement.placed].sum()))


# ---------------------------------------------------------------------------
# One locus
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LocusReads:
    """One locus's reference window and its filtered reads in polyG orientation.

    `reads` are the alignments that overlap the BED interval and pass the
    extraction's read filters. `repeat_start` and `repeat_end` are the BED
    interval's window indices in polyG orientation. ``ref_coords[r, i]`` is
    the reference coordinate that the alignment of read ``r`` gives to its
    base at polyG index ``i``; -1 where that base is soft-clipped, in an
    insertion, or past the read. `bed_start` and `bed_end` are the BED
    interval.
    """

    name: str
    ref_orientation: str
    window_start: int
    polyg_window: str
    repeat_start: int
    repeat_end: int
    reads: list[str]
    ref_coords: np.ndarray
    bed_start: int
    bed_end: int

    @property
    def marks(self) -> np.ndarray:
        """The marked bases of each read: aligned inside the BED interval."""
        return bed_marks(self.ref_coords, self.bed_start, self.bed_end)


def locus_reads(bam: pysam.AlignmentFile, name: str, chrom: str, bed_start: int,
                bed_end: int) -> LocusReads | None:
    """Reference window and filtered reads at one locus.

    Returns None when no read rebuilds the reference window. Raises
    ValueError, naming the locus, when the window holds no edge base on a
    side: the G run reaches the end of the window.
    """
    window_start = bed_start - REF_WINDOW_PAD
    window = get_reference_window(bam, chrom, window_start, bed_end + REF_WINDOW_PAD,
                                  min_mapq=_FILTERS["min_mapq"])
    if window is None:
        return None
    anchors = build_anchors(window, REF_WINDOW_PAD, bed_end - bed_start, k=FLANK_LEN)
    if anchors.status != "anchorable":
        raise ValueError(f"{name}: the reference window holds no edge base and flank on "
                         f"each side ({anchors.status})")
    polyg_window = revcomp(window) if anchors.ref_orientation == "-" else window
    records = [r for r, _ in filtered_alignments(bam, chrom, bed_start, bed_end, bed_start,
                                                 bed_end, qc=new_locus_qc(), **_FILTERS)]
    reads = polyg_reads([r.query_sequence for r in records], anchors.ref_orientation)
    return LocusReads(name, anchors.ref_orientation, window_start, polyg_window,
                      REF_WINDOW_PAD, REF_WINDOW_PAD + bed_end - bed_start, reads,
                      polyg_reference_coordinates(records, anchors.ref_orientation),
                      bed_start, bed_end)


def modal_pair(outer: np.ndarray, run: np.ndarray, toward_run: int
               ) -> tuple[int, int, int] | None:
    """The most common (outer, run-side) coordinate pair, with its read count.

    `outer` and `run` hold one pair per read. On a tie, the pair whose
    run-side coordinate is nearer the G run wins; if still tied, the pair
    whose outer coordinate is nearer the G run. Nearer means a larger
    ``toward_run * coordinate``: `toward_run` is +1 when the G run lies at
    larger reference coordinates, -1 when it lies at smaller ones.

    Returns ``(outer, run, count)``, or None when there is no pair.
    """
    if outer.size == 0:
        return None
    pairs, counts = np.unique(np.stack([outer, run], axis=1), axis=0, return_counts=True)
    top = pairs[counts == counts.max()]
    # Sort by the run-side key, then the outer key, both nearest the G run first.
    best = top[np.lexsort((-toward_run * top[:, 0], -toward_run * top[:, 1]))[0]]
    return int(best[0]), int(best[1]), int(counts.max())


@dataclasses.dataclass(frozen=True)
class FlankPosition:
    """One side's flank record, with the reads that voted for its position.

    Attributes
    ----------
    record : FlankRecord or None
        None when no kept read gives both of the side's coordinates.
    n_reads_positioned : int
        Kept reads whose alignment gives both the outer and the run-side
        base a coordinate.
    n_reads_at_position : int
        Those of them that give the chosen pair.
    """

    record: FlankRecord | None
    n_reads_positioned: int
    n_reads_at_position: int


def flank_records_from_alignments(loc: LocusReads, pair: FlankPair
                                  ) -> tuple[FlankPosition, FlankPosition]:
    """The upstream and downstream flank positions of a solved pair, from the alignments.

    The kept reads are those whose best placement under the pair has a
    value of at least 0. In each one, the flank occupies the read bases
    from its outer base to its run-side base at the best placement
    (:func:`~prolyg_phasing.io._flank_placement.flank_outer_bases`). The
    alignment gives each of these two bases a reference coordinate,
    unless the base is soft-clipped or in an insertion. Only a read that
    gives both coordinates votes, with its (outer, run-side) pair. The
    flank's position is the most common pair. On a tie, the pair whose
    run-side coordinate is nearer the G run wins; if still tied, the pair
    whose outer coordinate is nearer the G run (:func:`modal_pair`).
    `ref_positions` is the contiguous range between the two coordinates,
    in polyG read order.

    The record is None when no kept read gives both coordinates. An
    alignment keeps the read bases in reference order, so a pair from one
    read always runs toward the G run and spans at least two bases. Raises
    RuntimeError if the chosen pair does not.
    """
    placement = place_polyg_reads(loc.reads, pair.upstream, pair.downstream, loc.marks)
    up_outer, dn_outer = flank_outer_bases(loc.reads, placement, pair.upstream,
                                           pair.downstream)
    kept = np.flatnonzero(placement.kept)
    step = 1 if loc.ref_orientation == "+" else -1
    coords = loc.ref_coords
    read_bases = {
        "upstream": (up_outer[kept], placement.upstream_run_side[kept], step),
        "downstream": (dn_outer[kept], placement.downstream_run_side[kept], -step),
    }
    out = []
    for side, bases in zip(SIDES, (pair.upstream, pair.downstream), strict=True):
        outer_index, run_index, toward_run = read_bases[side]
        outer_coords = coords[kept, outer_index]
        run_coords = coords[kept, run_index]
        both = (outer_coords >= 0) & (run_coords >= 0)
        n_positioned = int(both.sum())
        voted = modal_pair(outer_coords[both], run_coords[both], toward_run)
        if voted is None:
            out.append(FlankPosition(None, 0, 0))
            continue
        outer, run, n_at = voted
        if toward_run * (run - outer) < 1:
            raise RuntimeError(
                f"{loc.name}: the {side} flank {bases} runs from reference position "
                f"{outer} (outer base) to {run} (run-side base), not toward the G run")
        if side == "upstream":
            positions = tuple(range(outer, run + step, step))
            record = FlankRecord(bases, positions, len(positions) - 1)
        else:
            positions = tuple(range(run, outer + step, step))
            record = FlankRecord(bases, positions, 0)
        out.append(FlankPosition(record, n_positioned, n_at))
    return out[0], out[1]


@dataclasses.dataclass
class LocusFlanks:
    """One locus's flank pair, and where each flank lies in the reference.

    `positions` holds the upstream and downstream position; its record is
    None when no kept read gives that flank both reference coordinates.
    """

    name: str
    ref_orientation: str
    pair: FlankPair | None
    positions: tuple[FlankPosition, FlankPosition]

    @property
    def records(self) -> tuple[FlankRecord | None, FlankRecord | None]:
        """The upstream and downstream flank records."""
        return self.positions[0].record, self.positions[1].record


def solve_locus(bam: pysam.AlignmentFile, name: str, chrom: str, bed_start: int,
                bed_end: int) -> LocusFlanks | None:
    """The flank pair at one locus; None when no read rebuilds the reference window.

    The reference coordinates of each flank come from the kept reads'
    alignments (:func:`flank_records_from_alignments`).
    """
    loc = locus_reads(bam, name, chrom, bed_start, bed_end)
    if loc is None:
        return None
    pair = solve_flank_pair(loc.reads, loc.marks)
    if pair is None:
        none = FlankPosition(None, 0, 0)
        return LocusFlanks(name, loc.ref_orientation, None, (none, none))
    return LocusFlanks(name, loc.ref_orientation, pair,
                       flank_records_from_alignments(loc, pair))


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FlankConstruction:
    """Flanks of one matched normal, with the per-side table.

    Attributes
    ----------
    sample : str
        The matched normal whose reads built the flanks.
    records : dict of str to (FlankRecord, FlankRecord)
        Upstream and downstream flank per locus with a reference window.
    sides : pandas.DataFrame
        One row per side of every BED locus.
    """

    sample: str
    records: dict[str, tuple[FlankRecord, FlankRecord]]
    sides: pd.DataFrame


SIDE_COLUMNS = ("locus", "side", "reference_window_status", "ref_orientation",
                "flank_bases", "flank_ref_positions", "objective", "n_reads",
                "n_reads_placed", "n_tied_pairs", "n_reads_positioned",
                "n_reads_at_position")

# Integer columns of the side table that a side with no reference window
# leaves empty.
_SIDE_INT_COLUMNS = ("n_reads", "n_reads_placed", "n_tied_pairs", "n_reads_positioned",
                     "n_reads_at_position")


def construct_flanks(loci: Sequence[tuple[str, LocusFlanks | None]], *, sample: str
                     ) -> FlankConstruction:
    """The side table and flank records from the solved loci, in BED order.

    A locus with None has no reference window: it gets a row on each side
    with the status `WINDOW_UNRECOVERABLE`, empty flank fields and no
    record. The locus columns (`objective`, `n_reads`, `n_reads_placed`,
    `n_tied_pairs`) repeat on both side rows. `n_reads_positioned` and
    `n_reads_at_position` are per side (see :class:`FlankPosition`); they
    are for diagnosis. `objective` is in the units of the
    decision record, half the half-unit value.

    Raises ValueError when a locus with a reference window has no pair with
    F > 0, or when no kept read gives a flank its reference coordinates; the
    message names every such locus or side.
    """
    no_pair = [name for name, loc in loci if loc is not None and loc.pair is None]
    if no_pair:
        raise ValueError(f"{sample}: no flank pair reaches F > 0 at {len(no_pair)} loci: "
                         f"{no_pair}")
    unlocated = [f"{name} {side}" for name, loc in loci if loc is not None
                 for side, record in zip(SIDES, loc.records, strict=True) if record is None]
    if unlocated:
        raise ValueError(f"{sample}: no kept read gives both reference positions for "
                         f"{len(unlocated)} flanks: {unlocated}")
    records: dict[str, tuple[FlankRecord, FlankRecord]] = {}
    rows = []
    for name, loc in loci:
        if loc is None:
            for side in SIDES:
                rows.append({"locus": name, "side": side,
                             "reference_window_status": WINDOW_UNRECOVERABLE,
                             "ref_orientation": "", "flank_bases": "",
                             "flank_ref_positions": "", "objective": None, "n_reads": None,
                             "n_reads_placed": None, "n_tied_pairs": None,
                             "n_reads_positioned": None, "n_reads_at_position": None})
            continue
        records[name] = loc.records
        for side, position in zip(SIDES, loc.positions, strict=True):
            record = position.record
            rows.append({
                "locus": name, "side": side, "reference_window_status": WINDOW_RECOVERED,
                "ref_orientation": loc.ref_orientation, "flank_bases": record.bases,
                "flank_ref_positions": ",".join(map(str, record.ref_positions)),
                "objective": loc.pair.objective_half_units / 2,
                "n_reads": loc.pair.n_reads, "n_reads_placed": loc.pair.n_reads_placed,
                "n_tied_pairs": len(loc.pair.optimal_pairs),
                "n_reads_positioned": position.n_reads_positioned,
                "n_reads_at_position": position.n_reads_at_position,
            })
    sides = pd.DataFrame(rows, columns=list(SIDE_COLUMNS))
    sides["objective"] = sides["objective"].astype(float)
    for column in _SIDE_INT_COLUMNS:
        sides[column] = sides[column].astype("Int64")
    return FlankConstruction(sample, records, sides)


_WORKER_BAM: dict[str, pysam.AlignmentFile] = {}


def _open_worker_bam(bam_path: str) -> None:
    _WORKER_BAM["bam"] = pysam.AlignmentFile(bam_path, "rb")


def _solve_bed_row(row: tuple[str, str, int, int]) -> tuple[str, LocusFlanks | None]:
    name, chrom, bed_start, bed_end = row
    return name, solve_locus(_WORKER_BAM["bam"], name, chrom, bed_start, bed_end)


def worker_count() -> int:
    """Worker processes for the flank search, from the live box.

    The usable cores (``os.sched_getaffinity``) less `_FREE_CORE_SHARE` of
    them, and no more than `_WORKER_MEMORY_SHARE` of ``MemAvailable`` in
    `_WORKER_MEMORY_BYTES` units. At least one.
    """
    cores = len(os.sched_getaffinity(0))
    by_cores = cores - max(1, int(np.ceil(cores * _FREE_CORE_SHARE)))
    with open("/proc/meminfo") as f:
        available_kb = next(int(line.split()[1]) for line in f
                            if line.startswith("MemAvailable:"))
    by_memory = int(available_kb * 1024 * _WORKER_MEMORY_SHARE // _WORKER_MEMORY_BYTES)
    return max(1, min(by_cores, by_memory))


def sample_name(bam_path: str | Path, sample: str | None = None) -> str:
    """The sample name of a BAM: its read-group SM, else `sample`.

    Raises ValueError when the read groups name more than one sample, when
    `sample` differs from the read-group SM, or when the BAM has no SM and
    `sample` is None.
    """
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        groups = bam.header.to_dict().get("RG", [])
    names = {group["SM"] for group in groups if "SM" in group}
    if len(names) > 1:
        raise ValueError(f"{bam_path}: the read groups name more than one sample: "
                         f"{sorted(names)}")
    if names:
        found = names.pop()
        if sample is not None and sample != found:
            raise ValueError(f"{bam_path}: the read-group sample {found!r} differs from "
                             f"the given sample {sample!r}")
        return found
    if sample is None:
        raise ValueError(f"{bam_path}: no read group names a sample (SM); give the "
                         "sample name")
    return sample


def build_flanks(bam_path: str | Path, bed_path: str | Path, *, n_workers: int,
                 sample: str | None = None) -> FlankConstruction:
    """The flank pair of every BED locus of one matched normal.

    Loci run in parallel over `n_workers` processes. A locus whose reference
    window no read rebuilds gets no flank and a side row with the status
    `WINDOW_UNRECOVERABLE`. The sample name comes from :func:`sample_name`.
    """
    sample = sample_name(bam_path, sample)
    rows = _load_bed(Path(bed_path))
    n_workers = max(1, min(n_workers, len(rows)))
    with Pool(n_workers, initializer=_open_worker_bam, initargs=(str(bam_path),)) as pool:
        results = pool.map(_solve_bed_row, rows, chunksize=1)
    return construct_flanks(results, sample=sample)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def write_flank_table(result: FlankConstruction, out_prefix: str | Path) -> Path:
    """Write the side TSV, ``{out_prefix}_flank_sides.tsv``, and return its path.

    The first column names the sample. Extraction reads the table with
    :func:`~prolyg_phasing.io._flank_placement.read_flank_table`.
    """
    path = Path(f"{out_prefix}_flank_sides.tsv")
    out = result.sides.copy()
    out.insert(0, "sample", result.sample)
    out.to_csv(path, sep="\t", index=False)
    return path
