"""Per-donor flank construction from the matched normal's reads.

At each locus side, a donor can carry a G where the reference has the edge
base, or a non-G base further out. That G belongs to the repeat in this
donor, so the flank must start past it. This module reads the matched
normal once and builds one :class:`FlankRecord` per locus side:

1. **Candidates.** In polyG orientation, walk outward from the repeat over
   the reference. Each non-G base within `_CANDIDATE_SPAN` bases is a
   candidate. The first candidate is the edge base. ``d`` is the distance
   from the edge base, outward (the edge base has ``d = 0``).
2. **Count.** In each read, place the outer marker (the `FLANKING_SEQ_LEN`
   reference bases just past the candidate span) by the read's alignment
   and a Hamming search around that projection. Read each candidate from
   the read sequence by its position from the placed marker, not from
   the aligner's pileup: next to a G run, the aligner can put an indel on
   a different base than the read shows. Only reads that place the
   marker count. At each candidate, ``n`` is the number of those reads
   that cover it and ``x`` the number with a G there.
3. **Error rate e.** The pooled G share over candidates far from the
   repeat, with enough reads, not next to a reference GGG, and with a G
   share too low for a donor G.
4. **Call.** Each candidate is "read error" (a read shows a G with
   probability e) or "G in the donor" (probability 0.5 or 1 - e, the
   better of the two). The higher binomial log-likelihood wins; a tie and
   a candidate with no reads are read error.
5. **Flank.** The flank's run-side base is the first candidate, from the
   repeat outward, that is called read error. The flank is the
   `FLANKING_SEQ_LEN` reference bases that end (upstream) or start
   (downstream) there. Other flank positions that are candidates called G
   in the donor are masked. If no candidate is called read error, the
   flank stays at the edge base and the side is flagged.

The unit of every count is the read. The PCR copies of one MI family count
as separate reads. This does not move a call, because ``x`` and ``n``
scale together, but the log-likelihood gap between the two explanations
is larger than the evidence of the independent molecules.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Sequence
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
from scipy.special import xlogy

from prolyg_phasing.io._anchors import (
    build_anchors,
    get_reference_window,
    hamming_search,
    revcomp,
)
from prolyg_phasing.io._flank_placement import (
    WINDOW_RECOVERED,
    WINDOW_UNRECOVERABLE,
    FlankRecord,
    _pack,
    _project,
    polyg_alignments,
    window_reference_position,
)
from prolyg_phasing.io.bam import (
    FLANKING_SEQ_LEN,
    REF_WINDOW_PAD,
    _load_bed,
    filtered_alignments,
    new_locus_qc,
)

# Candidates are the non-G reference bases within this many bases of the
# repeat, the edge base included.
_CANDIDATE_SPAN = 20

# The error rate pools only candidates at least this many bases from the
# repeat, where a donor G next to the run is rare.
_ERROR_MIN_DISTANCE = 10

# The error rate pools only candidates with at least this many reads.
_ERROR_MIN_READS = 20

# The error rate pools only candidates whose G share is below this; a higher
# share can come from a G in the donor.
_ERROR_MAX_G_SHARE = 0.05

# Read offsets on each side of the projected outer marker that the Hamming
# search scores.
_MARKER_SEARCH = 3

# Highest Hamming distance at which the outer marker counts as placed. It
# allows one read error in the marker.
_MARKER_HAMMING_MAX = 1

# A candidate is "next to a reference GGG" when this G run touches it.
_GGG = "GGG"

# Memory one worker process can use at a deep locus, with headroom.
_WORKER_MEMORY_BYTES = 1 << 30

# Share of the available memory that the workers may use together.
_WORKER_MEMORY_SHARE = 0.5

# Share of the usable cores that is left free for other work.
_FREE_CORE_SHARE = 0.125

# The read filters of the extraction, at their defaults.
_FILTERS = {"min_mapq": 40, "drop_chimeric": True, "max_tlen": 1000}

SIDES = ("upstream", "downstream")
CALLS = ("read_error", "g_in_donor")


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SideCandidates:
    """The candidates of one locus side, in polyG-orientation window indices.

    Attributes
    ----------
    upstream : bool
        True for the upstream side.
    edge_index : int
        Window index of the edge base.
    d : tuple of int
        Distance of each candidate from the edge base, outward.
    window_indices : tuple of int
        Window index of each candidate.
    ref_bases : str
        Reference base of each candidate.
    near_ggg : tuple of bool
        Whether a reference GGG touches the candidate on either side.
    marker_start : int
        Window index of the outer marker's first base, in read order.
    marker : str
        The outer marker bases, in polyG orientation.
    """

    upstream: bool
    edge_index: int
    d: tuple[int, ...]
    window_indices: tuple[int, ...]
    ref_bases: str
    near_ggg: tuple[bool, ...]
    marker_start: int
    marker: str


def list_candidates(polyg_window: str, edge_index: int, *, upstream: bool) -> SideCandidates:
    """The candidates and the outer marker of one locus side.

    The candidates are the non-G bases at distance ``d`` in
    ``[0, _CANDIDATE_SPAN)`` from the edge base, outward. Where the window
    is too short for the marker past that span, the span shrinks so that
    the marker stays inside the window.

    Parameters
    ----------
    polyg_window : str
        The reference window in polyG orientation.
    edge_index : int
        Window index of the edge base of this side.
    upstream : bool
        True for the upstream side (outward is toward index 0).
    """
    k = FLANKING_SEQ_LEN
    if upstream:
        span = max(0, min(_CANDIDATE_SPAN, edge_index - k + 1))
        step = -1
        marker_start = edge_index - span - k + 1
    else:
        span = max(0, min(_CANDIDATE_SPAN, len(polyg_window) - k - edge_index))
        step = 1
        marker_start = edge_index + span
    d = []
    indices = []
    for dist in range(span):
        i = edge_index + step * dist
        if polyg_window[i] != "G":
            d.append(dist)
            indices.append(i)
    near_ggg = tuple(
        polyg_window[max(0, i - len(_GGG)):i] == _GGG
        or polyg_window[i + 1:i + 1 + len(_GGG)] == _GGG
        for i in indices
    )
    return SideCandidates(
        upstream=upstream,
        edge_index=edge_index,
        d=tuple(d),
        window_indices=tuple(indices),
        ref_bases="".join(polyg_window[i] for i in indices),
        near_ggg=near_ggg,
        marker_start=marker_start,
        marker=polyg_window[marker_start:marker_start + k],
    )


# ---------------------------------------------------------------------------
# Count
# ---------------------------------------------------------------------------


def read_candidate_bases(
    reads: Sequence[str],
    cigars: Sequence[Sequence[Sequence[int]]],
    starts: np.ndarray,
    side: SideCandidates,
) -> tuple[np.ndarray, np.ndarray]:
    """The base each read shows at each candidate of one side.

    Parameters
    ----------
    reads, cigars, starts
        Reads in polyG orientation, as :func:`polyg_alignments` returns
        them.
    side : SideCandidates
        The side's candidates and outer marker.

    Returns
    -------
    bases : ndarray of uint8, shape (n_reads, n_candidates)
        The read base (ASCII) at each candidate, or 0 where the read does
        not place the marker or does not cover the candidate.
    placed : ndarray of bool, shape (n_reads,)
        Whether the read places the outer marker.

    Notes
    -----
    The marker's projection is the read index of its first base that the
    alignment covers with an M operation, minus that base's offset in the
    marker. A read with no such base does not place the marker. The
    Hamming search scores every read offset within `_MARKER_SEARCH` of the
    projection where the whole marker lies inside the read. The lowest
    distance wins; on a tie the offset nearest the projection wins, then
    the lower offset. A distance above `_MARKER_HAMMING_MAX` does not
    place the marker. A candidate is read at the read index that keeps
    its reference distance to the placed marker.
    """
    n_reads = len(reads)
    k = len(side.marker)
    n_cand = len(side.d)
    bases = np.zeros((n_reads, n_cand), dtype=np.uint8)
    placed = np.zeros(n_reads, dtype=bool)
    if n_reads == 0 or k == 0:
        return bases, placed
    lengths = np.fromiter((len(s) for s in reads), dtype=np.int64, count=n_reads)

    projection = np.full(n_reads, np.iinfo(np.int64).min, dtype=np.int64)
    for i in range(n_reads):
        for j in range(k):
            index, on_m = _project(cigars[i], int(starts[i]), side.marker_start + j,
                                   int(lengths[i]))
            if on_m:
                projection[i] = index - j
                break
    projected = np.flatnonzero(projection != np.iinfo(np.int64).min)

    # Hamming distance at each offset: one exact-length slice per (read,
    # offset), so `hamming_search` scores that one offset.
    shifts = np.arange(-_MARKER_SEARCH, _MARKER_SEARCH + 1)
    offsets = projection[projected][:, None] + shifts[None, :]
    inside = (offsets >= 0) & (offsets + k <= lengths[projected][:, None])
    rows, cols = np.nonzero(inside)
    slices = [reads[projected[r]][offsets[r, c]:offsets[r, c] + k]
              for r, c in zip(rows.tolist(), cols.tolist(), strict=True)]
    distance = np.full(offsets.shape, k + 1, dtype=np.int64)
    distance[rows, cols] = hamming_search(slices, side.marker)[0]
    # Order: distance, then nearness to the projection, then the lower offset.
    key = distance * (4 * _MARKER_SEARCH + 4) + 2 * np.abs(shifts) + (shifts > 0)
    best = key.argmin(axis=1)
    best_distance = distance[np.arange(len(projected)), best]
    ok = best_distance <= _MARKER_HAMMING_MAX
    placed_rows = projected[ok]
    marker_offset = offsets[np.arange(len(projected)), best][ok]
    placed[placed_rows] = True
    if placed_rows.size == 0 or n_cand == 0:
        return bases, placed

    matrix, _ = _pack([reads[i] for i in placed_rows.tolist()])
    read_index = marker_offset[:, None] + (np.asarray(side.window_indices)[None, :]
                                           - side.marker_start)
    covered = (read_index >= 0) & (read_index < lengths[placed_rows][:, None])
    clipped = np.clip(read_index, 0, matrix.shape[1] - 1)
    values = matrix[np.arange(len(placed_rows))[:, None], clipped]
    bases[placed_rows] = np.where(covered, values, 0)
    return bases, placed


def count_candidates(bases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per candidate, the reads that cover it (n) and those with a G there (x)."""
    n = (bases != 0).sum(axis=0).astype(np.int64)
    x = (bases == ord("G")).sum(axis=0).astype(np.int64)
    return n, x


# ---------------------------------------------------------------------------
# Error rate and call
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ErrorRate:
    """The pooled error rate e and the positions it pools.

    Attributes
    ----------
    e : float
        Pooled ``x / n``.
    x, n : int
        Pooled G reads and covering reads.
    position_g_shares : ndarray of float
        The G share of each pooled position.
    """

    e: float
    x: int
    n: int
    position_g_shares: np.ndarray

    @property
    def n_positions(self) -> int:
        return len(self.position_g_shares)


def estimate_error_rate(d: np.ndarray, near_ggg: np.ndarray, n: np.ndarray,
                        x: np.ndarray) -> ErrorRate:
    """Pooled G share over the candidates that cannot carry a donor G.

    A candidate is pooled when its distance is at least
    `_ERROR_MIN_DISTANCE`, no reference GGG touches it, it has at least
    `_ERROR_MIN_READS` reads, and its G share is below
    `_ERROR_MAX_G_SHARE`. The arguments run over every candidate of every
    locus side.
    """
    d, near_ggg, n, x = (np.asarray(a) for a in (d, near_ggg, n, x))
    keep = (d >= _ERROR_MIN_DISTANCE) & ~near_ggg.astype(bool) & (n >= _ERROR_MIN_READS)
    keep[keep] = x[keep] / n[keep] < _ERROR_MAX_G_SHARE
    x_pool = int(x[keep].sum())
    n_pool = int(n[keep].sum())
    if n_pool == 0:
        raise ValueError("no candidate qualifies for the error-rate estimate")
    return ErrorRate(e=x_pool / n_pool, x=x_pool, n=n_pool,
                     position_g_shares=x[keep] / n[keep])


def call_candidates(n: np.ndarray, x: np.ndarray, e: float
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binomial log-likelihoods of the two explanations, and the call.

    Returns ``(loglik_read_error, loglik_g_in_donor, g_in_donor)``. "Read
    error" uses ``p = e``. "G in the donor" uses the better of ``p = 0.5``
    and ``p = 1 - e``. A candidate is G in the donor only when its value
    is strictly higher, so a candidate with no reads is read error.
    """
    n = np.asarray(n, dtype=float)
    x = np.asarray(x, dtype=float)

    def loglik(p: float) -> np.ndarray:
        return xlogy(x, p) + xlogy(n - x, 1 - p)

    ll_error = loglik(e)
    ll_g = np.maximum(loglik(0.5), loglik(1 - e))
    return ll_error, ll_g, ll_g > ll_error


# ---------------------------------------------------------------------------
# Flank
# ---------------------------------------------------------------------------


def choose_flank(
    side: SideCandidates,
    g_in_donor: np.ndarray,
    polyg_window: str,
    *,
    ref_orientation: str,
    window_start: int,
) -> tuple[FlankRecord, int, bool]:
    """The flank of one side, the distance of its run-side base, and the flag.

    The run-side base is the first candidate called read error. When no
    candidate is called read error, it is the edge base and the flag is
    set. Other flank positions that are candidates called G in the donor
    are masked.
    """
    g_in_donor = np.asarray(g_in_donor, dtype=bool)
    read_error = np.flatnonzero(~g_in_donor)
    no_read_error = read_error.size == 0
    run_side_d = 0 if no_read_error else side.d[int(read_error[0])]
    k = FLANKING_SEQ_LEN
    if side.upstream:
        run_side = side.edge_index - run_side_d
        first = run_side - k + 1
    else:
        run_side = side.edge_index + run_side_d
        first = run_side
    masked = []
    for dist, is_g in zip(side.d, g_in_donor.tolist(), strict=True):
        step = dist - run_side_d
        if is_g and 0 < step < k:
            masked.append(k - 1 - step if side.upstream else step)
    ref_positions = tuple(
        window_reference_position(i, ref_orientation=ref_orientation,
                                  window_start=window_start, window_len=len(polyg_window))
        for i in range(first, first + k)
    )
    record = FlankRecord(
        bases=polyg_window[first:first + k],
        ref_positions=ref_positions,
        run_side_offset=k - 1 if side.upstream else 0,
        masked_offsets=tuple(sorted(masked)),
    )
    return record, run_side_d, no_read_error


# ---------------------------------------------------------------------------
# One locus
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LocusReads:
    """One locus's reference geometry and its filtered reads in polyG orientation.

    `records` are the pysam alignments that overlap the BED interval and
    pass the extraction's read filters; `reads`, `cigars` and `starts`
    are the same reads in polyG orientation.
    """

    name: str
    ref_orientation: str
    window_start: int
    polyg_window: str
    sides: dict[str, SideCandidates]
    records: list[pysam.AlignedSegment]
    reads: list[str]
    cigars: list[list[list[int]]]
    starts: np.ndarray


def locus_reads(bam: pysam.AlignmentFile, name: str, chrom: str, bed_start: int,
                bed_end: int) -> LocusReads | None:
    """Geometry and filtered reads at one locus.

    Returns None when no read rebuilds the reference window. Raises
    ValueError, naming the locus, when the window holds no edge base on a
    side: the G run reaches the end of the window.
    """
    window_start = bed_start - REF_WINDOW_PAD
    window = get_reference_window(bam, chrom, window_start, bed_end + REF_WINDOW_PAD,
                                  min_mapq=_FILTERS["min_mapq"])
    if window is None:
        return None
    anchors = build_anchors(window, REF_WINDOW_PAD, bed_end - bed_start, k=FLANKING_SEQ_LEN)
    if anchors.status != "anchorable":
        raise ValueError(f"{name}: the reference window holds no edge base and flank on "
                         f"each side ({anchors.status})")
    polyg_window = revcomp(window) if anchors.ref_orientation == "-" else window
    edges = {
        "upstream": REF_WINDOW_PAD - anchors.g_walk_up - 1,
        "downstream": REF_WINDOW_PAD + (bed_end - bed_start) + anchors.g_walk_dn,
    }
    sides = {s: list_candidates(polyg_window, edges[s], upstream=s == "upstream")
             for s in SIDES}
    records = [r for r, _ in filtered_alignments(bam, chrom, bed_start, bed_end, bed_start,
                                                 bed_end, qc=new_locus_qc(), **_FILTERS)]
    reads, cigars, starts = polyg_alignments(
        [r.query_sequence for r in records], [r.cigartuples for r in records],
        [r.reference_start for r in records], ref_orientation=anchors.ref_orientation,
        window_start=window_start, window_len=len(window))
    return LocusReads(name, anchors.ref_orientation, window_start, polyg_window, sides,
                      records, reads, cigars, starts)


@dataclasses.dataclass
class LocusCounts:
    """One locus's per-side candidates and counts (the count pass output)."""

    name: str
    ref_orientation: str
    window_start: int
    polyg_window: str
    sides: dict[str, SideCandidates]
    n: dict[str, np.ndarray]
    x: dict[str, np.ndarray]
    n_reads: int
    n_marker_placed: dict[str, int]


def count_locus(bam: pysam.AlignmentFile, name: str, chrom: str, bed_start: int,
                bed_end: int) -> LocusCounts | None:
    """Count pass at one locus; None when no read rebuilds the reference window."""
    loc = locus_reads(bam, name, chrom, bed_start, bed_end)
    if loc is None:
        return None
    n, x, n_placed = {}, {}, {}
    for s in SIDES:
        bases, placed = read_candidate_bases(loc.reads, loc.cigars, loc.starts, loc.sides[s])
        n[s], x[s] = count_candidates(bases)
        n_placed[s] = int(placed.sum())
    return LocusCounts(name, loc.ref_orientation, loc.window_start, loc.polyg_window,
                       loc.sides, n, x, len(loc.reads), n_placed)


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FlankConstruction:
    """Flanks of one matched normal, with their QC tables.

    Attributes
    ----------
    sample : str
        The matched normal whose reads built the flanks.
    records : dict of str to (FlankRecord, FlankRecord)
        Upstream and downstream flank per locus with a reference window.
    candidates : pandas.DataFrame
        One row per candidate.
    sides : pandas.DataFrame
        One row per side of every BED locus.
    error_rate : ErrorRate
    """

    sample: str
    records: dict[str, tuple[FlankRecord, FlankRecord]]
    candidates: pd.DataFrame
    sides: pd.DataFrame
    error_rate: ErrorRate


_SIDE_COLUMNS = ("locus", "side", "reference_window_status", "ref_orientation", "flank_d",
                 "flank_bases", "flank_ref_positions", "masked_offsets",
                 "no_read_error_candidate", "n_candidates", "n_reads", "n_marker_placed")

# Integer columns of the side table that a side with no reference window
# leaves empty.
_SIDE_INT_COLUMNS = ("flank_d", "no_read_error_candidate", "n_reads", "n_marker_placed")


def construct_flanks(loci: Sequence[tuple[str, LocusCounts | None]], *, sample: str
                     ) -> FlankConstruction:
    """Error rate, calls and flanks from the count pass of every BED locus of `sample`.

    `loci` lists each BED locus in BED order with its counts, or None when
    no read rebuilt its reference window. Such a locus gets a side row on
    each side with the status `WINDOW_UNRECOVERABLE` and empty flank fields,
    no candidate rows and no flank record; it adds nothing to the error
    rate.
    """
    counts = [c for _, c in loci if c is not None]
    rows = []
    for loc in counts:
        for s in SIDES:
            side = loc.sides[s]
            for i, dist in enumerate(side.d):
                rows.append((loc.name, s, dist, side.ref_bases[i], side.near_ggg[i],
                             int(loc.n[s][i]), int(loc.x[s][i])))
    cand = pd.DataFrame(rows, columns=["locus", "side", "d", "ref_base", "near_ggg", "n", "x"])
    error_rate = estimate_error_rate(cand["d"].to_numpy(), cand["near_ggg"].to_numpy(),
                                     cand["n"].to_numpy(), cand["x"].to_numpy())
    n = cand["n"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        cand["g_share"] = np.where(n > 0, cand["x"] / np.maximum(n, 1), np.nan)
    ll_error, ll_g, is_g = call_candidates(n, cand["x"].to_numpy(), error_rate.e)
    cand["loglik_read_error"] = ll_error
    cand["loglik_g_in_donor"] = ll_g
    cand["call"] = np.where(is_g, "g_in_donor", "read_error")
    cand["near_ggg"] = cand["near_ggg"].astype(int)

    records: dict[str, tuple[FlankRecord, FlankRecord]] = {}
    side_rows = []
    start = 0
    for name, loc in loci:
        if loc is None:
            for s in SIDES:
                side_rows.append({
                    "locus": name, "side": s, "reference_window_status": WINDOW_UNRECOVERABLE,
                    "ref_orientation": "", "flank_d": None, "flank_bases": "",
                    "flank_ref_positions": "", "masked_offsets": "",
                    "no_read_error_candidate": None, "n_candidates": 0, "n_reads": None,
                    "n_marker_placed": None,
                })
            continue
        pair = []
        for s in SIDES:
            side = loc.sides[s]
            stop = start + len(side.d)
            record, run_side_d, flag = choose_flank(
                side, is_g[start:stop], loc.polyg_window,
                ref_orientation=loc.ref_orientation, window_start=loc.window_start)
            start = stop
            pair.append(record)
            side_rows.append({
                "locus": loc.name, "side": s, "reference_window_status": WINDOW_RECOVERED,
                "ref_orientation": loc.ref_orientation,
                "flank_d": run_side_d, "flank_bases": record.bases,
                "flank_ref_positions": ",".join(map(str, record.ref_positions)),
                "masked_offsets": ",".join(map(str, record.masked_offsets)),
                "no_read_error_candidate": int(flag),
                "n_candidates": len(side.d), "n_reads": loc.n_reads,
                "n_marker_placed": loc.n_marker_placed[s],
            })
        records[loc.name] = (pair[0], pair[1])
    sides = pd.DataFrame(side_rows, columns=_SIDE_COLUMNS)
    for column in _SIDE_INT_COLUMNS:
        sides[column] = sides[column].astype("Int64")
    return FlankConstruction(sample, records, cand, sides, error_rate)


_WORKER_BAM: dict[str, pysam.AlignmentFile] = {}


def _open_worker_bam(bam_path: str) -> None:
    _WORKER_BAM["bam"] = pysam.AlignmentFile(bam_path, "rb")


def _count_bed_row(row: tuple[str, str, int, int]) -> tuple[str, LocusCounts | None]:
    name, chrom, bed_start, bed_end = row
    return name, count_locus(_WORKER_BAM["bam"], name, chrom, bed_start, bed_end)


def worker_count() -> int:
    """Worker processes for the count pass, from the live box.

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
    """Count pass over every BED locus of one matched normal, then the flanks.

    Loci run in parallel over `n_workers` processes. A locus whose reference
    window no read rebuilds gets no flank and a side row with the status
    `WINDOW_UNRECOVERABLE`. The sample name comes from :func:`sample_name`.
    """
    sample = sample_name(bam_path, sample)
    rows = _load_bed(Path(bed_path))
    n_workers = max(1, min(n_workers, len(rows)))
    with Pool(n_workers, initializer=_open_worker_bam, initargs=(str(bam_path),)) as pool:
        results = pool.map(_count_bed_row, rows, chunksize=2)
    return construct_flanks(results, sample=sample)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def write_flank_tables(result: FlankConstruction, out_prefix: str | Path) -> list[Path]:
    """Write the candidate TSV, the side TSV and the error-rate JSON.

    The files are ``{out_prefix}_flank_candidates.tsv``,
    ``{out_prefix}_flank_sides.tsv`` and ``{out_prefix}_error_rate.json``.
    Each names the sample in a first ``sample`` column or key. Extraction
    reads the side TSV with :func:`read_flank_table`.
    """
    prefix = str(out_prefix)
    paths = [Path(f"{prefix}_flank_candidates.tsv"), Path(f"{prefix}_flank_sides.tsv"),
             Path(f"{prefix}_error_rate.json")]
    for table, path in ((result.candidates, paths[0]), (result.sides, paths[1])):
        out = table.copy()
        out.insert(0, "sample", result.sample)
        out.to_csv(path, sep="\t", index=False)
    er = result.error_rate
    quantiles = [0.1, 0.5, 0.9, 0.99]
    paths[2].write_text(json.dumps({
        "sample": result.sample,
        "e": er.e, "x": er.x, "n": er.n, "n_positions": er.n_positions,
        "position_g_share_quantiles": dict(zip(
            [f"q{int(q * 100):02d}" for q in quantiles],
            np.quantile(er.position_g_shares, quantiles).tolist(), strict=True)),
    }, indent=1))
    return paths
