"""BAM panel extraction — `extract_panel` entry point.

Walks a BAM against a BED panel and produces an in-memory
`ExtractedPanel` of faithful per-(MI, strand, seq) read counts.

Every locus needs its two flanks from the matched normal's flank table
(`prolyg-phasing build-flanks`). Two steps per locus:

1. **Reference window** — reconstruct the reference on BED ± `pad` bp
   from MD tags, and check that both flanks lie in it with their bases.
2. **Extraction** — fetch overlapping alignments, apply the read filters
   (MAPQ, SA, TLEN, MI), place both flanks in each read
   (`_flank_placement.place_flanks`), group by `query_name` for pair
   collapse, merge or drop pair disagreements, and emit per-(MI, strand,
   seq) read counts.

Per-locus ``n_runs`` is set from the **data** (max parsed tuple length
across accepted reads), not from the reference tract.
This handles interrupter-pattern polymorphisms where the reference
predicts one ``S`` but the sample carries reads at multiple ``S``
values (e.g. PolyG3298: ~49% of reads parse to 2-tuples, ~46% to
1-tuples, both biologically real). The reference-derived
``reference_run_lengths`` is retained as diagnostic context.

`n_alleles` is panel-wide and decided at extraction time:
`n_alleles = max_observed_run_length + n_alleles_margin`, where
`max_observed_run_length` is the maximum G-run length seen in any
parsed read across the panel.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

import numpy as np
import pysam
from tqdm.auto import tqdm

from prolyg_phasing.io._anchors import (
    build_anchors,
    get_reference_window,
    revcomp,
)
from prolyg_phasing.io._flank_placement import (
    SIDE_DROP_REASONS,
    FlankPlacement,
    FlankRecord,
    place_flanks,
    read_flank_table,
    window_index,
)
from prolyg_phasing.io.panel import (
    EXTRACTION_SCHEMA_VERSION,
    ExtractedLocus,
    ExtractedPanel,
    ExtractionProvenance,
    parse_run_lengths,
)

FLANKING_SEQ_LEN = 10
REF_WINDOW_PAD = 50

PAIR_DISAGREEMENT_POLICY = "merge_by_quality"
DEFAULT_MIN_FLANKING_BASE_Q = 20

_COMP_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")

# Unmapped | secondary | supplementary. Tested as one flag word rather
# than three property reads, on a loop that runs once per alignment in
# the fetch window.
_DROP_FLAGS = 0x4 | 0x100 | 0x800


# ---------------------------------------------------------------------------
# BED loading
# ---------------------------------------------------------------------------


def _load_bed(bed_path: Path) -> list[tuple[str, str, int, int]]:
    """Load a 4-column BED: `(name, chrom, start, end)` per row."""
    rows: list[tuple[str, str, int, int]] = []
    with bed_path.open() as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            chrom, start, end, name = line.rstrip("\n").split("\t")[:4]
            rows.append((name, chrom, int(start), int(end)))
    return rows


# ---------------------------------------------------------------------------
# Per-locus geometry + extraction
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LocusGeometry:
    """The reference geometry of one locus in one BAM, with its two flanks.

    Attributes
    ----------
    ref_orientation : str
        "+" if the repeat reads as Gs on the reference, "-" otherwise.
    reference_window : str
        The reference on BED ± `REF_WINDOW_PAD`, in reference orientation.
    reference_window_start : int
        Reference coordinate of ``reference_window[0]``.
    upstream, downstream : FlankRecord
        The two flanks from the matched normal's flank table.
    g_walk_up, g_walk_dn : int
        The G bases between each BED end and that side's edge base.
    """

    ref_orientation: str
    reference_window: str
    reference_window_start: int
    upstream: FlankRecord
    downstream: FlankRecord
    g_walk_up: int
    g_walk_dn: int

    def polyg_index(self, position: int) -> int:
        """Index of reference coordinate `position` in the polyG-orientation window."""
        return window_index(position, ref_orientation=self.ref_orientation,
                            window_start=self.reference_window_start,
                            window_len=len(self.reference_window))

    @property
    def flanked_interval(self) -> tuple[int, int]:
        """Reference interval ``[lo, hi)`` from one flank's outer base to the other's.

        The flanking window for phasing lies outside it.
        """
        outer = (self.upstream.outer_position, self.downstream.outer_position)
        return min(outer), max(outer) + 1

    @property
    def reference_tract(self) -> str:
        """The reference strictly between the two run-side bases, in polyG orientation."""
        window = self.reference_window
        if self.ref_orientation == "-":
            window = revcomp(window)
        lo = self.polyg_index(self.upstream.run_side_position) + 1
        hi = self.polyg_index(self.downstream.run_side_position)
        return window[lo:hi]


def locus_geometry(
    bam: pysam.AlignmentFile,
    name: str,
    chrom: str,
    bed_start: int,
    bed_end: int,
    flanks: tuple[FlankRecord, FlankRecord],
    *,
    min_mapq: int,
) -> LocusGeometry | None:
    """The reference window of one locus from MD tags, checked against its flanks.

    Returns None when no read rebuilds the reference window. Raises
    ValueError, naming the locus, when a flank does not lie in the window
    with its own bases, or when the flanks do not enclose the repeat.
    """
    window_start = bed_start - REF_WINDOW_PAD
    window = get_reference_window(bam, chrom, window_start, bed_end + REF_WINDOW_PAD,
                                  min_mapq=min_mapq)
    if window is None:
        return None
    walk = build_anchors(window, REF_WINDOW_PAD, bed_end - bed_start, k=FLANKING_SEQ_LEN)
    geometry = LocusGeometry(
        ref_orientation=walk.ref_orientation,
        reference_window=window,
        reference_window_start=window_start,
        upstream=flanks[0],
        downstream=flanks[1],
        g_walk_up=walk.g_walk_up,
        g_walk_dn=walk.g_walk_dn,
    )
    polyg_window = revcomp(window) if walk.ref_orientation == "-" else window
    for side, flank in zip(("upstream", "downstream"), flanks, strict=True):
        indices = [geometry.polyg_index(p) for p in flank.ref_positions]
        inside = all(0 <= i < len(window) for i in indices)
        steps = all(b - a == 1 for a, b in zip(indices, indices[1:], strict=False))
        if not (inside and steps) or "".join(polyg_window[i] for i in indices) != flank.bases:
            raise ValueError(
                f"{name}: the {side} flank {flank.bases} at reference positions "
                f"{flank.ref_positions[0]}..{flank.ref_positions[-1]} does not match the "
                f"reference window of this BAM ({walk.ref_orientation} orientation)")
    if geometry.polyg_index(flanks[0].run_side_position) \
            >= geometry.polyg_index(flanks[1].run_side_position):
        raise ValueError(f"{name}: the upstream flank is not upstream of the downstream flank")
    return geometry


def place_locus_flanks(alignments: Sequence[pysam.AlignedSegment],
                       geometry: LocusGeometry) -> FlankPlacement:
    """Place both flanks of one locus in each alignment; one entry per alignment.

    This is the per-read decision of the extraction: the path and the drop
    reason on each side, and the tract (see :func:`place_flanks`).
    """
    return place_flanks(
        [r.query_sequence for r in alignments],
        [r.cigartuples for r in alignments],
        np.fromiter((r.reference_start for r in alignments), dtype=np.int64,
                    count=len(alignments)),
        upstream=geometry.upstream,
        downstream=geometry.downstream,
        ref_orientation=geometry.ref_orientation,
        reference_window=geometry.reference_window,
        reference_window_start=geometry.reference_window_start,
    )


# One side (upstream or downstream) of one read's flanking bases, as
# ``(start_distance, bases)``: ``bases[i]`` is the canonical-orientation
# base at distance ``start_distance + i`` from that side's flank, going
# outward. ``"."`` marks a distance the read says nothing about — an
# alignment gap, or a base below ``min_base_q``. Reads are A/C/G/T/N
# after upper-casing, so ``"."`` is never a real base, and it is already
# the placeholder the emitted flanking string uses. Leading and trailing
# ``"."`` are always stripped, so ``start_distance + len(bases)`` is the
# side's observed extent and the empty side is ``(0, "")``.
FlankingSide = tuple[int, str]

_EMPTY_SIDE: FlankingSide = (0, "")

# Per-threshold translation table turning a read's quality bytes into a
# per-position flag: 1 below `min_base_q`, 0 at or above it. One
# `bytes.translate` then reduces the whole read to a mark string that
# `find` can walk, which is how the masking below locates the runs to
# blank out without touching the positions that pass.
_LOW_QUALITY_MARKS: dict[int, bytes] = {}


def _low_quality_marks(min_base_q: int) -> bytes:
    """Translation table marking quality bytes below `min_base_q` with ``1``."""
    table = _LOW_QUALITY_MARKS.get(min_base_q)
    if table is None:
        table = bytes(1 if q < min_base_q else 0 for q in range(256))
        _LOW_QUALITY_MARKS[min_base_q] = table
    return table


def _mask_low_quality(bases: str, marks: bytes, first_low: int) -> str:
    """`bases` with every position `marks` flags as sub-threshold set to ``"."``.

    `first_low` is the index of the first flagged position, as already
    located by the caller. Sub-threshold bases come in runs — usually one,
    at a read end — so the string is rebuilt run by run rather than
    position by position.
    """
    parts: list[str] = []
    cursor = 0
    low_start = first_low
    while low_start >= 0:
        low_end = marks.find(0, low_start + 1)
        if low_end < 0:
            low_end = len(marks)
        parts.append(bases[cursor:low_start])
        parts.append("." * (low_end - low_start))
        cursor = low_end
        low_start = marks.find(1, low_end)
    parts.append(bases[cursor:])
    return "".join(parts)


def _assemble_side(
    pieces: list[tuple[int, str]], outward_first: bool, may_have_dots: bool
) -> FlankingSide:
    """Join one side's per-CIGAR-block runs into a single `FlankingSide`.

    `pieces` are disjoint ``(start_distance, bases)`` runs in reference
    order. Set `outward_first` for the side whose distances *decrease*
    with reference position (so the runs arrive back-to-front). Gaps
    between runs are filled with ``"."``; leading and trailing ``"."``
    are stripped so the returned extent is the observed one.

    `may_have_dots` says whether the run bases can contain ``"."`` at
    all. A read whose every base cleared `min_base_q` was never masked,
    and a BAM sequence is IUPAC codes only, so a single unmasked run is
    already normalized and needs no strip.
    """
    if not pieces:
        return _EMPTY_SIDE
    if outward_first:
        pieces = pieces[::-1]
    start = pieces[0][0]
    if len(pieces) == 1:
        if not may_have_dots:
            return pieces[0] if pieces[0][1] else _EMPTY_SIDE
        text = pieces[0][1]
    else:
        parts: list[str] = []
        cursor = start
        for piece_start, piece_text in pieces:
            if piece_start > cursor:
                parts.append("." * (piece_start - cursor))
            parts.append(piece_text)
            cursor = piece_start + len(piece_text)
        text = "".join(parts)
    return _strip_side(start, text)


def _strip_side(start: int, text: str) -> FlankingSide:
    """Normalize a raw ``(start, text)`` by dropping leading/trailing ``"."``."""
    covered = text.lstrip(".")
    if not covered:
        return _EMPTY_SIDE
    return (start + len(text) - len(covered), covered.rstrip("."))


def _alignment_flanking(
    r: pysam.AlignedSegment,
    *,
    lo: int,
    hi: int,
    ref_orientation: str,
    min_base_q: int,
) -> tuple[FlankingSide, FlankingSide]:
    """Flanking bases for one alignment, by canonical distance from the flank.

    `lo` and `hi` bound the reference interval ``[lo, hi)`` that the two
    flanks and the repeat cover (`LocusGeometry.flanked_interval`).
    Returns ``(upstream_side, downstream_side)`` as `FlankingSide` pairs.
    Distance 0 on each side is the base immediately adjacent to that
    side's flank going outward (away from the polyG). Bases are in
    canonical orientation (complemented for ``"-"`` orientation loci).
    Positions the alignment skips (insertions, soft clips, deletions)
    and bases below `min_base_q` are left as ``"."``.

    Walks the CIGAR in whole aligned blocks and slices the read's bases
    per block. The per-position work this replaces — one
    `get_aligned_pairs` tuple, one quality test and one dict store per
    aligned base — was the single largest cost in `extract_panel`.
    """
    seq = r.query_sequence
    quals = r.query_qualities
    if seq is None or quals is None:
        return _EMPTY_SIDE, _EMPTY_SIDE
    cigar = r.cigartuples
    if not cigar:
        return _EMPTY_SIDE, _EMPTY_SIDE

    # Both orientations see the same geometry: one reference region left
    # of the flanks (`refpos < lo`) and one right of them (`refpos >=
    # hi`). Orientation decides which region is the upstream side, which
    # way distances run, and whether bases are complemented.
    if ref_orientation == "+":
        bases = seq.upper()
    else:
        bases = seq.upper().translate(_COMP_TABLE)

    # Mask sub-threshold bases once, so the per-block fills below are
    # plain slices with no per-position quality test. Most reads clear
    # the threshold everywhere, and for those the `find` below is the
    # whole cost.
    marks = quals.tobytes().translate(_low_quality_marks(min_base_q))
    first_low = marks.find(1)
    masked = first_low >= 0
    if masked:
        bases = _mask_low_quality(bases, marks, first_low)

    left_pieces: list[tuple[int, str]] = []
    right_pieces: list[tuple[int, str]] = []
    qpos = 0
    pos = r.reference_start
    for op, op_len in cigar:
        if op == 0 or op == 7 or op == 8:      # M / = / X — consumes both
            end = pos + op_len
            if pos < lo:
                left_end = end if end < lo else lo
                # Distance runs backwards from `lo`, so reverse the run.
                left_pieces.append((lo - left_end, bases[qpos:qpos + left_end - pos][::-1]))
            if end > hi:
                right_start = pos if pos > hi else hi
                right_pieces.append(
                    (right_start - hi, bases[qpos + right_start - pos:qpos + op_len])
                )
            qpos += op_len
            pos = end
        elif op == 1 or op == 4 or op == 6:    # I / S / P — consumes query
            # P consumes neither cursor per the SAM spec, but pysam's own
            # get_aligned_pairs advances the query cursor across it. This
            # walk reproduces pysam, so the two agree base for base on
            # any alignment either of them can be handed.
            qpos += op_len
        elif op == 2 or op == 3:               # D / N — consumes reference
            pos += op_len
        # op == 5 (H) consumes neither.

    # A gap between runs also introduces ".", so only a single run can
    # skip the strip.
    left = _assemble_side(left_pieces, outward_first=True, may_have_dots=masked)
    right = _assemble_side(right_pieces, outward_first=False, may_have_dots=masked)
    return (left, right) if ref_orientation == "+" else (right, left)


def _consensus(a: FlankingSide, b: FlankingSide) -> FlankingSide:
    """Per-position consensus of two flanking sides.

    Agreement keeps the base; disagreement drops the position to
    ``"."``; a position covered by only one side is kept as-is. Sides
    that do not overlap are concatenated across a ``"."`` gap, and an
    overlap the two sides spell identically — the usual case for a
    read pair — is taken whole, so the per-position merge runs only
    where the mates actually disagree.
    """
    start_a, text_a = a
    start_b, text_b = b
    if not text_a:
        return b
    if not text_b:
        return a
    if start_b < start_a:
        start_a, text_a, start_b, text_b = start_b, text_b, start_a, text_a
    end_a = start_a + len(text_a)
    end_b = start_b + len(text_b)
    if start_b >= end_a:
        return (start_a, text_a + "." * (start_b - end_a) + text_b)

    head_len = start_b - start_a
    overlap_len = (end_a if end_a < end_b else end_b) - start_b
    overlap_a = text_a[head_len:head_len + overlap_len]
    overlap_b = text_b[:overlap_len]
    if overlap_a == overlap_b:
        overlap = overlap_a
    else:
        overlap = "".join(
            ca if ca == cb else (cb if ca == "." else (ca if cb == "." else "."))
            for ca, cb in zip(overlap_a, overlap_b, strict=True)
        )
    tail = text_a[head_len + overlap_len:] if end_a >= end_b else text_b[overlap_len:]
    return _strip_side(start_a, text_a[:head_len] + overlap + tail)


def _merge_tract_by_quality(
    seq_a: str,
    seq_b: str,
    quality_a: Sequence[int] | None,
    quality_b: Sequence[int] | None,
) -> str | None:
    """Merge two mates' equal-length tracts by base quality.

    At each position where the mates differ, the base with the higher
    Phred quality wins. Return ``None`` when the tracts differ in length,
    when either mate has no stored qualities, or when the mates differ at
    a position where their qualities are equal. The caller then drops the
    pair.

    Parameters
    ----------
    seq_a, seq_b : str
        Tracts in polyG orientation.
    quality_a, quality_b : sequence of int or None
        Per-base Phred qualities of the two tracts, in the same
        orientation and over the same slice as ``seq_a`` and ``seq_b``.

    Returns
    -------
    str or None
        The merged tract, or ``None`` when no merge is possible.
    """
    if len(seq_a) != len(seq_b) or quality_a is None or quality_b is None:
        return None
    merged = list(seq_a)
    for i, (base_a, base_b) in enumerate(zip(seq_a, seq_b, strict=True)):
        if base_a == base_b:
            continue
        if quality_a[i] == quality_b[i]:
            return None
        if quality_b[i] > quality_a[i]:
            merged[i] = base_b
    return "".join(merged)


def _tract_qualities(
    alignment: pysam.AlignedSegment,
    tract_start: int,
    tract_end: int,
    ref_orientation: str,
) -> Sequence[int] | None:
    """Base qualities of an alignment's tract, in polyG orientation.

    The tract is the polyG-orientation read on ``[tract_start,
    tract_end)``. Only a pair whose tracts differ reads them, so the
    qualities of the other reads are never decoded.
    """
    qualities = alignment.query_qualities
    if qualities is None:
        return None
    if ref_orientation == "-":
        qualities = qualities[::-1]
    return qualities[tract_start:tract_end]


def _merge_pair_flanking(
    mates: list[tuple[FlankingSide, FlankingSide]],
) -> tuple[FlankingSide, FlankingSide]:
    """Per-position consensus over up-to-2 mates' flanking sides.

    Agreement keeps the base; disagreement drops the position. Singleton
    input returns its single mate's sides unchanged. Empty input returns
    two empty sides.
    """
    if not mates:
        return _EMPTY_SIDE, _EMPTY_SIDE
    if len(mates) == 1:
        return mates[0]
    (up_a, dn_a), (up_b, dn_b) = mates
    return _consensus(up_a, up_b), _consensus(dn_a, dn_b)


def filtered_alignments(
    bam: pysam.AlignmentFile,
    chrom: str,
    fetch_start: int,
    fetch_end: int,
    bed_start: int,
    bed_end: int,
    *,
    min_mapq: int,
    drop_chimeric: bool,
    max_tlen: int,
    qc: dict,
) -> Iterator[tuple[pysam.AlignedSegment, bool]]:
    """Alignments in the fetch window that pass the extraction's read filters.

    Yields ``(alignment, bed_overlap)`` in fetch order. The filters drop
    unmapped, secondary and supplementary alignments, mapping quality
    below `min_mapq`, an SA tag (when `drop_chimeric`), ``|TLEN| >
    max_tlen`` and an empty stored sequence. An alignment that overlaps
    the BED interval must also carry an MI tag. For alignments that
    overlap the BED interval, the filters add to the
    ``n_alignments_overlap_locus`` and ``n_alignments_drop_{mapq,sa,tlen}``
    counters in `qc`.
    """
    for r in bam.fetch(chrom, fetch_start, fetch_end):
        if r.flag & _DROP_FLAGS:
            continue
        reference_end = r.reference_end
        if reference_end is None:
            continue
        bed_overlap = r.reference_start < bed_end and reference_end > bed_start

        # Alignment-level filters apply to both kinds.
        if r.mapping_quality < min_mapq:
            if bed_overlap:
                qc["n_alignments_overlap_locus"] += 1
                qc["n_alignments_drop_mapq"] += 1
            continue
        if drop_chimeric and r.has_tag("SA"):
            if bed_overlap:
                qc["n_alignments_overlap_locus"] += 1
                qc["n_alignments_drop_sa"] += 1
            continue
        template_length = r.template_length
        tlen = abs(template_length) if template_length else 0
        if tlen > max_tlen:
            if bed_overlap:
                qc["n_alignments_overlap_locus"] += 1
                qc["n_alignments_drop_tlen"] += 1
            continue
        # `query_length` is the stored sequence length, and pysam returns
        # `query_sequence is None` for exactly the reads that store none.
        # Reading the length skips decoding the sequence for every read
        # in the window that turns out not to need it.
        if r.query_length == 0:
            continue

        if bed_overlap:
            qc["n_alignments_overlap_locus"] += 1
            if not r.has_tag("MI"):
                continue
        yield r, bed_overlap




# The per-side counters of reads that a flank drops, one per side and reason.
FLANK_DROP_COUNTERS = tuple(
    f"n_alignments_drop_{side}_{reason}"
    for side in ("upstream", "downstream") for reason in SIDE_DROP_REASONS
)


def new_locus_qc() -> dict[str, int]:
    """Zeroed per-locus QC counters of one extraction."""
    return {
        "n_alignments_overlap_locus": 0,
        "n_alignments_drop_mapq": 0,
        "n_alignments_drop_sa": 0,
        "n_alignments_drop_tlen": 0,
        "n_alignments_drop_flank": 0,
        **{name: 0 for name in FLANK_DROP_COUNTERS},
        "n_alignments_drop_flanks_crossed": 0,
        "n_read_pairs": 0,
        "n_read_pairs_drop_disagree": 0,
        "n_read_pairs_merged": 0,
    }


def _count_flank_drops(placement: FlankPlacement, qc: dict) -> None:
    """Add the flank drops of one locus to `qc`.

    A read that fails on both sides adds to one counter on each side.
    ``n_alignments_drop_flank`` counts each dropped read once.
    """
    qc["n_alignments_drop_flank"] += int((~placement.kept).sum())
    for side, drops in (("upstream", placement.upstream_drop),
                        ("downstream", placement.downstream_drop)):
        for reason in SIDE_DROP_REASONS:
            qc[f"n_alignments_drop_{side}_{reason}"] += int((drops == reason).sum())
    qc["n_alignments_drop_flanks_crossed"] += int(placement.flanks_crossed.sum())


def _extract_locus(
    bam: pysam.AlignmentFile,
    chrom: str,
    bed_start: int,
    bed_end: int,
    *,
    geometry: LocusGeometry,
    min_mapq: int,
    drop_chimeric: bool,
    max_tlen: int,
    min_base_q: int,
) -> tuple[Counter, list[str], int, int, dict]:
    """Extract reads at one locus.

    Returns ``(read_counts, flanking_seqs, flanking_up_width,
    flanking_dn_width, qc)``. ``read_counts`` is a
    ``Counter[(mi, strand, seq, flanking_id)]`` of accepted reads
    (post-pair-collapse); ``flanking_seqs`` is the per-locus deduped
    flanking string list (indexed by ``flanking_id``), each of the form
    ``up_seq + "|" + dn_seq`` with ``.`` for uncovered/low-Q positions.
    ``flanking_up_width`` / ``flanking_dn_width`` are the per-locus
    extents observed.
    """
    qc = new_locus_qc()
    ref_orientation = geometry.ref_orientation
    lo, hi = geometry.flanked_interval

    # Wide fetch: ``bed ± max_tlen`` so the mate of a bed-spanning read is
    # visible even if its alignment lies fully outside the polyG bed. With
    # paired-end Illumina at insert sizes up to ``max_tlen``, a fragment's
    # far mate sits at most ``max_tlen`` bp from any base of its pair, so
    # this window captures every relevant mate. Bed-spanning vs
    # flanking-only is decided per-alignment by reference-interval
    # overlap; the flank placement runs only on bed-spanning mates, since
    # a flanking-only mate does not cover the repeat.
    fetch_start = max(0, bed_start - max_tlen)
    fetch_end = bed_end + max_tlen

    # First pass: keep every alignment that clears the alignment-level
    # filters, in fetch order, tagged bed-spanning or flanking-only. The
    # flank placement runs once over all bed-spanning reads of this locus.
    candidates: list[tuple[pysam.AlignedSegment, bool]] = []
    bed_alignments: list[pysam.AlignedSegment] = []

    for r, bed_overlap in filtered_alignments(
        bam, chrom, fetch_start, fetch_end, bed_start, bed_end,
        min_mapq=min_mapq, drop_chimeric=drop_chimeric, max_tlen=max_tlen, qc=qc,
    ):
        candidates.append((r, bed_overlap))
        if bed_overlap:
            bed_alignments.append(r)

    placement = place_locus_flanks(bed_alignments, geometry)
    _count_flank_drops(placement, qc)
    # Read back as Python lists: the pass below indexes them one read at
    # a time, and unboxing a numpy scalar per read costs more than the
    # whole conversion.
    kept = placement.kept.tolist()
    tracts = placement.tracts.tolist()
    tract_starts = (placement.upstream_run_side + 1).tolist()
    tract_ends = placement.downstream_run_side.tolist()

    # Second pass, still in fetch order: apply the flank and MI decisions
    # and group by query_name for pair-collapse. Each entry is
    #   (bed_mate, flanking_only_alignment)
    # with exactly one of the two set. `bed_mate` is a 6-tuple
    # ``(mi_prefix, strand, tract, up_side, dn_side, tract_source)``. A
    # flanking-only alignment is carried as-is and its flanking is
    # computed later, and only if the fragment turns out to have a
    # bed-spanning mate to attach it to: most flanking-only alignments in
    # the ``± max_tlen`` window belong to fragments that never reach the
    # polyG, and computing their flanking here is the single largest
    # source of wasted work at a locus.
    by_qn: dict[
        str,
        list[tuple[tuple | None, pysam.AlignedSegment | None]],
    ] = defaultdict(list)

    bed_index = 0
    for r, bed_overlap in candidates:
        if not bed_overlap:
            by_qn[r.query_name].append((None, r))
            continue

        i = bed_index
        bed_index += 1
        if not kept[i]:
            continue

        try:
            mi_tag = r.get_tag("MI")
        except KeyError:
            continue
        mi_prefix, _, strand = str(mi_tag).rpartition("/")
        if strand not in ("A", "B") or not mi_prefix:
            continue

        up_flank, dn_flank = _alignment_flanking(
            r, lo=lo, hi=hi, ref_orientation=ref_orientation, min_base_q=min_base_q,
        )
        tract_source = (r, tract_starts[i], tract_ends[i])
        by_qn[r.query_name].append(
            ((mi_prefix, strand, tracts[i], up_flank, dn_flank, tract_source), None)
        )

    # Pair-collapse: group by query_name. Drop qns with no bed-spanning
    # mate (the fragment never reached the polyG and can't be assigned an
    # MI/strand/length), then drop >2-mate groups. A flanking-only mate
    # that covers nothing outside the flanks is not a mate at all, so it
    # is dropped before the group-size test. Tract disagreement checks
    # only apply when both mates are bed-spanning: equal-length tracts
    # merge by base quality, and a tie or a length difference drops the
    # pair. Flanking sides are
    # pair-merged with per-position consensus (drop on disagreement); a
    # bed-spanning + flanking-only pair has no disagreement possible at
    # any single position the flanking-only mate covers — both feed into
    # the merge as today.
    accepted: list[tuple[str, str, str, FlankingSide, FlankingSide]] = []
    for _qn, mates in by_qn.items():
        bed_mates = [bed for bed, _ in mates if bed is not None]
        if not bed_mates:
            continue
        flank_sides: list[tuple[FlankingSide, FlankingSide]] = []
        for _bed, alignment in mates:
            if alignment is None:
                continue
            up_flank, dn_flank = _alignment_flanking(
                alignment, lo=lo, hi=hi, ref_orientation=ref_orientation,
                min_base_q=min_base_q,
            )
            if up_flank[1] or dn_flank[1]:
                flank_sides.append((up_flank, dn_flank))
        if len(bed_mates) + len(flank_sides) > 2:
            continue
        prefs = {m[0] for m in bed_mates}
        strands = {m[1] for m in bed_mates}
        if len(prefs) != 1 or len(strands) != 1:
            continue
        qc["n_read_pairs"] += 1
        tract = bed_mates[0][2]
        if len(bed_mates) == 2 and tract != bed_mates[1][2]:
            # Qualities are read only for pairs whose tracts differ.
            merged_seq = None
            if len(tract) == len(bed_mates[1][2]):
                merged_seq = _merge_tract_by_quality(
                    tract, bed_mates[1][2],
                    _tract_qualities(*bed_mates[0][5], ref_orientation),
                    _tract_qualities(*bed_mates[1][5], ref_orientation),
                )
            if merged_seq is None:
                qc["n_read_pairs_drop_disagree"] += 1
                continue
            qc["n_read_pairs_merged"] += 1
            tract = merged_seq
        up_merged, dn_merged = _merge_pair_flanking(
            [(m[3], m[4]) for m in bed_mates] + flank_sides
        )
        accepted.append((bed_mates[0][0], bed_mates[0][1], tract, up_merged, dn_merged))

    # Determine per-locus flanking widths: max distance seen across all
    # accepted reads, on each side.
    flanking_up_width = 0
    flanking_dn_width = 0
    for _mi, _strand, _tract, (up_start, up_text), (dn_start, dn_text) in accepted:
        up_extent = up_start + len(up_text)
        if up_extent > flanking_up_width:
            flanking_up_width = up_extent
        dn_extent = dn_start + len(dn_text)
        if dn_extent > flanking_dn_width:
            flanking_dn_width = dn_extent

    # Build per-read flanking strings, dedup, assemble Counter. Each side
    # is one covered run inside a `width`-wide field, so the string is
    # three slices of a shared "." run concatenated around it — no
    # per-position fill.
    #
    # Dedup on the two sides rather than on the built string: a side
    # carries no leading or trailing ".", so the run and its offset are
    # recoverable from the padded field and the two keys agree exactly.
    # Reads repeat their flanking often enough that keying on the sides
    # skips building most of the strings, which run to a kilobyte each.
    up_dots = "." * flanking_up_width
    dn_dots = "." * flanking_dn_width
    flanking_seqs: list[str] = []
    flanking_sides_to_id: dict[tuple[FlankingSide, FlankingSide], int] = {}
    read_counts: Counter = Counter()
    for mi_, strand_, tract_, up, dn in accepted:
        fid = flanking_sides_to_id.get((up, dn))
        if fid is None:
            up_start, up_text = up
            dn_start, dn_text = dn
            # The upstream field is written most-distant-first, so its
            # run goes in reversed and its `start` pads the right end.
            fid = len(flanking_seqs)
            flanking_seqs.append("".join((
                up_dots[:flanking_up_width - up_start - len(up_text)],
                up_text[::-1],
                up_dots[:up_start],
                "|",
                dn_dots[:dn_start],
                dn_text,
                dn_dots[:flanking_dn_width - dn_start - len(dn_text)],
            )))
            flanking_sides_to_id[(up, dn)] = fid
        read_counts[(mi_, strand_, tract_, fid)] += 1

    return read_counts, flanking_seqs, flanking_up_width, flanking_dn_width, qc


# ---------------------------------------------------------------------------
# Panel orchestration
# ---------------------------------------------------------------------------


def _prolyg_version() -> str:
    try:
        return _pkg_version("prolyG")
    except PackageNotFoundError:
        return "unknown"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_panel(
    bam_path: str | Path,
    bed_path: str | Path,
    flanks_path: str | Path,
    *,
    min_mapq: int = 40,
    drop_chimeric: bool = True,
    max_tlen: int = 1000,
    n_alleles_margin: int = 1,
    min_base_q: int = DEFAULT_MIN_FLANKING_BASE_Q,
    verbose: bool = False,
) -> ExtractedPanel:
    """Walk a BAM against a BED and produce an in-memory `ExtractedPanel`.

    Per locus:

    1. Reconstruct the reference window from MD tags, determine
       polyG-canonical orientation, and check both flanks from the flank
       table against it.
    2. If the window is recovered, fetch alignments overlapping the BED
       interval, apply parser-side QC filters, place both flanks in each
       read, pair-collapse R1+R2 by `query_name`, and emit per-(MI,
       strand, seq) read counts.

    `n_alleles` is set to `max_observed_run_length + n_alleles_margin`,
    where the max is taken across all parsed reads in all loci. Reads
    are not pre-filtered against `n_alleles` at extraction; the panel
    is faithful and the inference adapter (prolyG's `to_loci`) is what
    gates on the model's state space.

    Parameters
    ----------
    bam_path : path
        Sorted, indexed BAM. Must carry MD tags and MI tags.
    bed_path : path
        4-column BED: `chrom\\tstart\\tend\\tname`.
    flanks_path : path
        The per-side flank table that ``prolyg-phasing build-flanks``
        writes for this donor's matched normal. It must hold exactly the
        loci of the BED.
    min_mapq : int, default 40
        Drop alignments with `mapping_quality < min_mapq`.
    drop_chimeric : bool, default True
        Drop alignments with an SA tag.
    max_tlen : int, default 1000
        Drop alignments with `|TLEN| > max_tlen`.
    n_alleles_margin : int, default 1
        Headroom over `max_observed_run_length` when setting panel-
        wide `n_alleles`. `1` matches the current SD14 convention.
    verbose : bool, default False
        When True, show a tqdm progress bar over the BED iteration
        (one tick per locus). Silent by default to keep tests quiet
        and notebook output uncluttered.

    Returns
    -------
    ExtractedPanel

    Raises
    ------
    ValueError
        When the flank table and the BED do not hold the same loci, or a
        flank does not match the reference window of its locus.
    """
    bam_path = Path(bam_path)
    bed_path = Path(bed_path)
    flanks_path = Path(flanks_path)

    bed_rows = _load_bed(bed_path)
    flank_table = read_flank_table(flanks_path)
    bed_names = [name for name, *_ in bed_rows]
    missing = [name for name in bed_names if name not in flank_table.records]
    if missing:
        raise ValueError(f"{flanks_path}: no flanks for {len(missing)} BED loci: "
                         f"{missing[:10]}")
    extra = sorted(set(flank_table.records) - set(bed_names))
    if extra:
        raise ValueError(f"{flanks_path}: flanks for {len(extra)} loci not in the BED: "
                         f"{extra[:10]}")

    bam = pysam.AlignmentFile(str(bam_path), "rb")

    # One pass: extract a locus and immediately turn it into its
    # `ExtractedLocus`, so the locus's `Counter` — one 4-tuple key per
    # deduped row, and a full panel has millions of them — is released
    # before the next locus starts. Holding every locus's Counter across
    # a second pass kept those tuples alive to the end of the run, which
    # both raised peak memory and made every cyclic-collection pass walk
    # them again.
    loci: dict[str, ExtractedLocus] = {}
    max_observed_run_length = 0

    bed_iter = (
        tqdm(bed_rows, desc="extract_panel", unit="locus") if verbose else bed_rows
    )

    try:
        for name, chrom, bed_start, bed_end in bed_iter:
            flanks = flank_table.records[name]
            geometry = locus_geometry(bam, name, chrom, bed_start, bed_end, flanks,
                                      min_mapq=min_mapq)
            if geometry is None:
                counts: Counter = Counter()
                flanking_seqs: list[str] = []
                up_width = 0
                dn_width = 0
                qc = new_locus_qc()
            else:
                counts, flanking_seqs, up_width, dn_width, qc = _extract_locus(
                    bam, chrom, bed_start, bed_end,
                    geometry=geometry,
                    min_mapq=min_mapq,
                    drop_chimeric=drop_chimeric,
                    max_tlen=max_tlen,
                    min_base_q=min_base_q,
                )

            locus = _build_locus(
                name, chrom, bed_start, bed_end,
                flanks=flanks, geometry=geometry, counts=counts, qc=qc,
                flanking_seqs=flanking_seqs, up_width=up_width, dn_width=dn_width,
            )
            loci[name] = locus
            if locus.max_observed_run_length > max_observed_run_length:
                max_observed_run_length = locus.max_observed_run_length
    finally:
        bam.close()

    n_alleles = max_observed_run_length + n_alleles_margin

    provenance = ExtractionProvenance(
        bam_path=str(bam_path),
        bed_path=str(bed_path),
        flank_table_path=str(flanks_path),
        flank_table_sha256=_file_sha256(flanks_path),
        flank_table_sample=flank_table.sample,
        extraction_version=_prolyg_version(),
        extraction_date=datetime.date.today().isoformat(),
        min_mapq=min_mapq,
        drop_chimeric=drop_chimeric,
        max_tlen=max_tlen,
        pair_disagreement_policy=PAIR_DISAGREEMENT_POLICY,
        n_alleles_margin=n_alleles_margin,
        min_base_q=min_base_q,
        extraction_schema_version=EXTRACTION_SCHEMA_VERSION,
    )

    return ExtractedPanel(loci=loci, n_alleles=n_alleles, provenance=provenance)


def _build_locus(
    name: str,
    chrom: str,
    bed_start: int,
    bed_end: int,
    *,
    flanks: tuple[FlankRecord, FlankRecord],
    geometry: LocusGeometry | None,
    counts: Counter,
    qc: dict,
    flanking_seqs: list[str],
    up_width: int,
    dn_width: int,
) -> ExtractedLocus:
    """Turn one locus's extraction output into its `ExtractedLocus`.

    ``n_runs`` and ``max_observed_run_length`` are set from the **data**:
    the longest parsed tuple, and the longest G-run, across the accepted
    reads at this locus. The reference-derived ``reference_run_lengths``
    is kept as diagnostic context but is not authoritative for $S_i$ —
    the inference adapter (prolyG's ``to_loci``) recomputes the
    locus-level run count from the observed patterns, and extraction
    stays faithful with no pattern filter here. `geometry` is None when
    the BAM does not rebuild the reference window of the locus.
    """
    if counts:
        mi_arr = np.asarray([k[0] for k in counts], dtype=object)
        strand_arr = np.asarray([k[1] for k in counts], dtype="U1")
        seq_arr = np.asarray([k[2] for k in counts], dtype=object)
        flanking_id_arr = np.asarray([k[3] for k in counts], dtype=np.int32)
        count_arr = np.asarray(list(counts.values()), dtype=np.int64)
    else:
        mi_arr = np.empty(0, dtype=object)
        strand_arr = np.empty(0, dtype="U1")
        seq_arr = np.empty(0, dtype=object)
        flanking_id_arr = np.empty(0, dtype=np.int32)
        count_arr = np.empty(0, dtype=np.int64)

    flanking_seq_arr = np.asarray(flanking_seqs, dtype=object)

    if geometry is not None:
        reference_tract = geometry.reference_tract
        ref_run_lengths = parse_run_lengths(reference_tract)
        ref_orientation = geometry.ref_orientation
        g_walk_up, g_walk_dn = geometry.g_walk_up, geometry.g_walk_dn
        status = "recovered"
        # The flanking window starts past each flank's outer base. The
        # upstream field is written most-distant-first, so its start is
        # its most distant base.
        lo, hi = geometry.flanked_interval
        if ref_orientation == "+":
            flanking_up_ref_pos_start = lo - up_width
            flanking_dn_ref_pos_start = hi
        else:
            flanking_up_ref_pos_start = hi + up_width - 1
            flanking_dn_ref_pos_start = lo - 1
    else:
        reference_tract = ""
        ref_run_lengths = ()
        ref_orientation = ""
        g_walk_up = g_walk_dn = 0
        status = "unrecoverable"
        flanking_up_ref_pos_start = 0
        flanking_dn_ref_pos_start = 0

    per_locus_max_n_runs = 0
    per_locus_max = 0
    for s in seq_arr:
        runs = parse_run_lengths(str(s))
        if runs:
            if len(runs) > per_locus_max_n_runs:
                per_locus_max_n_runs = len(runs)
            m = max(runs)
            if m > per_locus_max:
                per_locus_max = m
    n_runs = per_locus_max_n_runs if per_locus_max_n_runs > 0 else 0

    return ExtractedLocus(
        mi=mi_arr,
        strand=strand_arr,
        seq=seq_arr,
        count=count_arr,
        flanking_id=flanking_id_arr,
        flanking_seq=flanking_seq_arr,
        n_runs=n_runs,
        reference_tract=reference_tract,
        reference_run_lengths=ref_run_lengths,
        chrom=chrom,
        bed_start=bed_start,
        bed_end=bed_end,
        bed_name=name,
        ref_orientation=ref_orientation,
        upstream_flank=flanks[0],
        downstream_flank=flanks[1],
        reference_window_status=status,
        max_observed_run_length=per_locus_max,
        n_reads=int(count_arr.sum()),
        g_walk_up=g_walk_up,
        g_walk_dn=g_walk_dn,
        flanking_up_width=up_width,
        flanking_dn_width=dn_width,
        flanking_up_ref_pos_start=flanking_up_ref_pos_start,
        flanking_dn_ref_pos_start=flanking_dn_ref_pos_start,
        **qc,
    )
