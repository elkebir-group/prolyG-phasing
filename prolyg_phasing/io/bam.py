"""BAM panel extraction — `extract_panel` entry point.

Walks a BAM against a BED panel and produces an in-memory
`ExtractedPanel` of faithful per-(MI, strand, seq) read counts.

Two passes:

1. **Anchorability pass** — per BED interval, reconstruct a `±pad` bp
   reference window from MD tags and try to construct upstream +
   downstream anchors (`_anchors.build_anchors`).
2. **Extraction pass** — per anchorable locus, fetch overlapping
   alignments, apply parser-side QC filters (MAPQ, SA, TLEN, anchor
   placement), group by `query_name` for pair-collapse, drop pair
   disagreements, and emit per-(MI, strand, seq) read counts.

Per-locus ``n_runs`` is set from the **data** (max parsed tuple length
across accepted reads), not from the reference inter-anchor sequence.
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

import datetime
from collections import Counter, defaultdict
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

import numpy as np
import pysam
from tqdm.auto import tqdm

from prolyg_phasing.io._anchors import (
    AnchorResult,
    build_anchors,
    get_reference_window,
    hamming_search,
    revcomp,
)
from prolyg_phasing.io.panel import (
    ExtractedLocus,
    ExtractedPanel,
    ExtractionProvenance,
    parse_run_lengths,
)

FLANKING_SEQ_LEN = 10
REF_WINDOW_PAD = 50

EXTRACTION_SCHEMA_VERSION = "2"
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
# Per-locus anchorability + extraction
# ---------------------------------------------------------------------------


def _build_anchors_for_locus(
    bam: pysam.AlignmentFile,
    chrom: str,
    bed_start: int,
    bed_end: int,
    *,
    pad: int,
    k: int,
    min_mapq: int,
) -> AnchorResult:
    """Anchor extraction for one BED interval: MD-tag reference + anchor walk."""
    win_start = bed_start - pad
    win_end = bed_end + pad
    ref = get_reference_window(bam, chrom, win_start, win_end, min_mapq=min_mapq)
    return build_anchors(
        ref_window=ref,
        bed_offset=pad,
        bed_len=bed_end - bed_start,
        k=k,
    )


# One side (upstream or downstream) of one read's flanking bases, as
# ``(start_distance, bases)``: ``bases[i]`` is the canonical-orientation
# base at distance ``start_distance + i`` from that side's anchor, going
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
    bed_start: int,
    bed_end: int,
    g_walk_up: int,
    g_walk_dn: int,
    anchor_k: int,
    ref_orientation: str,
    min_base_q: int,
) -> tuple[FlankingSide, FlankingSide]:
    """Flanking bases for one alignment, by canonical distance from the anchor.

    Returns ``(upstream_side, downstream_side)`` as `FlankingSide` pairs.
    Distance 0 on each side is the base immediately adjacent to that
    side's anchor going outward (away from the polyG). Bases are in
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
    # of the anchors (`refpos < lo`) and one right of them (`refpos >=
    # hi`). Orientation decides which region is the upstream side, which
    # way distances run, and whether bases are complemented.
    if ref_orientation == "+":
        lo = bed_start - g_walk_up - anchor_k
        hi = bed_end + g_walk_dn + anchor_k
        bases = seq.upper()
    else:
        lo = bed_start - g_walk_dn - anchor_k
        hi = bed_end + g_walk_up + anchor_k
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


def _extract_locus(
    bam: pysam.AlignmentFile,
    chrom: str,
    bed_start: int,
    bed_end: int,
    *,
    ref_orientation: str,
    anchor_up: str,
    anchor_dn: str,
    anchor_k: int,
    g_walk_up: int,
    g_walk_dn: int,
    min_mapq: int,
    drop_chimeric: bool,
    max_tlen: int,
    anchor_hamming_max: int,
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
    qc = {
        "n_alignments_overlap_locus": 0,
        "n_alignments_drop_mapq": 0,
        "n_alignments_drop_sa": 0,
        "n_alignments_drop_tlen": 0,
        "n_alignments_drop_anchor": 0,
        "n_read_pairs": 0,
        "n_read_pairs_drop_disagree": 0,
    }

    # Wide fetch: ``bed ± max_tlen`` so the mate of a bed-spanning read is
    # visible even if its alignment lies fully outside the polyG bed. With
    # paired-end Illumina at insert sizes up to ``max_tlen``, a fragment's
    # far mate sits at most ``max_tlen`` bp from any base of its pair, so
    # this window captures every relevant mate. Bed-spanning vs
    # flanking-only is decided per-alignment by reference-interval
    # overlap; the anchor check runs only on bed-spanning mates, since a
    # flanking-only mate cannot contain both anchors by construction.
    fetch_start = max(0, bed_start - max_tlen)
    fetch_end = bed_end + max_tlen

    # First pass: keep every alignment that clears the alignment-level
    # filters, in fetch order, tagged bed-spanning or flanking-only. The
    # anchor placement is deferred to one batched Hamming search over all
    # bed-spanning reads of this locus, so it is not decided here.
    candidates: list[tuple[pysam.AlignedSegment, str | None]] = []
    bed_seqs: list[str] = []

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
            seq = r.query_sequence.upper()
            if ref_orientation == "-":
                seq = revcomp(seq)
            candidates.append((r, seq))
            bed_seqs.append(seq)
        else:
            candidates.append((r, None))

    # Anchor placement for every bed-spanning read of this locus at once.
    # Read back as Python lists: the pass below indexes them one read at
    # a time, and unboxing a numpy scalar per read costs more than the
    # whole conversion.
    up_distances, up_offsets = hamming_search(bed_seqs, anchor_up)
    dn_distances, dn_offsets = hamming_search(bed_seqs, anchor_dn)
    up_distances = up_distances.tolist()
    up_offsets = up_offsets.tolist()
    dn_distances = dn_distances.tolist()
    dn_offsets = dn_offsets.tolist()

    # Second pass, still in fetch order: apply the anchor and MI decisions
    # and group by query_name for pair-collapse. Each entry is
    #   (bed_mate, flanking_only_alignment)
    # with exactly one of the two set. `bed_mate` is a 5-tuple
    # ``(mi_prefix, strand, ia_seq, up_side, dn_side)``. A flanking-only
    # alignment is carried as-is and its flanking is computed later, and
    # only if the fragment turns out to have a bed-spanning mate to
    # attach it to: most flanking-only alignments in the ``± max_tlen``
    # window belong to fragments that never reach the polyG, and
    # computing their flanking here is the single largest source of
    # wasted work at a locus.
    by_qn: dict[
        str,
        list[tuple[tuple | None, pysam.AlignedSegment | None]],
    ] = defaultdict(list)

    bed_index = 0
    for r, seq in candidates:
        if seq is None:
            by_qn[r.query_name].append((None, r))
            continue

        i = bed_index
        bed_index += 1
        if up_distances[i] > anchor_hamming_max or dn_distances[i] > anchor_hamming_max:
            qc["n_alignments_drop_anchor"] += 1
            continue

        ia_start = up_offsets[i] + anchor_k
        ia_end = dn_offsets[i]
        if ia_end < ia_start:
            qc["n_alignments_drop_anchor"] += 1
            continue
        ia_seq = seq[ia_start:ia_end]

        try:
            mi_tag = r.get_tag("MI")
        except KeyError:
            continue
        mi_prefix, _, strand = str(mi_tag).rpartition("/")
        if strand not in ("A", "B") or not mi_prefix:
            continue

        up_flank, dn_flank = _alignment_flanking(
            r,
            bed_start=bed_start, bed_end=bed_end,
            g_walk_up=g_walk_up, g_walk_dn=g_walk_dn,
            anchor_k=anchor_k,
            ref_orientation=ref_orientation,
            min_base_q=min_base_q,
        )
        by_qn[r.query_name].append(((mi_prefix, strand, ia_seq, up_flank, dn_flank), None))

    # Pair-collapse: group by query_name. Drop qns with no bed-spanning
    # mate (the fragment never reached the polyG and can't be assigned an
    # MI/strand/length), then drop >2-mate groups. A flanking-only mate
    # that covers nothing outside the anchors is not a mate at all, so it
    # is dropped before the group-size test. Pair-disagreement checks
    # only apply when both mates are bed-spanning. Flanking sides are
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
                alignment,
                bed_start=bed_start, bed_end=bed_end,
                g_walk_up=g_walk_up, g_walk_dn=g_walk_dn,
                anchor_k=anchor_k,
                ref_orientation=ref_orientation,
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
        if len(bed_mates) == 2 and bed_mates[0][2] != bed_mates[1][2]:
            qc["n_read_pairs_drop_disagree"] += 1
            continue
        up_merged, dn_merged = _merge_pair_flanking(
            [(m[3], m[4]) for m in bed_mates] + flank_sides
        )
        accepted.append((bed_mates[0][0], bed_mates[0][1], bed_mates[0][2], up_merged, dn_merged))

    # Determine per-locus flanking widths: max distance seen across all
    # accepted reads, on each side.
    flanking_up_width = 0
    flanking_dn_width = 0
    for _mi, _strand, _ia, (up_start, up_text), (dn_start, dn_text) in accepted:
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
    for mi_, strand_, ia_, up, dn in accepted:
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
        read_counts[(mi_, strand_, ia_, fid)] += 1

    return read_counts, flanking_seqs, flanking_up_width, flanking_dn_width, qc


# ---------------------------------------------------------------------------
# Panel orchestration
# ---------------------------------------------------------------------------


def _prolyg_version() -> str:
    try:
        return _pkg_version("prolyG")
    except PackageNotFoundError:
        return "unknown"


def extract_panel(
    bam_path: str | Path,
    bed_path: str | Path,
    *,
    min_mapq: int = 40,
    drop_chimeric: bool = True,
    max_tlen: int = 1000,
    anchor_hamming_max: int = 1,
    pair_disagreement_policy: str = "drop",
    n_alleles_margin: int = 1,
    min_base_q: int = DEFAULT_MIN_FLANKING_BASE_Q,
    verbose: bool = False,
) -> ExtractedPanel:
    """Walk a BAM against a BED and produce an in-memory `ExtractedPanel`.

    Two passes per locus:

    1. Reconstruct the reference window from MD tags, determine
       polyG-canonical orientation, and try to build upstream +
       downstream anchors.
    2. If anchorable, fetch alignments overlapping the BED interval,
       apply parser-side QC filters, pair-collapse R1+R2 by
       `query_name`, and emit per-(MI, strand, seq) read counts.

    `n_alleles` is set to `max_observed_run_length + n_alleles_margin`,
    where the max is taken across all parsed reads in all loci. Reads
    are not pre-filtered against `n_alleles` at extraction; the panel
    is faithful and the inference adapter (`to_loci`) is what gates
    on the model's state space.

    Parameters
    ----------
    bam_path : path
        Sorted, indexed BAM. Must carry MD tags and MI tags.
    bed_path : path
        4-column BED: `chrom\\tstart\\tend\\tname`.
    min_mapq : int, default 40
        Drop alignments with `mapping_quality < min_mapq`.
    drop_chimeric : bool, default True
        Drop alignments with an SA tag.
    max_tlen : int, default 1000
        Drop alignments with `|TLEN| > max_tlen`.
    anchor_hamming_max : int, default 1
        Cap on per-anchor Hamming distance for accepted placement.
    pair_disagreement_policy : str, default "drop"
        Currently only ``"drop"`` is supported. R1+R2 pairs with
        different inter-anchor `seq` are dropped and counted in
        `n_read_pairs_drop_disagree`. Future policies may add
        ``"r1"`` / ``"keep_both"`` on demand.
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
    """
    if pair_disagreement_policy != "drop":
        raise NotImplementedError(
            f"pair_disagreement_policy={pair_disagreement_policy!r} not supported; "
            "only 'drop' is implemented."
        )

    bam_path = Path(bam_path)
    bed_path = Path(bed_path)

    bed_rows = _load_bed(bed_path)

    bam = pysam.AlignmentFile(str(bam_path), "rb")

    # Per-locus read counts + anchor info + QC, accumulated.
    per_locus_counts: dict[str, Counter] = {}
    per_locus_anchor: dict[str, AnchorResult] = {}
    per_locus_qc: dict[str, dict] = {}
    per_locus_bed: dict[str, tuple[str, int, int]] = {}
    per_locus_flanking_seqs: dict[str, list[str]] = {}
    per_locus_flanking_up_width: dict[str, int] = {}
    per_locus_flanking_dn_width: dict[str, int] = {}

    max_observed_run_length = 0

    bed_iter = (
        tqdm(bed_rows, desc="extract_panel", unit="locus") if verbose else bed_rows
    )

    try:
        for name, chrom, bed_start, bed_end in bed_iter:
            per_locus_bed[name] = (chrom, bed_start, bed_end)

            anchor = _build_anchors_for_locus(
                bam, chrom, bed_start, bed_end,
                pad=REF_WINDOW_PAD, k=FLANKING_SEQ_LEN, min_mapq=min_mapq,
            )
            per_locus_anchor[name] = anchor

            if anchor.status != "anchorable":
                per_locus_counts[name] = Counter()
                per_locus_qc[name] = {
                    "n_alignments_overlap_locus": 0,
                    "n_alignments_drop_mapq": 0,
                    "n_alignments_drop_sa": 0,
                    "n_alignments_drop_tlen": 0,
                    "n_alignments_drop_anchor": 0,
                    "n_read_pairs": 0,
                    "n_read_pairs_drop_disagree": 0,
                }
                per_locus_flanking_seqs[name] = []
                per_locus_flanking_up_width[name] = 0
                per_locus_flanking_dn_width[name] = 0
                continue

            read_counts, flanking_seqs, up_width, dn_width, qc = _extract_locus(
                bam, chrom, bed_start, bed_end,
                ref_orientation=anchor.ref_orientation,
                anchor_up=anchor.upstream_anchor,
                anchor_dn=anchor.downstream_anchor,
                anchor_k=FLANKING_SEQ_LEN,
                g_walk_up=anchor.g_walk_up,
                g_walk_dn=anchor.g_walk_dn,
                min_mapq=min_mapq,
                drop_chimeric=drop_chimeric,
                max_tlen=max_tlen,
                anchor_hamming_max=anchor_hamming_max,
                min_base_q=min_base_q,
            )
            per_locus_counts[name] = read_counts
            per_locus_qc[name] = qc
            per_locus_flanking_seqs[name] = flanking_seqs
            per_locus_flanking_up_width[name] = up_width
            per_locus_flanking_dn_width[name] = dn_width
    finally:
        bam.close()

    # Build ExtractedLocus per BED row. The panel-wide
    # `max_observed_run_length` is the running maximum of the per-locus
    # ones, taken from the same parse this loop already needs.
    loci: dict[str, ExtractedLocus] = {}
    for name, chrom, bed_start, bed_end in bed_rows:
        anchor = per_locus_anchor[name]
        counts = per_locus_counts[name]
        qc = per_locus_qc[name]

        if anchor.ref_inter_anchor_seq:
            ref_run_lengths = parse_run_lengths(anchor.ref_inter_anchor_seq)
        else:
            ref_run_lengths = ()

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

        flanking_seqs = per_locus_flanking_seqs[name]
        flanking_seq_arr = np.asarray(flanking_seqs, dtype=object)
        up_width = per_locus_flanking_up_width[name]
        dn_width = per_locus_flanking_dn_width[name]

        if anchor.status == "anchorable":
            if anchor.ref_orientation == "+":
                flanking_up_ref_pos_start = (
                    bed_start - anchor.g_walk_up - FLANKING_SEQ_LEN - up_width
                )
                flanking_dn_ref_pos_start = (
                    bed_end + anchor.g_walk_dn + FLANKING_SEQ_LEN
                )
            else:
                flanking_up_ref_pos_start = (
                    bed_end + anchor.g_walk_up + FLANKING_SEQ_LEN + up_width - 1
                )
                flanking_dn_ref_pos_start = (
                    bed_start - anchor.g_walk_dn - FLANKING_SEQ_LEN - 1
                )
        else:
            flanking_up_ref_pos_start = 0
            flanking_dn_ref_pos_start = 0

        # ``n_runs`` is set from the data: max parsed tuple length across all
        # accepted reads at this locus. The reference-derived
        # ``reference_run_lengths`` is kept as diagnostic context but is no
        # longer authoritative for $S_i$. The inference adapter
        # (``ExtractedPanel.to_loci``) recomputes the locus-level max run
        # count from the observed patterns; extraction stays faithful (no
        # pattern filter here).
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
        if per_locus_max > max_observed_run_length:
            max_observed_run_length = per_locus_max

        loci[name] = ExtractedLocus(
            mi=mi_arr,
            strand=strand_arr,
            seq=seq_arr,
            count=count_arr,
            flanking_id=flanking_id_arr,
            flanking_seq=flanking_seq_arr,
            n_runs=n_runs,
            ref_inter_anchor_seq=anchor.ref_inter_anchor_seq,
            reference_run_lengths=ref_run_lengths,
            chrom=chrom,
            bed_start=bed_start,
            bed_end=bed_end,
            bed_name=name,
            ref_orientation=anchor.ref_orientation,
            upstream_anchor=anchor.upstream_anchor,
            downstream_anchor=anchor.downstream_anchor,
            anchorability_status=anchor.status,
            max_observed_run_length=per_locus_max,
            n_alignments_overlap_locus=qc["n_alignments_overlap_locus"],
            n_alignments_drop_mapq=qc["n_alignments_drop_mapq"],
            n_alignments_drop_sa=qc["n_alignments_drop_sa"],
            n_alignments_drop_tlen=qc["n_alignments_drop_tlen"],
            n_alignments_drop_anchor=qc["n_alignments_drop_anchor"],
            n_read_pairs=qc["n_read_pairs"],
            n_read_pairs_drop_disagree=qc["n_read_pairs_drop_disagree"],
            n_reads=int(count_arr.sum()),
            g_walk_up=anchor.g_walk_up,
            g_walk_dn=anchor.g_walk_dn,
            flanking_up_width=up_width,
            flanking_dn_width=dn_width,
            flanking_up_ref_pos_start=flanking_up_ref_pos_start,
            flanking_dn_ref_pos_start=flanking_dn_ref_pos_start,
        )

    n_alleles = max_observed_run_length + n_alleles_margin

    provenance = ExtractionProvenance(
        bam_path=str(bam_path),
        bed_path=str(bed_path),
        extraction_version=_prolyg_version(),
        extraction_date=datetime.date.today().isoformat(),
        min_mapq=min_mapq,
        drop_chimeric=drop_chimeric,
        max_tlen=max_tlen,
        anchor_hamming_max=anchor_hamming_max,
        pair_disagreement_policy=pair_disagreement_policy,
        n_alleles_margin=n_alleles_margin,
        min_base_q=min_base_q,
        extraction_schema_version=EXTRACTION_SCHEMA_VERSION,
    )

    return ExtractedPanel(loci=loci, n_alleles=n_alleles, provenance=provenance)
