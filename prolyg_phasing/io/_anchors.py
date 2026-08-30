"""Internal anchor utilities for BAM extraction.

Three pieces:

1. Reference reconstruction (`get_reference_window`) — pull a window of
   the reference genome from BAM MD tags, no external FASTA required.
2. Anchor construction (`build_anchors`) — walk boundary Gs out from
   the BED interval to the first non-G, then take an additional `k` bp
   to form the upstream and downstream anchor strings.
3. Anchor search in reads (`hamming_search`) — slide an anchor along
   every read of a batch and return each read's lowest-Hamming-distance
   placement.

All functions are pysam-coupled only where they need to read alignment
records; the anchor-construction logic itself is pure string work.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
from numpy.lib.stride_tricks import as_strided

if TYPE_CHECKING:
    import pysam


COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s: str) -> str:
    """Reverse-complement of `s` over IUPAC bases ACGTN (case-preserving)."""
    return s.translate(COMP)[::-1]


# Reads are scored against an anchor in blocks of this many at a time.
# The whole point of the batch is to pay numpy's per-call overhead once
# for many reads instead of once per read, so the block wants to be
# large; the cap keeps the (block, offsets, k) comparison buffer to a few
# MB regardless of how deep the locus is.
_HAMMING_BLOCK = 4096


def hamming_search(read_seqs: list[str], anchor: str) -> tuple[np.ndarray, np.ndarray]:
    """Slide `anchor` across every read in `read_seqs`; return per-read `(distance, offset)`.

    Reads that carry `anchor` verbatim are settled by a substring
    search. The rest are scored at every offset, with one set of numpy
    calls per block of reads rather than one set per read.

    Parameters
    ----------
    read_seqs : list of str
        Reads to search, upper-case ASCII. Lengths may differ.
    anchor : str
        The anchor to place, upper-case ASCII.

    Returns
    -------
    distances : ndarray of int64, shape (len(read_seqs),)
        Lowest Hamming distance of `anchor` against each read.
    offsets : ndarray of int64, shape (len(read_seqs),)
        Offset into each read at which that distance is attained,
        leftmost on a tie.

    A read shorter than `anchor` admits no placement and scores
    ``(len(anchor), -1)``.
    """
    k = len(anchor)
    n = len(read_seqs)
    distances = np.full(n, k, dtype=np.int64)
    offsets = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return distances, offsets

    # An exact occurrence of the anchor is a distance-0 placement, so it
    # is a global minimum, and `str.find` returns the leftmost one — the
    # same tie-break the sliding scan applies. Most reads carry the
    # anchor verbatim, and `find` is a fast substring search, so this
    # leaves only the minority of reads to be scored position by
    # position. Reads shorter than the anchor never match and fall
    # through to the scan, which scores them ``(k, -1)``.
    exact = np.fromiter((s.find(anchor) for s in read_seqs), dtype=np.int64, count=n)
    matched = exact >= 0
    distances[matched] = 0
    offsets[matched] = exact[matched]

    to_scan = np.flatnonzero(~matched)
    if to_scan.size == 0:
        return distances, offsets

    scan_seqs = [read_seqs[i] for i in to_scan.tolist()]
    anchor_bytes = np.frombuffer(anchor.encode("ascii"), dtype=np.uint8)
    lengths = np.fromiter((len(s) for s in scan_seqs), dtype=np.int64, count=len(scan_seqs))

    for lo in range(0, len(scan_seqs), _HAMMING_BLOCK):
        hi = min(lo + _HAMMING_BLOCK, len(scan_seqs))
        block = scan_seqs[lo:hi]
        block_lengths = lengths[lo:hi]
        width = int(block_lengths.max())
        if width < k:
            continue  # every read in the block is shorter than the anchor
        n_offsets = width - k + 1

        # One ragged-to-rectangular pack per block: right-pad every read
        # to the block's longest, concatenate, and view as (n_block, width).
        flat = "".join(s.ljust(width, "\0") for s in block).encode("ascii")
        matrix = np.frombuffer(flat, dtype=np.uint8).reshape(len(block), width)
        windows = as_strided(
            matrix,
            shape=(len(block), n_offsets, k),
            strides=(width, 1, 1),
            writeable=False,
        )
        block_distances = (windows != anchor_bytes).sum(axis=2, dtype=np.int16)

        # An offset past a read's own last full-length window scored
        # against padding, so rule it out: k + 1 exceeds every real
        # distance (which is at most k), and argmin takes the leftmost
        # minimum, so the winner is always the leftmost real best.
        past_end = np.arange(n_offsets)[None, :] > (block_lengths - k)[:, None]
        block_distances[past_end] = k + 1

        best = block_distances.argmin(axis=1)
        rows = np.arange(len(block))
        placeable = block_lengths >= k
        block_rows = to_scan[lo:hi]
        distances[block_rows] = np.where(placeable, block_distances[rows, best], k)
        offsets[block_rows] = np.where(placeable, best, -1)

    return distances, offsets


# ---------------------------------------------------------------------------
# Reference window from MD tags
# ---------------------------------------------------------------------------


def get_reference_window(
    bam: pysam.AlignmentFile,
    chrom: str,
    win_start: int,
    win_end: int,
    *,
    min_mapq: int,
) -> str | None:
    """Reconstruct reference on ``[win_start, win_end)`` from any clean spanning read.

    Uses the BAM MD tag (via `pysam.AlignedSegment.get_reference_sequence()`)
    to recover the reference without an external FASTA. Iterates the
    fetch results and returns the first read that:

    - is primary (not secondary / supplementary / unmapped),
    - passes `min_mapq`,
    - is not chimeric (no SA tag),
    - has an MD tag,
    - fully spans the window (`reference_start <= win_start`,
      `reference_end >= win_end`),
    - has no soft-clip CIGAR ops (op == 4).

    Returns the window in upper-case, or `None` if no clean spanning
    read exists.
    """
    for r in bam.fetch(chrom, win_start, win_end):
        if r.is_secondary or r.is_supplementary or r.is_unmapped:
            continue
        if r.mapping_quality < min_mapq:
            continue
        if r.has_tag("SA"):
            continue
        if not r.has_tag("MD"):
            continue
        if r.reference_start is None or r.reference_end is None:
            continue
        if r.reference_start > win_start or r.reference_end < win_end:
            continue
        if r.cigartuples and any(op == 4 for op, _ in r.cigartuples):
            continue
        try:
            ref = r.get_reference_sequence()
        except Exception:
            continue
        return ref[win_start - r.reference_start : win_end - r.reference_start].upper()
    return None


# ---------------------------------------------------------------------------
# Anchor construction
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AnchorResult:
    """Output of :func:`build_anchors`.

    `status` is one of:

    - `"anchorable"` — both anchors constructed.
    - `"anchor_unstable_upstream"` — upstream anchor cannot be constructed.
    - `"anchor_unstable_downstream"` — downstream anchor cannot be constructed.
    - `"anchor_unstable_both"` — neither anchor.
    - `"ref_window_unrecoverable"` — caller passed `None` for `ref_window`.

    `ref_orientation` is `"+"` if the BED interval is G-dominant on the
    reference (no revcomp needed for polyG-canonical view), `"-"` if
    C-dominant (revcomp the read to canonicalize). Empty when status
    is `ref_window_unrecoverable`.

    `ref_inter_anchor_seq` is the canonical-orientation reference
    sequence strictly between the upstream and downstream anchors —
    the polyG region as the reference says it should be. Empty when
    either anchor failed.
    """

    status: str
    ref_orientation: str
    upstream_anchor: str
    downstream_anchor: str
    ref_inter_anchor_seq: str
    g_walk_up: int
    g_walk_dn: int


def determine_orientation(ref_window: str, bed_offset: int, bed_len: int) -> str:
    """`+` if the BED interval is G-dominant on the reference, else `-`."""
    bed = ref_window[bed_offset : bed_offset + bed_len].upper()
    return "+" if bed.count("G") >= bed.count("C") else "-"


def build_anchors(
    ref_window: str | None,
    bed_offset: int,
    bed_len: int,
    *,
    k: int,
) -> AnchorResult:
    """Build upstream + downstream anchors from a `±pad`-bp reference window.

    The walk: starting at the BED interval boundary on the
    canonical-orientation reference, step outward through Gs until the
    first non-G base. Take `k` bp further outward as the anchor; the
    anchor ends (upstream) or starts (downstream) at the first non-G
    base — i.e., the interrupter / linker base.

    `ref_window` is the reference on `[bed_start - pad, bed_end + pad)`
    in **reference orientation** (not yet canonicalized). The function
    determines orientation and reverse-complements as needed.

    Parameters
    ----------
    ref_window : str or None
        Reference sequence over the BED interval ±pad. Pass `None`
        when MD-tag reconstruction failed (returns status
        `"ref_window_unrecoverable"`).
    bed_offset : int
        Offset of the BED interval start within `ref_window`. Equals
        the `pad` used when fetching the window.
    bed_len : int
        Length of the BED interval (`bed_end - bed_start`).
    k : int
        Anchor length in bp (typically 10).

    Returns
    -------
    AnchorResult
    """
    if ref_window is None:
        return AnchorResult(
            status="ref_window_unrecoverable",
            ref_orientation="",
            upstream_anchor="",
            downstream_anchor="",
            ref_inter_anchor_seq="",
            g_walk_up=0,
            g_walk_dn=0,
        )

    orientation_letter = determine_orientation(ref_window, bed_offset, bed_len)
    if orientation_letter == "+":
        canon = ref_window
        bed_start_c = bed_offset
        bed_end_c = bed_offset + bed_len
    else:
        canon = revcomp(ref_window)
        win_len = len(canon)
        bed_start_c = win_len - (bed_offset + bed_len)
        bed_end_c = win_len - bed_offset

    win_len = len(canon)

    # Upstream walk: step from bed_start_c - 1 leftward through Gs.
    i = bed_start_c - 1
    walk_up = 0
    while i >= 0 and canon[i] == "G":
        i -= 1
        walk_up += 1
    if i < 0:
        anchor_up = ""
    else:
        anchor_up_start = i - k + 1
        if anchor_up_start < 0:
            anchor_up = ""
        else:
            anchor_up = canon[anchor_up_start : i + 1]

    # Downstream walk: step from bed_end_c rightward through Gs.
    j = bed_end_c
    walk_dn = 0
    while j < win_len and canon[j] == "G":
        j += 1
        walk_dn += 1
    if j >= win_len:
        anchor_dn = ""
    else:
        anchor_dn_end = j + k
        if anchor_dn_end > win_len:
            anchor_dn = ""
        else:
            anchor_dn = canon[j : anchor_dn_end]

    up_ok = bool(anchor_up)
    dn_ok = bool(anchor_dn)
    if up_ok and dn_ok:
        status = "anchorable"
    elif not up_ok and not dn_ok:
        status = "anchor_unstable_both"
    elif not up_ok:
        status = "anchor_unstable_upstream"
    else:
        status = "anchor_unstable_downstream"

    # Reference inter-anchor sequence: from end of upstream anchor to start
    # of downstream anchor. Only well-defined when both anchors built.
    if up_ok and dn_ok:
        # i+1 is the first base after the upstream anchor (start of the
        # polyG region). j is the first base of the downstream anchor.
        ref_inter_anchor_seq = canon[i + 1 : j]
    else:
        ref_inter_anchor_seq = ""

    return AnchorResult(
        status=status,
        ref_orientation=orientation_letter,
        upstream_anchor=anchor_up,
        downstream_anchor=anchor_dn,
        ref_inter_anchor_seq=ref_inter_anchor_seq,
        g_walk_up=walk_up,
        g_walk_dn=walk_dn,
    )
