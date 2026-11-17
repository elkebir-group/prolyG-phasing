"""Internal reference utilities for BAM extraction.

Two pieces:

1. Reference reconstruction (`get_reference_window`) — pull a window of
   the reference genome from BAM MD tags, no external FASTA required.
2. Reference walk (`reference_walk`) — the locus orientation, and the G
   bases between each BED end and the first non-G reference base.

All functions are pysam-coupled only where they need to read alignment
records; the walk itself is pure string work.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pysam


COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s: str) -> str:
    """Reverse-complement of `s` over IUPAC bases ACGTN (case-preserving)."""
    return s.translate(COMP)[::-1]


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
# Reference walk
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ReferenceWalk:
    """The strand of a locus's G run and the G bases past each BED end.

    `ref_orientation` is "+" when the BED interval holds at least as many G
    as C on the reference, else "-". `g_walk_up` and `g_walk_dn` count the
    G bases, in polyG orientation, between each BED end and the first
    non-G reference base outward (or the window end).
    """

    ref_orientation: str
    g_walk_up: int
    g_walk_dn: int


def determine_orientation(ref_window: str, bed_offset: int, bed_len: int) -> str:
    """`+` if the BED interval is G-dominant on the reference, else `-`."""
    bed = ref_window[bed_offset : bed_offset + bed_len].upper()
    return "+" if bed.count("G") >= bed.count("C") else "-"


def reference_walk(ref_window: str, bed_offset: int, bed_len: int) -> ReferenceWalk:
    """The strand and the G walk past each BED end, from the reference window.

    `ref_window` is the reference on ``[bed_start - pad, bed_end + pad)`` in
    reference orientation, and `bed_offset` is `pad`.
    """
    orientation = determine_orientation(ref_window, bed_offset, bed_len)
    canon = ref_window if orientation == "+" else revcomp(ref_window)
    bed_start = bed_offset if orientation == "+" else len(canon) - (bed_offset + bed_len)
    bed_end = bed_start + bed_len
    i = bed_start - 1
    while i >= 0 and canon[i] == "G":
        i -= 1
    j = bed_end
    while j < len(canon) and canon[j] == "G":
        j += 1
    return ReferenceWalk(orientation, bed_start - 1 - i, j - bed_end)
