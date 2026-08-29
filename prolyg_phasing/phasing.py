"""Per-locus SNV discovery + flanking-haplotype phasing.

Consumes the per-locus deduplicated flanking strings emitted by
``prolyg_phasing.io.bam._extract_locus``. Each flanking string has the form
``up + "|" + dn``, fixed length per locus, padded with ``'.'`` at
uncovered positions; per-row ``count`` weights rows sharing the same
``flanking_id``.

SNV discovery: :func:`find_informative_snvs` calls per-position
major/minor alleles passing coverage + VAF filters. ``N`` is invisible
to depth and to allele identity. Tri-allelic positions silently drop
the third base.

Haplotype phasing: :func:`determine_flanking_haplotypes` phases the
informative SNVs at each locus into two haplotype profiles
(``hap_major_profile`` / ``hap_minor_profile``). The anchor SNV
(highest per-position coverage) defines the major/minor labels;
other SNVs are flipped if their per-position major co-segregates
with the anchor's minor. SNVs with insufficient double-coverage
against the anchor are dropped.

Per-read haplotype assignment (:func:`assign_flanking_haplotypes`) is
downstream.

"Haplotype" here refers exclusively to the *flanking-SNV-defined*
locus partition (label set ``{hap_major, hap_minor}``). The
interrupter-pattern haplotype $h \\in \\mathcal{H}_i$ is a separate,
orthogonal evidence source (the multi-run length-vector notation).

Every label this module produces is a direct readout of a read's own
sequence. The anchored counterpart, which re-votes one panel's reads
against a *different*, already-phased panel's haplotype profile, lives in
:mod:`prolyg_phasing.ref_phasing` instead — a caller-supplied pairing
(e.g. a tumor bulk sample voted against its matched normal), not a concept
this module knows about.
"""
from __future__ import annotations

import dataclasses
import pickle
from pathlib import Path
from typing import Self

import numpy as np
from scipy import sparse

from prolyg_phasing.io.panel import ExtractedLocus, ExtractedPanel

_BASES = ("A", "C", "G", "T", "N", ".")
_N_COLS = len(_BASES)
_ACGT = ("A", "C", "G", "T")

#: Family × position cells per consensus block. Bounds the dense ``(n_fam, block, 4)``
#: accumulator :func:`family_consensus_counts` reduces over: at 4 × 8 bytes a cell,
#: 2**22 cells is a ~134 MB peak regardless of how deep the locus is, and the block
#: never changes the result -- only how much of the window is scored at a time.
CONSENSUS_POSITION_BLOCK = 2 ** 22


def flanking_char_matrix(flanking_seq: np.ndarray, total_width: int) -> np.ndarray:
    """Expand unique flanking strings to an ``(n_flanking, total_width)`` U1 matrix.

    Uses numpy's fixed-width-unicode view
    (``astype(f"U{total_width}").view("U1")``) to split every string into its
    characters at C speed — far faster than a per-character Python list build,
    which dominates the per-locus cost at the panel's ~1 kb flanking widths.
    Every string must be exactly ``total_width`` characters; a malformed string
    is rejected up front (the view would otherwise silently truncate or
    null-pad it). The output is byte-identical to ``np.array([list(s) for s in
    flanking_seq], dtype="U1")``.
    """
    seq = np.asarray(flanking_seq)
    if seq.shape[0] == 0:
        return np.empty((0, total_width), dtype="U1")
    lengths = np.char.str_len(seq.astype(str))
    bad = np.unique(lengths[lengths != total_width])
    if bad.size:
        raise ValueError(
            f"flanking_seq char-array shape mismatch: {int((lengths != total_width).sum())} "
            f"string(s) of length(s) {sorted(int(x) for x in bad)} != expected "
            f"{total_width}"
        )
    return seq.astype(f"U{total_width}").view("U1").reshape(seq.shape[0], total_width)


@dataclasses.dataclass(frozen=True)
class SNVCandidate:
    """One informative position passing the coverage + VAF filter.

    ``ref_pos`` is the canonical-orientation reference position; for
    ``-`` loci the flanking strings are revcomp'd so the string-index →
    ref-pos mapping reverses (see :func:`find_informative_snvs`).
    """

    locus_id: str
    ref_pos: int
    segment: str
    string_index: int
    allele_major: str
    allele_minor: str
    vaf_minor: float
    coverage: int


def aggregate_position_frequencies(
    flanking_seq: np.ndarray,
    flanking_id: np.ndarray,
    count: np.ndarray,
    flanking_up_width: int,
    flanking_dn_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-position base counts for the upstream and downstream segments.

    Walks each unique flanking string once, weighted by the total
    ``count`` across all rows sharing that ``flanking_id``.

    Parameters
    ----------
    flanking_seq : (n_flanking,) object
        Unique flanking strings, each of length
        ``flanking_up_width + 1 + flanking_dn_width`` (``"|"`` separator).
    flanking_id : (n_rows,) int
        Row → unique-string index. Must be non-negative.
    count : (n_rows,) int
        Per-row read count (post-pair-collapse).
    flanking_up_width, flanking_dn_width : int

    Returns
    -------
    up_freqs : (flanking_up_width, 6) int64
    dn_freqs : (flanking_dn_width, 6) int64
        Column order: A, C, G, T, N, '.'.
    """
    n_flanking = int(flanking_seq.shape[0])
    up_freqs = np.zeros((flanking_up_width, _N_COLS), dtype=np.int64)
    dn_freqs = np.zeros((flanking_dn_width, _N_COLS), dtype=np.int64)
    if n_flanking == 0 or (flanking_up_width == 0 and flanking_dn_width == 0):
        return up_freqs, dn_freqs

    assert (flanking_id >= 0).all(), (
        "flanking_id contains negative values — schema-v1 panel passed to "
        "aggregator (expected to short-circuit on zero widths upstream)"
    )

    fid_weight = np.zeros(n_flanking, dtype=np.int64)
    np.add.at(fid_weight, flanking_id, count)

    total_width = flanking_up_width + 1 + flanking_dn_width
    arr = flanking_char_matrix(flanking_seq, total_width)

    for col, ch in enumerate(_BASES):
        mask = arr == ch
        per_pos = (mask * fid_weight[:, None]).sum(axis=0)
        if flanking_up_width > 0:
            up_freqs[:, col] = per_pos[:flanking_up_width]
        if flanking_dn_width > 0:
            dn_freqs[:, col] = per_pos[flanking_up_width + 1:]

    return up_freqs, dn_freqs


def family_consensus_at_positions(
    flanking_seq: np.ndarray,
    flanking_id: np.ndarray,
    count: np.ndarray,
    mi: np.ndarray,
    positions: list[int],
    total_width: int,
) -> np.ndarray:
    """Per-family consensus base at each of a few flanking-string offsets.

    The same collapse rule as :func:`family_consensus_counts` — read-count
    weighted vote pooled across both strands, called iff the maximum weight is
    positive and unique, abstaining on a tie or an all-``.``/``N`` family — read
    out per family instead of counted per base. A caller that needs to see two
    positions *of the same molecule* (the phasing linkage gate, which asks which
    haplotype a molecule carries) cannot work from the per-position counts.

    Unblocked on purpose: the callers pass a handful of positions, so the
    ``(n_fam, n_pos, 4)`` accumulator is small where
    :func:`family_consensus_counts`'s whole-window one is not.

    Parameters
    ----------
    flanking_seq, flanking_id, count, mi
        As :func:`family_consensus_counts`.
    positions
        Offsets into the ``up + "|" + dn`` flanking string.
    total_width
        ``flanking_up_width + 1 + flanking_dn_width``.

    Returns
    -------
    (n_fam, len(positions)) U1
        Consensus base per family, ``''`` where the family abstains. Row order
        is the ``np.unique`` order of ``mi``; no caller depends on it.
    """
    pos = np.asarray(positions, dtype=np.int64)
    if int(np.asarray(flanking_id).shape[0]) == 0 or pos.size == 0:
        return np.zeros((0, pos.size), dtype="U1")

    arr = flanking_char_matrix(flanking_seq, total_width)[:, pos]
    fam_idx = np.unique(np.asarray(mi), return_inverse=True)[1].reshape(-1)
    n_fam = int(fam_idx.max()) + 1
    cf = sparse.csr_matrix(
        (np.asarray(count, dtype=np.float64), (fam_idx, np.asarray(flanking_id))),
        shape=(n_fam, arr.shape[0]),
    )
    # w_f(p, b) = Σ_{rows r ∈ f} count[r] · [char_r(p) == b]; exact on integer
    # counts via float64, as in family_consensus_counts.
    acc = np.empty((n_fam, pos.size, 4), dtype=np.float64)
    for b, base in enumerate(_ACGT):
        acc[..., b] = cf @ (arr == base).astype(np.float64)

    max_w = acc.max(axis=2)
    valid = (max_w > 0) & ((acc == max_w[..., None]).sum(axis=2) == 1)
    out = np.full(max_w.shape, "", dtype="U1")
    out[valid] = np.array(_ACGT, dtype="U1")[acc.argmax(axis=2)[valid]]
    return out


def family_consensus_counts(
    flanking_seq: np.ndarray,
    flanking_id: np.ndarray,
    count: np.ndarray,
    mi: np.ndarray,
    flanking_up_width: int,
    flanking_dn_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Strand-pooled UMI-family consensus ACGT counts per flanking position.

    Collapses a locus's deduped ``(mi, strand, seq, count)`` rows to molecular
    **families** — a family is one distinct ``mi`` (both duplex strands *and*
    every PCR duplicate of the source molecule pool into one, mirroring the
    on-target family count in ``prolyG.cn.family_counts.family_counts``) —
    then counts, per position, how many families call each base.

    Family consensus at a position ``p`` (the collapse rule): each family votes
    with its read-count-weighted base tally over all its rows (both strands),

        ``w_f(b) = Σ_{rows r ∈ f} count[r] · [char_r(p) == b]``,

    and the family's call is ``argmax_b w_f(b)`` **iff** the maximum weight is
    strictly positive *and* unique. A family that is all ``.``/``N`` at ``p``
    (no ACGT vote) abstains; an exact within-family tie (a two-way
    PCR/sequencing disagreement) also **abstains** — it contributes to no
    base's count. The per-position family count for base ``b`` is then the
    number of families whose consensus is ``b``.

    Strand-pooled by construction: a family is one molecule, not a strand, so
    (unlike the read-based strand-resolved ``nA_a…`` columns) there is no A/B
    split here.

    Parameters
    ----------
    flanking_seq : (n_flanking,) object
        Unique flanking strings, each ``up + "|" + dn`` of length
        ``flanking_up_width + 1 + flanking_dn_width`` (mirrors
        :func:`aggregate_position_frequencies`).
    flanking_id : (n_rows,) int
        Row → unique-string index.
    count : (n_rows,) int
        Per-row read count (post pair-collapse).
    mi : (n_rows,) object
        Per-row molecular identifier; a family = one distinct value.
    flanking_up_width, flanking_dn_width : int

    Returns
    -------
    up_fam : (flanking_up_width, 4) int64
    dn_fam : (flanking_dn_width, 4) int64
        Family consensus counts; column order A, C, G, T.
    """
    up_fam = np.zeros((flanking_up_width, 4), dtype=np.int64)
    dn_fam = np.zeros((flanking_dn_width, 4), dtype=np.int64)
    n_rows = int(np.asarray(flanking_id).shape[0])
    if n_rows == 0 or (flanking_up_width == 0 and flanking_dn_width == 0):
        return up_fam, dn_fam

    total_width = flanking_up_width + 1 + flanking_dn_width
    arr = flanking_char_matrix(flanking_seq, total_width)   # (n_flanking, total_width)

    # Map mi -> contiguous family index; aggregate each row's read count onto its
    # (family, unique-flanking-string) cell. This cf matrix is very sparse (one
    # nonzero per distinct (family, string) pair, ≤ n_rows), so the per-position
    # per-base consensus weight is a cheap sparse-times-dense product rather than
    # a dense (n_fam, total_width, 4) scatter over the ~1 kb window.
    fam_idx = np.unique(np.asarray(mi), return_inverse=True)[1].reshape(-1)
    n_fam = int(fam_idx.max()) + 1
    count = np.asarray(count, dtype=np.float64)
    cf = sparse.csr_matrix(
        (count, (fam_idx, np.asarray(flanking_id))),
        shape=(n_fam, arr.shape[0]),
    )

    fam_counts = np.zeros((total_width, 4), dtype=np.int64)
    # acc[f, p, b] = Σ_{rows r ∈ f} count[r] · [char_r(p) == b]  (exact on integer
    # counts via float64). The family axis cannot be reduced before the argmax — the
    # consensus is per family — so the window is walked in position blocks instead:
    # a deep locus (thousands of families over a ~1 kb window) would otherwise
    # materialize acc and the four dense (n_flanking, total_width) comparands whole,
    # which is hundreds of MB against a per-locus working set the pipeline sizes at
    # ~50 MB. The block width only bounds memory; every reduction below is over the
    # full family axis, so the result is identical to scoring the window at once.
    block = max(1, CONSENSUS_POSITION_BLOCK // max(n_fam, 1))
    for p0 in range(0, total_width, block):
        p1 = min(p0 + block, total_width)
        acc = np.empty((n_fam, p1 - p0, 4), dtype=np.float64)
        for b, base in enumerate(_ACGT):
            acc[..., b] = cf @ (arr[:, p0:p1] == base).astype(np.float64)

        # Per-family consensus: argmax base, valid iff the max weight is > 0 and
        # unique (a tie or an all-`.`/`N` family abstains).
        max_w = acc.max(axis=2)                         # (n_fam, p1 - p0)
        consensus = acc.argmax(axis=2)                  # first max on a tie
        n_at_max = (acc == max_w[..., None]).sum(axis=2)
        valid = (max_w > 0) & (n_at_max == 1)
        for b in range(4):
            fam_counts[p0:p1, b] = (valid & (consensus == b)).sum(axis=0)

    if flanking_up_width > 0:
        up_fam = fam_counts[:flanking_up_width]
    if flanking_dn_width > 0:
        dn_fam = fam_counts[flanking_up_width + 1:]
    return up_fam, dn_fam


def find_informative_snvs(
    panel: ExtractedPanel,
    *,
    min_coverage: int = 25,
    vaf_min: float = 0.2,
) -> dict[str, list[SNVCandidate]]:
    """Per-locus informative SNV candidates on flanking sequences.

    A position is included iff:

    - ACGT-only read-**family** depth ≥ ``min_coverage``, AND
    - minor-base VAF (minor families / ACGT family depth) ≥ ``vaf_min``.

    Both are counted over read families, one vote per molecule
    (:func:`family_consensus_counts`), matching the somatic and BAF channels
    on the prolyG side, which reuse the same collapse rule. A molecule
    sequenced $n$ times is one observation of the base it carries.

    Major/minor alleles are the top-2 by count over ``{A, C, G, T}``.
    ``N`` is invisible to depth and to allele identity. Tri-allelic
    positions silently drop the third base.

    ``min_coverage = 25`` replaces the read-unit 50. Unlike the length-channel
    screens, this one could not be re-derived to a constant that reproduces the
    old admitted set per sample: the family floor matching the read-50 screen
    runs from 16 on the most duplicated panel extraction to 44 on the least
    (median 24, IQR 24-27), and only 1 of the 13 has an exact integer match at
    all. That spread is the defect, not an obstacle to fixing it — at a fixed 50
    reads the screen demands ~25 families of a 2.0x library and ~6 of a 7.9x
    one.

    25 sits just above the cohort median of that per-sample floor and inside its
    IQR. It puts median admitted positions at 783 against the read screen's 788;
    24, the median itself, gives 793. The two bracket the read screen's median
    at equal distance, so this is a choice between the two sides of it rather
    than a measured optimum — note that the sibling ``min_rfs`` migration took
    the admitting side on the argument that a screen's false negatives are
    invisible, and 25 is the stricter side of that pair.

    What the change removes is the duplication dependence, not the spread.
    Admitted positions correlate with reads per family at r = +0.494 under the
    read screen and r = +0.067 under this one; the residual spread across
    samples is real variation in how many het SNVs they carry. The two shallow
    libraries gain ~9% of their positions and the deepest loses ~10%.

    Ref-position mapping by orientation:

    =============  ===========  ================================
    orientation    segment      ref_pos
    =============  ===========  ================================
    ``+``          upstream     ``flanking_up_ref_pos_start + i``
    ``+``          downstream   ``flanking_dn_ref_pos_start + i``
    ``-``          upstream     ``flanking_up_ref_pos_start - i``
    ``-``          downstream   ``flanking_dn_ref_pos_start - i``
    =============  ===========  ================================

    Parameters
    ----------
    panel : ExtractedPanel
        Schema-v2 panel with per-locus flanking strings.
    min_coverage : int, default 25
        Minimum ACGT **family** depth. Stringency-matched to the read-unit 50 it
        replaces, not independently calibrated — there is no SNV ground truth on
        this cohort. Do not read it as an optimum.
    vaf_min : float, default 0.2
        Unchanged: it was already a share, so only the unit under it moved.

    Returns
    -------
    dict[str, list[SNVCandidate]]
        Every locus in ``panel.loci`` gets a key; the list is empty
        when no position passes the filter. ``len()`` over the dict
        gives the panel size; sum of ``len(v) > 0`` gives the
        informative-locus count.
    """
    out: dict[str, list[SNVCandidate]] = {}
    for locus_id, locus in panel.loci.items():
        up_w = locus.flanking_up_width
        dn_w = locus.flanking_dn_width
        if up_w == 0 and dn_w == 0:
            out[locus_id] = []
            continue
        up_freqs, dn_freqs = family_consensus_counts(
            locus.flanking_seq, locus.flanking_id, locus.count, locus.mi,
            up_w, dn_w,
        )
        out[locus_id] = _collect_candidates(
            locus_id=locus_id,
            up_freqs=up_freqs,
            dn_freqs=dn_freqs,
            up_ref_start=locus.flanking_up_ref_pos_start,
            dn_ref_start=locus.flanking_dn_ref_pos_start,
            ref_orientation=locus.ref_orientation,
            min_coverage=min_coverage,
            vaf_min=vaf_min,
        )
    return out


def ref_pos_at_offset(ref_orientation: str, ref_start: int, offset: int) -> int:
    """Genomic position of ``offset`` into a flanking segment starting at ``ref_start``.

    ``ref_orientation`` is ``"+"`` or ``"-"`` (see :attr:`ExtractedLocus.ref_orientation`).
    Inverse of :func:`offset_at_ref_pos`.
    """
    sign = 1 if ref_orientation == "+" else -1
    return ref_start + sign * offset


def offset_at_ref_pos(
    ref_orientation: str, ref_start: int, ref_pos: int, width: int,
) -> int | None:
    """Offset into a ``width``-long flanking segment at ``ref_start`` for genomic ``ref_pos``.

    Inverse of :func:`ref_pos_at_offset`. Returns ``None`` when ``ref_pos`` does
    not land in ``[0, width)`` of this segment (used by
    :func:`~prolyg_phasing.ref_phasing._remap_snv_to_target_position` to check
    the ``up`` then ``dn`` segment of a *different* locus's flanking geometry).
    """
    sign = 1 if ref_orientation == "+" else -1
    offset = sign * (ref_pos - ref_start)
    return offset if 0 <= offset < width else None


def _collect_candidates(
    *,
    locus_id: str,
    up_freqs: np.ndarray,
    dn_freqs: np.ndarray,
    up_ref_start: int,
    dn_ref_start: int,
    ref_orientation: str,
    min_coverage: int,
    vaf_min: float,
) -> list[SNVCandidate]:
    out: list[SNVCandidate] = []
    for segment, freqs, ref_start in (
        ("up", up_freqs, up_ref_start),
        ("dn", dn_freqs, dn_ref_start),
    ):
        for i in range(freqs.shape[0]):
            acgt = freqs[i, :4]
            coverage = int(acgt.sum())
            if coverage < min_coverage:
                continue
            order = np.argsort(acgt, kind="stable")[::-1]
            minor_count = int(acgt[order[1]])
            if minor_count == 0:
                continue
            vaf_minor = minor_count / coverage
            if vaf_minor < vaf_min:
                continue
            out.append(SNVCandidate(
                locus_id=locus_id,
                ref_pos=ref_pos_at_offset(ref_orientation, ref_start, i),
                segment=segment,
                string_index=i,
                allele_major=_BASES[int(order[0])],
                allele_minor=_BASES[int(order[1])],
                vaf_minor=vaf_minor,
                coverage=coverage,
            ))
    return out


# ---------------------------------------------------------------------------
# Haplotype phasing
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FlankingHaplotype:
    """Per-locus phasing of informative flanking SNVs into two profiles.

    The two profiles are anonymous locus-level partition labels —
    ``hap_major`` is the more-populous biological haplotype at this
    locus (anchored to the anchor SNV's per-position major allele) and
    ``hap_minor`` is its complement.

    Note on naming: ``snvs[k].allele_major`` is the *per-position*
    more-frequent base at SNV ``k``. ``hap_major_profile[k]``
    is the base on the *locus-level* major haplotype at SNV ``k``.
    These agree for the anchor and for any non-anchor SNV that is
    in-phase with the anchor; they diverge (``hap_major_profile[k] ==
    snvs[k].allele_minor``) when SNV ``k`` is out-of-phase and was
    flipped during linkage resolution.

    Attributes
    ----------
    locus_id
        Locus identifier; mirrors :class:`SNVCandidate.locus_id`.
    snvs
        Phased SNVs, in profile order. Excludes any SNVs dropped for
        insufficient double-coverage against the anchor (see
        ``dropped_low_linkage_snvs``).
    hap_major_profile, hap_minor_profile
        Length-``len(snvs)`` tuples of base calls; one per SNV, in the
        same order as ``snvs``.
    anchor_index
        Index into ``snvs`` of the anchor SNV (highest per-position
        coverage; ties broken by list order). ``-1`` for zero-SNV loci.
    pair_discordance
        Length-``len(snvs)`` tuple. ``pair_discordance[k]`` is the
        fraction of (anchor, ``snvs[k]``) double-covered molecules whose
        consensus bases disagree with the chosen phase. ``0.0`` at the
        anchor index (trivially in-phase with itself).
    pair_double_coverage
        Length-``len(snvs)`` tuple. Molecules covering both the anchor and
        ``snvs[k]`` with a consensus base ∈ ``{major, minor}`` at both
        positions. At the anchor index, equals the anchor's own
        per-position ACGT family depth.
    max_discordance
        ``max(pair_discordance[k] for k != anchor_index)``. ``0.0`` for
        zero-SNV or single-SNV loci. Bounded in ``[0, 0.5]``.
    n_anchor_families
        Anchor SNV's per-position ACGT family depth
        (= ``snvs[anchor].coverage``). ``0`` for zero-SNV loci.
    dropped_low_linkage_snvs
        SNVs excluded from ``snvs`` because their double-coverage with
        the anchor fell below ``min_linkage_families``.
    """

    locus_id: str
    snvs: tuple[SNVCandidate, ...]
    hap_major_profile: tuple[str, ...]
    hap_minor_profile: tuple[str, ...]
    anchor_index: int
    pair_discordance: tuple[float, ...]
    pair_double_coverage: tuple[int, ...]
    max_discordance: float
    n_anchor_families: int
    dropped_low_linkage_snvs: tuple[SNVCandidate, ...]


def determine_flanking_haplotypes(
    panel: ExtractedPanel,
    snvs_by_locus: dict[str, list[SNVCandidate]],
    *,
    min_linkage_families: int = 10,
) -> dict[str, FlankingHaplotype]:
    """Phase per-locus informative SNVs into two flanking-haplotype profiles.

    Algorithm (per locus):

    1. Pick the anchor SNV: highest ``coverage`` (ties broken by input
       list order).
    2. For each non-anchor SNV ``k``, count **molecules** whose family
       consensus is ∈ ``{major, minor}`` at both the anchor and SNV ``k``.
       If ``n_AA + n_BB ≥ n_AB + n_BA``, SNV ``k`` is in-phase
       (``hap_major_profile[k] = k.allele_major``); else flipped
       (``hap_major_profile[k] = k.allele_minor``).
    3. If the double-coverage total falls below ``min_linkage_families``,
       drop SNV ``k`` (record in ``dropped_low_linkage_snvs``); the
       locus still emits with the remaining SNVs.

    Step 2 asks which haplotype a molecule carries, so it reads both
    positions off the same molecule (:func:`family_consensus_at_positions`,
    the collapse rule the discovery screen above and the prolyG-side somatic
    and BAF channels all use). A molecule that abstains at either position —
    no ACGT vote, or an internal tie — supports neither phase and is counted
    in neither.

    Every locus in ``panel.loci`` gets a key in the output; loci with
    no informative SNVs emit a sentinel :class:`FlankingHaplotype` with
    empty profiles and ``anchor_index == -1``.

    Parameters
    ----------
    panel
        Schema-v2 extracted panel.
    snvs_by_locus
        Output of :func:`find_informative_snvs` (every-locus dict).
        Loci absent from this dict are treated as zero-SNV.
    min_linkage_families
        Minimum molecules calling both the anchor and ``snvs[k]`` for the
        flip decision to be made. SNVs below this threshold are dropped
        from the profile.

        ``10`` replaces the read-unit ``50``, and is **not** a stringency
        match to it. The 50 was a round number rather than a calibrated one, so
        reproducing the set it admitted would inherit an arbitrary target; the
        family floor that does reproduce it runs 12 to 39 across the 13 panels
        (median 19), tracking each library's molecule count.

        10 is where the evidence bound sits. The flip decision is a majority
        over molecules, and below ~10 no split separates linkage from
        independence at conventional levels: an unconditional sign test reaches
        p ≤ 0.01 first at 8 molecules, and conditioning on the observed allele
        margins — which is what a germline het pair actually has — puts it near
        10.

        What the change buys is yield, and that is measured against replicate
        libraries rather than against a rule of ours. SD13 and SD14 each
        contribute two separately prepped libraries of one germline differing
        2.8x in duplication, so the phase relation between two positions must
        agree across them. At floor 10 the two libraries agree on 96.0 % (SD13)
        and 94.7 % (SD14) of the relations both admit, against 95.5 % and 93.8 %
        under the read-unit 50 — unchanged within noise — while admitting about
        25 % more partner SNVs (223 vs 178 comparable relations on SD13). No
        candidate rule scored differently on that check, including a per-pair
        Fisher test, which is why this stayed a count.

        The residual 4-6 % cross-library disagreement is not touched by any
        floor in this range and is a separate defect.

    Returns
    -------
    dict[str, FlankingHaplotype]
        Per-locus phasing, with every locus in ``panel.loci`` keyed.
    """
    out: dict[str, FlankingHaplotype] = {}
    for locus_id, locus in panel.loci.items():
        snvs = list(snvs_by_locus.get(locus_id, []))
        out[locus_id] = _phase_locus(locus_id, locus, snvs, min_linkage_families)
    return out


def _phase_locus(
    locus_id: str,
    locus: ExtractedLocus,
    snvs: list[SNVCandidate],
    min_linkage_families: int,
) -> FlankingHaplotype:
    if not snvs:
        return FlankingHaplotype(
            locus_id=locus_id, snvs=(), hap_major_profile=(),
            hap_minor_profile=(), anchor_index=-1,
            pair_discordance=(), pair_double_coverage=(),
            max_discordance=0.0, n_anchor_families=0,
            dropped_low_linkage_snvs=(),
        )

    # Anchor: highest coverage; ties broken by list order (argmax returns first).
    anchor_idx_in = int(np.argmax([s.coverage for s in snvs]))
    anchor = snvs[anchor_idx_in]

    if len(snvs) == 1:
        return FlankingHaplotype(
            locus_id=locus_id,
            snvs=(anchor,),
            hap_major_profile=(anchor.allele_major,),
            hap_minor_profile=(anchor.allele_minor,),
            anchor_index=0,
            pair_discordance=(0.0,),
            pair_double_coverage=(anchor.coverage,),
            max_discordance=0.0,
            n_anchor_families=anchor.coverage,
            dropped_low_linkage_snvs=(),
        )

    bases = _family_bases_at_snvs(locus, snvs)

    kept_snvs: list[SNVCandidate] = []
    kept_indices_in: list[int] = []
    hap_major_l: list[str] = []
    hap_minor_l: list[str] = []
    discordance_l: list[float] = []
    double_cov_l: list[int] = []
    dropped: list[SNVCandidate] = []

    for k, snv_k in enumerate(snvs):
        if k == anchor_idx_in:
            kept_snvs.append(snv_k)
            kept_indices_in.append(k)
            hap_major_l.append(snv_k.allele_major)
            hap_minor_l.append(snv_k.allele_minor)
            discordance_l.append(0.0)
            double_cov_l.append(snv_k.coverage)
            continue

        n_MM, n_mm, n_Mm, n_mM = _pair_counts(
            bases, anchor_idx_in, k, anchor, snv_k,
        )
        in_phase = n_MM + n_mm
        out_phase = n_Mm + n_mM
        total = in_phase + out_phase
        if total < min_linkage_families:
            dropped.append(snv_k)
            continue
        if in_phase >= out_phase:
            hap_major_l.append(snv_k.allele_major)
            hap_minor_l.append(snv_k.allele_minor)
            discordance = out_phase / total
        else:
            hap_major_l.append(snv_k.allele_minor)
            hap_minor_l.append(snv_k.allele_major)
            discordance = in_phase / total
        kept_snvs.append(snv_k)
        kept_indices_in.append(k)
        discordance_l.append(discordance)
        double_cov_l.append(total)

    new_anchor_index = kept_indices_in.index(anchor_idx_in)
    non_anchor_disc = [
        d for i, d in enumerate(discordance_l) if i != new_anchor_index
    ]
    max_disc = max(non_anchor_disc) if non_anchor_disc else 0.0

    return FlankingHaplotype(
        locus_id=locus_id,
        snvs=tuple(kept_snvs),
        hap_major_profile=tuple(hap_major_l),
        hap_minor_profile=tuple(hap_minor_l),
        anchor_index=new_anchor_index,
        pair_discordance=tuple(discordance_l),
        pair_double_coverage=tuple(double_cov_l),
        max_discordance=max_disc,
        n_anchor_families=anchor.coverage,
        dropped_low_linkage_snvs=tuple(dropped),
    )


def per_row_bases_at_positions(
    locus: ExtractedLocus, positions: list[int],
) -> np.ndarray:
    """Return (n_rows, len(positions)) U1 array of bases at flanking-string offsets.

    ``positions`` are 0-based offsets into the locus's ``up + "|" + dn``
    flanking string. Uncovered positions in a row read as ``'.'`` (the
    extraction pad). Shared by the normal-side
    :func:`per_row_bases_at_snvs` (offsets from each SNV's
    ``string_index``) and the anchored path (offsets remapped from the
    reference panel's SNV ``ref_pos`` via
    :func:`~prolyg_phasing.ref_phasing._remap_snv_to_target_position`).
    """
    assert (locus.flanking_id >= 0).all(), (
        "flanking_id contains negative values; phasing requires a schema-v2 panel"
    )
    n_pos = len(positions)
    n_flanking = len(locus.flanking_seq)
    fid_bases = np.full((n_flanking, n_pos), ".", dtype="U1")
    for f in range(n_flanking):
        s = str(locus.flanking_seq[f])
        for k, p in enumerate(positions):
            fid_bases[f, k] = s[p]
    return fid_bases[locus.flanking_id]


def per_row_bases_at_snvs(
    locus: ExtractedLocus, snvs: list[SNVCandidate],
) -> np.ndarray:
    """Return (n_rows, n_snvs) U1 array of called bases at each SNV position.

    Maps each SNV to its flanking-string offset from ``string_index``
    (downstream SNVs shifted past the ``up`` segment and ``"|"``
    separator), then delegates to :func:`per_row_bases_at_positions`.
    """
    positions: list[int] = []
    for snv in snvs:
        if snv.segment == "up":
            positions.append(snv.string_index)
        else:
            positions.append(locus.flanking_up_width + 1 + snv.string_index)
    return per_row_bases_at_positions(locus, positions)


def _family_bases_at_snvs(
    locus: ExtractedLocus, snvs: list[SNVCandidate],
) -> np.ndarray:
    """``(n_families, n_snvs)`` consensus base per molecule at each SNV.

    The family counterpart of :func:`per_row_bases_at_snvs`: one row per
    molecule rather than per deduplicated read record, so a pair of positions
    can be read off the same molecule.
    """
    positions = [
        snv.string_index if snv.segment == "up"
        else locus.flanking_up_width + 1 + snv.string_index
        for snv in snvs
    ]
    return family_consensus_at_positions(
        locus.flanking_seq, locus.flanking_id, locus.count, locus.mi, positions,
        locus.flanking_up_width + 1 + locus.flanking_dn_width,
    )


def _pair_counts(
    bases: np.ndarray,
    anchor_idx: int,
    k: int,
    anchor: SNVCandidate,
    snv_k: SNVCandidate,
) -> tuple[int, int, int, int]:
    """Molecules carrying each of the four (anchor, SNV ``k``) base pairs.

    ``bases`` is one row per molecule, so each pair count is a molecule count.
    A molecule that abstains at either position falls into none of the four.
    """
    aM, am = anchor.allele_major, anchor.allele_minor
    kM, km = snv_k.allele_major, snv_k.allele_minor
    anchor_calls = bases[:, anchor_idx]
    k_calls = bases[:, k]
    n_MM = int(((anchor_calls == aM) & (k_calls == kM)).sum())
    n_mm = int(((anchor_calls == am) & (k_calls == km)).sum())
    n_Mm = int(((anchor_calls == aM) & (k_calls == km)).sum())
    n_mM = int(((anchor_calls == am) & (k_calls == kM)).sum())
    return n_MM, n_mm, n_Mm, n_mM


# ---------------------------------------------------------------------------
# Per-row flanking-haplotype assignment
# ---------------------------------------------------------------------------


LABEL_MAJOR = 0
LABEL_MINOR = 1
LABEL_UNK = -1


@dataclasses.dataclass(frozen=True)
class FlankingHaplotypeAssignment:
    """Per-row haplotype labels for one locus.

    Labels classify each deduplicated `(mi, strand, seq, flanking_id)`
    record (henceforth "row") against the locus's two flanking-haplotype
    profiles (:class:`FlankingHaplotype`). A row's `count` is unchanged
    here; downstream consumers weight by `count` when aggregating.

    Attributes
    ----------
    locus_id
        Locus identifier; mirrors :class:`FlankingHaplotype.locus_id`.
    n_rows
        Number of rows at this locus (= ``len(locus.mi)``).
    label
        Length-``n_rows`` int8 array. Values:
        ``0`` (``LABEL_MAJOR``), ``1`` (``LABEL_MINOR``),
        ``-1`` (``LABEL_UNK``).
    n_major
        Length-``n_rows`` int8 array. Positions where the row's called
        base equals ``hap_major_profile[k]``.
    n_minor
        Length-``n_rows`` int8 array. Positions where the row's called
        base equals ``hap_minor_profile[k]``.
    n_other
        Length-``n_rows`` int8 array. Positions where the row's called
        base is ACGT but matches neither profile (rare third allele or
        consistent family-level miscall). Uncovered positions (``'.'``)
        are excluded.
    """

    locus_id: str
    n_rows: int
    label: np.ndarray
    n_major: np.ndarray
    n_minor: np.ndarray
    n_other: np.ndarray


def assign_flanking_haplotypes(
    panel: ExtractedPanel,
    phased: dict[str, FlankingHaplotype],
    *,
    min_informative_positions: int = 1,
    vote_margin: int = 0,
) -> dict[str, FlankingHaplotypeAssignment]:
    """Per-row haplotype labels for every locus in ``panel.loci``.

    Decision rule (per row), counting only SNVs kept in
    :attr:`FlankingHaplotype.snvs` (low-linkage SNVs in
    :attr:`FlankingHaplotype.dropped_low_linkage_snvs` are ignored):

    - ``n_major + n_minor < min_informative_positions`` → ``LABEL_UNK``
    - ``n_major - n_minor > vote_margin`` → ``LABEL_MAJOR``
    - ``n_minor - n_major > vote_margin`` → ``LABEL_MINOR``
    - else (tie / near-tie) → ``LABEL_UNK``

    Positions where the row's base is neither in ``{hap_major_profile[k],
    hap_minor_profile[k]}`` nor uncovered (``'.'``) are counted in
    ``n_other`` and ignored for voting.

    Loci absent from ``phased`` (or with empty ``snvs``) emit an
    all-``LABEL_UNK`` assignment of the correct length.

    Parameters
    ----------
    panel
        Schema-v2 extracted panel.
    phased
        Output of :func:`determine_flanking_haplotypes`.
    min_informative_positions
        Minimum ``n_major + n_minor`` to make a non-``unk`` call.
    vote_margin
        Minimum ``|n_major - n_minor|`` to make a non-``unk`` call.

    Returns
    -------
    dict[str, FlankingHaplotypeAssignment]
        Per-locus assignment, with every locus in ``panel.loci`` keyed.
    """
    out: dict[str, FlankingHaplotypeAssignment] = {}
    for locus_id, locus in panel.loci.items():
        n_rows = int(locus.mi.shape[0])
        hap = phased.get(locus_id)
        if hap is None or len(hap.snvs) == 0:
            out[locus_id] = all_unk_assignment(locus_id, n_rows)
            continue
        out[locus_id] = _assign_locus(
            locus_id=locus_id,
            locus=locus,
            hap=hap,
            min_informative_positions=min_informative_positions,
            vote_margin=vote_margin,
        )
    return out


def all_unk_assignment(locus_id: str, n_rows: int) -> FlankingHaplotypeAssignment:
    zeros = np.zeros(n_rows, dtype=np.int8)
    return FlankingHaplotypeAssignment(
        locus_id=locus_id,
        n_rows=n_rows,
        label=np.full(n_rows, LABEL_UNK, dtype=np.int8),
        n_major=zeros.copy(),
        n_minor=zeros.copy(),
        n_other=zeros.copy(),
    )


def _assign_locus(
    *,
    locus_id: str,
    locus: ExtractedLocus,
    hap: FlankingHaplotype,
    min_informative_positions: int,
    vote_margin: int,
) -> FlankingHaplotypeAssignment:
    bases = per_row_bases_at_snvs(locus, list(hap.snvs))  # (n_rows, n_snvs) U1
    return vote_assignment(
        locus_id,
        bases,
        np.asarray(hap.hap_major_profile, dtype="U1"),
        np.asarray(hap.hap_minor_profile, dtype="U1"),
        min_informative_positions,
        vote_margin,
    )


def vote_assignment(
    locus_id: str,
    bases: np.ndarray,
    major_profile: np.ndarray,
    minor_profile: np.ndarray,
    min_informative_positions: int,
    vote_margin: int,
) -> FlankingHaplotypeAssignment:
    """Per-row major/minor/unk vote of ``bases`` against two haplotype profiles.

    Shared by the normal-side :func:`_assign_locus` and the anchored
    :func:`~prolyg_phasing.ref_phasing.assign_flanking_haplotypes_ref`,
    so both use byte-identical voting. ``bases`` is ``(n_rows,
    n_profile_snvs)`` U1; ``major_profile``
    / ``minor_profile`` are length-``n_profile_snvs`` U1 base arrays aligned
    with the columns of ``bases``.
    """
    n_rows = bases.shape[0]
    is_major = bases == major_profile[None, :]
    is_minor = bases == minor_profile[None, :]
    is_uncov = bases == "."
    is_other = ~(is_major | is_minor | is_uncov)

    n_major = is_major.sum(axis=1).astype(np.int8)
    n_minor = is_minor.sum(axis=1).astype(np.int8)
    n_other = is_other.sum(axis=1).astype(np.int8)

    label = np.full(n_rows, LABEL_UNK, dtype=np.int8)
    informative = (n_major.astype(np.int32) + n_minor.astype(np.int32)) >= min_informative_positions
    diff = n_major.astype(np.int32) - n_minor.astype(np.int32)
    label[informative & (diff > vote_margin)] = LABEL_MAJOR
    label[informative & (-diff > vote_margin)] = LABEL_MINOR

    return FlankingHaplotypeAssignment(
        locus_id=locus_id,
        n_rows=n_rows,
        label=label,
        n_major=n_major,
        n_minor=n_minor,
        n_other=n_other,
    )


# ---------------------------------------------------------------------------
# Panel-level container
# ---------------------------------------------------------------------------


class PicklePanel:
    """Mixin: single-``.pkl`` save/load for a panel-level container dataclass.

    Shared by :class:`PhasingPanel` and
    :class:`~prolyg_phasing.ref_phasing.RefPhasingPanel` — both persist as one
    pickle of the whole in-memory dataclass, nothing schema-specific.
    """

    def save_pickle(self, path: str | Path) -> None:
        """Write a single ``.pkl`` of the in-memory dataclass."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_pickle(cls, path: str | Path) -> Self:
        with Path(path).open("rb") as f:
            return pickle.load(f)


@dataclasses.dataclass
class PhasingPanel(PicklePanel):
    """Full phasing artifact for one sample: SNVs + haplotypes + assignments.

    Carries the outputs of PRs 2-4 in a single typed container that
    rides alongside `ExtractedPanel` on disk (sidecar pickle). Built
    from a schema-v2 :class:`ExtractedPanel` via :meth:`from_panel`.
    """

    snvs: dict[str, list[SNVCandidate]]
    haplotypes: dict[str, FlankingHaplotype]
    assignments: dict[str, FlankingHaplotypeAssignment]

    @classmethod
    def from_panel(
        cls,
        panel: ExtractedPanel,
        *,
        min_coverage: int = 25,
        vaf_min: float = 0.2,
        min_linkage_families: int = 10,
        min_informative_positions: int = 1,
        vote_margin: int = 0,
    ) -> PhasingPanel:
        """Run SNV discovery, haplotype phasing, and per-read assignment on a panel."""
        snvs = find_informative_snvs(
            panel, min_coverage=min_coverage, vaf_min=vaf_min,
        )
        haplotypes = determine_flanking_haplotypes(
            panel, snvs, min_linkage_families=min_linkage_families,
        )
        assignments = assign_flanking_haplotypes(
            panel, haplotypes,
            min_informative_positions=min_informative_positions,
            vote_margin=vote_margin,
        )
        return cls(snvs=snvs, haplotypes=haplotypes, assignments=assignments)
