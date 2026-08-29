"""Anchored flanking-haplotype phasing: vote one panel against another's profile.

Re-votes one panel's reads against a *different*, already-phased panel's
haplotype profiles (:class:`~prolyg_phasing.phasing.FlankingHaplotype`): no
SNV discovery or haplotype-profile construction happens here, only
re-locating the reference panel's germline SNVs inside the target panel's
own flanking-string coordinates (by genomic position) and voting target
reads against the surviving profile with the same rule
:func:`~prolyg_phasing.phasing.assign_flanking_haplotypes` uses for
same-panel voting. Which two panels play "target" and "reference" — e.g. a
tumor bulk sample voted against its matched normal — is a decision the
caller makes; this module carries no such vocabulary.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from prolyg_phasing.io.panel import ExtractedLocus, ExtractedPanel
from prolyg_phasing.phasing import (
    FlankingHaplotype,
    FlankingHaplotypeAssignment,
    PicklePanel,
    SNVCandidate,
    all_unk_assignment,
    offset_at_ref_pos,
    per_row_bases_at_positions,
    vote_assignment,
)


def _remap_snv_to_target_position(
    snv: SNVCandidate, target_locus: ExtractedLocus,
) -> int | None:
    """Flanking-string offset of a reference-panel SNV in the target locus.

    Inverts :func:`~prolyg_phasing.phasing.ref_pos_at_offset` (via
    :func:`~prolyg_phasing.phasing.offset_at_ref_pos`) against the target
    locus's own flanking geometry, checking the ``up`` then ``dn`` segment.
    Genomic ``ref_pos`` is the cross-panel key: target and reference differ
    in ``g_walk`` / width, so a SNV's reference ``string_index`` is not its
    target offset. Returns the 0-based offset into the target
    ``up + "|" + dn`` string, or ``None`` if the target locus does not cover
    the SNV's genomic position (shorter window, or the base now sits inside
    the target's tract).
    """
    i_up = offset_at_ref_pos(
        target_locus.ref_orientation, target_locus.flanking_up_ref_pos_start,
        snv.ref_pos, target_locus.flanking_up_width,
    )
    if i_up is not None:
        return i_up
    i_dn = offset_at_ref_pos(
        target_locus.ref_orientation, target_locus.flanking_dn_ref_pos_start,
        snv.ref_pos, target_locus.flanking_dn_width,
    )
    if i_dn is not None:
        return target_locus.flanking_up_width + 1 + i_dn
    return None


@dataclasses.dataclass
class RefPhasingPanel(PicklePanel):
    """Per-read haplotype labels for one panel, anchored on another's profile.

    Mirrors :class:`~prolyg_phasing.phasing.PhasingPanel`'s shape but for
    the anchored case: the haplotype *profiles* come from the reference
    panel's own phasing output
    (:class:`~prolyg_phasing.phasing.FlankingHaplotype`), and only per-read
    voting runs on the target — no target-side SNV discovery or profile
    construction. Reuses
    :class:`~prolyg_phasing.phasing.FlankingHaplotypeAssignment` and
    :class:`~prolyg_phasing.phasing.SNVCandidate` unchanged, so existing
    ``phasing.pkl`` files stay loadable.

    Attributes
    ----------
    target_sample_id, reference_sample_id : str
    assignments : dict[str, FlankingHaplotypeAssignment]
        Per-target-locus per-read labels (``label`` ∈ {``LABEL_MAJOR``,
        ``LABEL_MINOR``, ``LABEL_UNK``}), keyed by ``locus_id``. Every
        target locus is keyed.
    usable_snvs : dict[str, tuple[SNVCandidate, ...]]
        The reference panel's SNVs the target locus actually covers
        (survived the ``ref_pos`` remap), in profile order. Per-locus
        phasing power is ``len(usable_snvs[locus_id])``; an empty tuple
        means the locus produced all-``unk`` labels.
    """

    target_sample_id: str
    reference_sample_id: str
    assignments: dict[str, FlankingHaplotypeAssignment]
    usable_snvs: dict[str, tuple[SNVCandidate, ...]]


def assign_flanking_haplotypes_ref(
    target_panel: ExtractedPanel,
    reference_phased: dict[str, FlankingHaplotype],
    *,
    target_sample_id: str,
    reference_sample_id: str,
    min_informative_positions: int = 1,
    vote_margin: int = 0,
) -> RefPhasingPanel:
    """Phase each target read against the reference panel's haplotypes.

    The germline SNV set and the ``hap_major_profile`` / ``hap_minor_profile``
    come from the reference panel's own
    :func:`~prolyg_phasing.phasing.determine_flanking_haplotypes` output. This
    function re-locates each reference SNV in the target flanking string by
    genomic ``ref_pos`` (:func:`_remap_snv_to_target_position`), drops SNVs
    the target locus does not cover, and votes each target read against the
    surviving profile using the same rule as
    :func:`~prolyg_phasing.phasing.assign_flanking_haplotypes`.

    A read is labeled ``hap_major`` / ``hap_minor`` when it covers
    ≥ ``min_informative_positions`` usable reference SNVs with a margin
    > ``vote_margin``, else ``LABEL_UNK`` ("could not phase"). Loci absent
    from ``reference_phased``, with no germline SNVs, or with none covered
    in the target, emit all-``unk`` labels and an empty ``usable_snvs``
    entry.

    Parameters
    ----------
    target_panel : ExtractedPanel
        Schema-v2 target-sample panel.
    reference_phased : dict[str, FlankingHaplotype]
        Reference panel's phasing output (``PhasingPanel.haplotypes``).
    target_sample_id, reference_sample_id : str
    min_informative_positions : int, default 1
    vote_margin : int, default 0

    Returns
    -------
    RefPhasingPanel
    """
    assignments: dict[str, FlankingHaplotypeAssignment] = {}
    usable: dict[str, tuple[SNVCandidate, ...]] = {}

    for locus_id, target_locus in target_panel.loci.items():
        n_rows = int(target_locus.mi.shape[0])
        hap = reference_phased.get(locus_id)
        if hap is None or len(hap.snvs) == 0:
            assignments[locus_id] = all_unk_assignment(locus_id, n_rows)
            usable[locus_id] = ()
            continue

        positions: list[int] = []
        kept_snvs: list[SNVCandidate] = []
        kept_major: list[str] = []
        kept_minor: list[str] = []
        for k, snv in enumerate(hap.snvs):
            p = _remap_snv_to_target_position(snv, target_locus)
            if p is None:
                continue
            positions.append(p)
            kept_snvs.append(snv)
            kept_major.append(hap.hap_major_profile[k])
            kept_minor.append(hap.hap_minor_profile[k])

        if not positions:
            assignments[locus_id] = all_unk_assignment(locus_id, n_rows)
            usable[locus_id] = ()
            continue

        bases = per_row_bases_at_positions(target_locus, positions)
        assignments[locus_id] = vote_assignment(
            locus_id,
            bases,
            np.asarray(kept_major, dtype="U1"),
            np.asarray(kept_minor, dtype="U1"),
            min_informative_positions,
            vote_margin,
        )
        usable[locus_id] = tuple(kept_snvs)

    return RefPhasingPanel(
        target_sample_id=target_sample_id,
        reference_sample_id=reference_sample_id,
        assignments=assignments,
        usable_snvs=usable,
    )
