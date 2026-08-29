"""Panel-wide output tables over an already-phased panel.

Every column here is a direct read of a field :mod:`prolyg_phasing.phasing`
or :mod:`prolyg_phasing.io.panel` already computes. The one genuinely new
number, ``phase_interrupter_concordance_major/minor`` in
:func:`build_locus_summary`, joins two orthogonal evidence channels that
already exist (the flanking-SNV phase label and the per-family majority
interrupter pattern) and asks whether they agree.

Three tables (``build_haplotype_calls``/``build_locus_summary``/
``build_snv_calls``) describe a same-panel :class:`~prolyg_phasing.phasing.PhasingPanel`.
A fourth, ``build_ref_phasing_summary``, describes an
:class:`~prolyg_phasing.ref_phasing.RefPhasingPanel` instead — a
different result type, since an anchored panel borrows its haplotype
profiles from a reference panel rather than phasing its own.

:func:`haplotypes_from_snv_calls_table` is the one function here that goes
the other way, table to object: it reconstructs the
``dict[str, FlankingHaplotype]`` an anchored-phasing run needs directly
from a ``snv_calls.tsv``, without unpickling the reference's full
``phasing.pkl`` — which also carries ``PhasingPanel.assignments``, an
O(reads) structure the anchored path never reads.

Lazy ``pandas`` import in each builder, matching
``ExtractedLocus.pattern_breakdown``'s convention (`pyproject.toml`: the
``io`` module's DataFrame helpers stay lazy so the ``bam`` extra remains
pysam-only).
"""
from __future__ import annotations

import numpy as np

from prolyg_phasing.io.format import format_pattern
from prolyg_phasing.io.panel import ExtractedPanel, majority_pattern_per_rf
from prolyg_phasing.phasing import (
    LABEL_MAJOR,
    LABEL_MINOR,
    LABEL_UNK,
    FlankingHaplotype,
    PhasingPanel,
    SNVCandidate,
)
from prolyg_phasing.ref_phasing import RefPhasingPanel

_LABEL_NAME = {LABEL_MAJOR: "hap_major", LABEL_MINOR: "hap_minor", LABEL_UNK: "unk"}


def build_haplotype_calls(panel: ExtractedPanel, pp: PhasingPanel):
    """One row per ``(locus, mi, strand)`` — the finest grain already on file.

    Columns: ``locus_id``, ``mi``, ``strand``, ``count``,
    ``interrupter_pattern`` (per-family majority pattern, from
    :meth:`~prolyg_phasing.io.panel.ExtractedLocus.pattern_breakdown`),
    ``phase_label`` (``hap_major``/``hap_minor``/``unk``), ``n_major``,
    ``n_minor``, ``n_other`` (the vote evidence behind the label).

    Not collapsed to one row per family: a family can carry both strands and
    they can disagree.
    """
    import pandas as pd

    frames = []
    for locus_id, locus in panel.loci.items():
        pb = locus.pattern_breakdown()
        assignment = pp.assignments[locus_id]
        frames.append(pd.DataFrame({
            "locus_id": locus_id,
            "mi": pb["mi"],
            "strand": pb["strand"],
            "count": pb["count"],
            "interrupter_pattern": pb["pattern"],
            "phase_label": [_LABEL_NAME[int(x)] for x in assignment.label],
            "n_major": assignment.n_major,
            "n_minor": assignment.n_minor,
            "n_other": assignment.n_other,
        }))
    if not frames:
        return pd.DataFrame(columns=[
            "locus_id", "mi", "strand", "count", "interrupter_pattern",
            "phase_label", "n_major", "n_minor", "n_other",
        ])
    return pd.concat(frames, ignore_index=True)


def build_snv_calls(pp: PhasingPanel):
    """One row per ``(locus, SNV)``, both phased and dropped-for-low-linkage.

    Columns: ``locus_id``, ``ref_pos``, ``segment``, ``string_index``,
    ``allele_major``, ``allele_minor``, ``vaf_minor``, ``coverage``,
    ``status`` (``phased`` / ``dropped_low_linkage``), and — blank for a
    dropped SNV, since it was never assigned a phase — ``hap_major_base``,
    ``hap_minor_base`` (from ``hap_major_profile[k]``/``hap_minor_profile[k]``;
    these can differ from ``allele_major``/``allele_minor`` when a SNV was
    flipped during linkage resolution), ``is_anchor``, ``pair_discordance``,
    ``pair_double_coverage``.
    """
    import pandas as pd

    rows: list[dict] = []
    for locus_id, hap in pp.haplotypes.items():
        for k, snv in enumerate(hap.snvs):
            rows.append({
                "locus_id": locus_id,
                "ref_pos": snv.ref_pos,
                "segment": snv.segment,
                "string_index": snv.string_index,
                "allele_major": snv.allele_major,
                "allele_minor": snv.allele_minor,
                "vaf_minor": snv.vaf_minor,
                "coverage": snv.coverage,
                "status": "phased",
                "hap_major_base": hap.hap_major_profile[k],
                "hap_minor_base": hap.hap_minor_profile[k],
                "is_anchor": k == hap.anchor_index,
                "pair_discordance": hap.pair_discordance[k],
                "pair_double_coverage": hap.pair_double_coverage[k],
            })
        for snv in hap.dropped_low_linkage_snvs:
            rows.append({
                "locus_id": locus_id,
                "ref_pos": snv.ref_pos,
                "segment": snv.segment,
                "string_index": snv.string_index,
                "allele_major": snv.allele_major,
                "allele_minor": snv.allele_minor,
                "vaf_minor": snv.vaf_minor,
                "coverage": snv.coverage,
                "status": "dropped_low_linkage",
                "hap_major_base": None,
                "hap_minor_base": None,
                "is_anchor": None,
                "pair_discordance": None,
                "pair_double_coverage": None,
            })
    columns = [
        "locus_id", "ref_pos", "segment", "string_index", "allele_major",
        "allele_minor", "vaf_minor", "coverage", "status", "hap_major_base",
        "hap_minor_base", "is_anchor", "pair_discordance",
        "pair_double_coverage",
    ]
    return pd.DataFrame(rows, columns=columns)


def _modal_pattern_share(patterns: np.ndarray, mask: np.ndarray) -> float:
    """Modal value's share of ``patterns[mask]``; ``NaN`` when ``mask`` is empty."""
    n = int(mask.sum())
    if n == 0:
        return float("nan")
    _, counts = np.unique(patterns[mask], return_counts=True)
    return float(counts.max() / n)


def build_locus_summary(panel: ExtractedPanel, pp: PhasingPanel, *, min_pattern_freq: float = 0.1):
    """One row per locus: family count, admitted patterns, phasing QC.

    ``phase_interrupter_concordance_major``/``_minor`` join the flanking-SNV
    phase label against the per-family majority interrupter pattern used in
    :func:`build_haplotype_calls`: among rows labeled ``hap_major``
    (``hap_minor``), the modal interrupter pattern's share. Row-weighted
    (each deduplicated ``(mi, strand)`` row counts once), not read-weighted,
    so the number is exactly what a consumer gets by filtering
    ``haplotype_calls.tsv`` on ``phase_label`` and taking the modal
    ``interrupter_pattern``'s share. ``NaN`` when a locus has no rows with
    that label.

    ``min_pattern_freq`` (default 0.1) is the display threshold for
    ``admitted_patterns``; it is this table's own key, independent of
    ``fit.min_pattern_freq`` (the founder-admission gate the PCR-chemistry
    fit uses).
    """
    import pandas as pd

    rows: list[dict] = []
    for locus_id, locus in panel.loci.items():
        mi = np.asarray(locus.mi)
        n_rfs = int(np.unique(mi).size)
        n_rows = int(mi.shape[0])

        # One majority_pattern_per_rf pass serves both the per-family
        # frequencies (observed_pattern_frequencies' own logic, inlined) and
        # the per-row pattern array (pattern_breakdown's own logic, inlined)
        # -- each of those two helpers, plus select_patterns_above_freq,
        # would otherwise repeat this same per-row scan.
        mi_to_majority = majority_pattern_per_rf(locus)
        pattern_counts: dict[tuple[str, ...], int] = {}
        for majority in mi_to_majority.values():
            pattern_counts[majority] = pattern_counts.get(majority, 0) + 1
        freqs = (
            {h: c / n_rfs for h, c in pattern_counts.items()} if n_rfs else {}
        )
        admitted = {h for h, f in freqs.items() if f >= min_pattern_freq}
        admitted_sorted = sorted(admitted, key=lambda h: -freqs[h])
        admitted_patterns = ";".join(
            f"{format_pattern(h)}:{freqs[h]:.4g}" for h in admitted_sorted
        )

        hap = pp.haplotypes[locus_id]
        assignment = pp.assignments[locus_id]
        counts = np.asarray(locus.count, dtype=np.int64)
        labels = assignment.label

        pb_pattern = np.asarray(
            [format_pattern(mi_to_majority.get(str(m), ())) for m in mi],
            dtype=object,
        )
        is_major = labels == LABEL_MAJOR
        is_minor = labels == LABEL_MINOR

        rows.append({
            "locus_id": locus_id,
            "n_rfs": n_rfs,
            "n_rows": n_rows,
            "admitted_patterns": admitted_patterns,
            "n_admitted_patterns": len(admitted),
            "hap_major_profile": "".join(hap.hap_major_profile),
            "hap_minor_profile": "".join(hap.hap_minor_profile),
            "n_snvs_phased": len(hap.snvs),
            "n_snvs_dropped_low_linkage": len(hap.dropped_low_linkage_snvs),
            "max_discordance": hap.max_discordance,
            "n_anchor_families": hap.n_anchor_families,
            "reads_total": int(counts.sum()),
            "reads_major": int(counts[is_major].sum()),
            "reads_minor": int(counts[is_minor].sum()),
            "reads_unk": int(counts[labels == LABEL_UNK].sum()),
            "phase_interrupter_concordance_major": _modal_pattern_share(pb_pattern, is_major),
            "phase_interrupter_concordance_minor": _modal_pattern_share(pb_pattern, is_minor),
        })
    return pd.DataFrame(rows)


def build_ref_phasing_summary(target_panel: ExtractedPanel, anchored: RefPhasingPanel):
    """One row per locus: anchored-phasing QC over the target panel.

    Unlike :func:`build_locus_summary`, the target panel never phases its own
    haplotypes — the profile is borrowed from the reference panel
    (:func:`~prolyg_phasing.ref_phasing.assign_flanking_haplotypes_ref`)
    — so there is no ``hap_major_profile``/interrupter-concordance pair here,
    only the vote outcome and the SNV coverage that produced it.

    Columns: ``locus_id``, ``n_usable_snvs`` (reference SNVs the target
    locus's own flanking window covers, after the ``ref_pos`` remap),
    ``n_rows``, ``n_families``/``n_families_labeled`` (distinct molecules;
    the labeled subset excludes ``unk``), ``reads_total``/``reads_major``/
    ``reads_minor``/``reads_unk`` (read-count-weighted, summed
    ``locus.count``).
    """
    import pandas as pd

    rows: list[dict] = []
    for locus_id, locus in target_panel.loci.items():
        assignment = anchored.assignments[locus_id]
        labels = assignment.label
        counts = np.asarray(locus.count, dtype=np.int64)
        mi = np.asarray(locus.mi)
        labeled = np.unique(mi[(labels == LABEL_MAJOR) | (labels == LABEL_MINOR)])
        rows.append({
            "locus_id": locus_id,
            "n_usable_snvs": len(anchored.usable_snvs[locus_id]),
            "n_rows": int(labels.shape[0]),
            "n_families": int(np.unique(mi).size),
            "n_families_labeled": int(labeled.size),
            "reads_total": int(counts.sum()),
            "reads_major": int(counts[labels == LABEL_MAJOR].sum()),
            "reads_minor": int(counts[labels == LABEL_MINOR].sum()),
            "reads_unk": int(counts[labels == LABEL_UNK].sum()),
        })
    return pd.DataFrame(rows)


def _snv_from_row(locus_id: str, r) -> SNVCandidate:
    """Rebuild one :class:`~prolyg_phasing.phasing.SNVCandidate` from a ``build_snv_calls`` row.

    Shared by the phased and dropped-for-low-linkage reconstruction loops in
    :func:`haplotypes_from_snv_calls_table` — both read the same seven
    ``build_snv_calls`` columns.
    """
    return SNVCandidate(
        locus_id=locus_id, ref_pos=int(r.ref_pos), segment=r.segment,
        string_index=int(r.string_index), allele_major=r.allele_major,
        allele_minor=r.allele_minor, vaf_minor=float(r.vaf_minor),
        coverage=int(r.coverage),
    )


def haplotypes_from_snv_calls_table(df) -> dict[str, FlankingHaplotype]:
    """Reconstruct per-locus :class:`~prolyg_phasing.phasing.FlankingHaplotype`
    from a ``build_snv_calls`` table (e.g. loaded back from ``snv_calls.tsv``).

    Exact inverse of :func:`build_snv_calls` on the fields it wrote: every
    phased row's ``ref_pos``/``segment``/``string_index``/``allele_major``/
    ``allele_minor``/``vaf_minor``/``coverage`` rebuilds its
    :class:`~prolyg_phasing.phasing.SNVCandidate`, ``hap_major_base``/
    ``hap_minor_base`` rebuild the two profiles, and ``is_anchor``/
    ``pair_discordance``/``pair_double_coverage`` rebuild the rest. Row order
    within a locus is preserved (the profile's ``k``-th entry stays the
    table's ``k``-th row for that locus), so this reconstructs the original
    :class:`~prolyg_phasing.phasing.FlankingHaplotype` field for field, not
    just an equivalent-behaving stand-in.

    A locus with zero informative SNVs contributes no rows to
    ``build_snv_calls`` at all, so it has no key here either — the same
    "locus absent means zero-SNV" convention
    :func:`~prolyg_phasing.ref_phasing.assign_flanking_haplotypes_ref`
    already applies to a ``phasing.pkl``-sourced ``reference_phased`` dict
    that never has a missing key. A caller iterating this dict directly
    (rather than through ``assign_flanking_haplotypes_ref``) must
    account for that difference.
    """
    out: dict[str, FlankingHaplotype] = {}
    for locus_id, locus_df in df.groupby("locus_id", sort=False):
        locus_id = str(locus_id)
        phased = locus_df[locus_df["status"] == "phased"]
        dropped = locus_df[locus_df["status"] == "dropped_low_linkage"]

        snvs = tuple(_snv_from_row(locus_id, r) for r in phased.itertuples())
        dropped_snvs = tuple(_snv_from_row(locus_id, r) for r in dropped.itertuples())

        anchor_positions = [i for i, v in enumerate(phased["is_anchor"]) if bool(v)]
        anchor_index = anchor_positions[0] if anchor_positions else -1
        pair_discordance = tuple(float(v) for v in phased["pair_discordance"])
        non_anchor_disc = [
            d for i, d in enumerate(pair_discordance) if i != anchor_index
        ]
        n_anchor_families = (
            int(phased["coverage"].iloc[anchor_index]) if anchor_index >= 0 else 0
        )

        out[locus_id] = FlankingHaplotype(
            locus_id=locus_id,
            snvs=snvs,
            hap_major_profile=tuple(phased["hap_major_base"]),
            hap_minor_profile=tuple(phased["hap_minor_base"]),
            anchor_index=anchor_index,
            pair_discordance=pair_discordance,
            pair_double_coverage=tuple(int(v) for v in phased["pair_double_coverage"]),
            max_discordance=max(non_anchor_disc) if non_anchor_disc else 0.0,
            n_anchor_families=n_anchor_families,
            dropped_low_linkage_snvs=dropped_snvs,
        )
    return out
