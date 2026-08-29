"""Three panel-wide output tables over an already-phased panel.

Every column here is a direct read of a field :mod:`prolyg_phasing.phasing`
or :mod:`prolyg_phasing.io.panel` already computes. The one genuinely new
number, ``phase_interrupter_concordance_major/minor`` in
:func:`build_locus_summary`, joins two orthogonal evidence channels that
already exist (the flanking-SNV phase label and the per-family majority
interrupter pattern) and asks whether they agree.

Lazy ``pandas`` import in each builder, matching
``ExtractedLocus.pattern_breakdown``'s convention (`pyproject.toml`: the
``io`` module's DataFrame helpers stay lazy so the ``bam`` extra remains
pysam-only).
"""
from __future__ import annotations

import numpy as np

from prolyg_phasing.io.format import format_pattern
from prolyg_phasing.io.panel import ExtractedPanel, majority_pattern_per_rf
from prolyg_phasing.phasing import LABEL_MAJOR, LABEL_MINOR, LABEL_UNK, PhasingPanel

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
