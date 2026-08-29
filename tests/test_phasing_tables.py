"""Tests for prolyg_phasing.io.phasing_tables."""
from __future__ import annotations

import numpy as np
import pandas as pd

from prolyg_phasing.io.panel import ExtractedLocus, ExtractedPanel, ExtractionProvenance
from prolyg_phasing.io.phasing_tables import (
    build_haplotype_calls,
    build_locus_summary,
    build_snv_calls,
)
from prolyg_phasing.phasing import PhasingPanel


def _make_provenance() -> ExtractionProvenance:
    return ExtractionProvenance(
        bam_path="/synthetic/test.bam",
        bed_path="/synthetic/test.bed",
        extraction_version="0.0.0-test",
        extraction_date="2026-05-22",
        min_mapq=40,
        drop_chimeric=True,
        max_tlen=1000,
        anchor_hamming_max=1,
        pair_disagreement_policy="drop",
        n_alleles_margin=1,
        extraction_schema_version="2",
    )


def _make_locus(
    *,
    flanking_strings: list[str],
    counts: list[int],
    seqs: list[str],
    up_width: int,
    dn_width: int,
) -> ExtractedLocus:
    """One family (one row, one read) per count unit.

    Unlike ``test_phasing.py``'s ``_make_locus``, ``seqs`` gives per-row
    control of the interrupter-pattern string independent of the flanking
    string, so a fixture can set the SNV-phasing group and the
    interrupter-pattern group independently — needed to test
    ``phase_interrupter_concordance``.
    """
    string_of_row = np.repeat(
        np.arange(len(flanking_strings), dtype=np.int32),
        np.asarray(counts, dtype=np.int64),
    )
    n_rows = int(string_of_row.size)
    assert len(seqs) == n_rows
    mi = np.asarray([f"M{i}" for i in range(n_rows)], dtype=object)
    strand = np.asarray(["A"] * n_rows, dtype="U1")
    seq = np.asarray(seqs, dtype=object)
    flanking_id = string_of_row
    count = np.ones(n_rows, dtype=np.int64)
    flanking_seq = np.asarray(flanking_strings, dtype=object)
    total = int(count.sum())

    return ExtractedLocus(
        mi=mi, strand=strand, seq=seq, count=count,
        n_runs=1, ref_inter_anchor_seq="GGGGGG", reference_run_lengths=(6,),
        chrom="chr1", bed_start=100, bed_end=106, bed_name="LocusX",
        ref_orientation="+",
        upstream_anchor="ACGTACGTAC", downstream_anchor="TGCATGCATG",
        anchorability_status="anchorable",
        max_observed_run_length=6,
        n_alignments_overlap_locus=total * 2,
        n_alignments_drop_mapq=0, n_alignments_drop_sa=0,
        n_alignments_drop_tlen=0, n_alignments_drop_anchor=0,
        n_read_pairs=total, n_read_pairs_drop_disagree=0, n_reads=total,
        flanking_id=flanking_id, flanking_seq=flanking_seq,
        flanking_up_width=up_width, flanking_dn_width=dn_width,
        flanking_up_ref_pos_start=1000, flanking_dn_ref_pos_start=2000,
    )


def _panel(locus_id: str, locus: ExtractedLocus) -> ExtractedPanel:
    return ExtractedPanel(
        loci={locus_id: locus}, n_alleles=7, provenance=_make_provenance(),
    )


# ---------------------------------------------------------------------------
# build_haplotype_calls
# ---------------------------------------------------------------------------


def test_haplotype_calls_columns_and_grain():
    # 70 "major" families (flanking "CA|TT", interrupter pattern "A"), 30
    # "minor" families (flanking "GG|TT", pattern "-", no interrupters).
    locus = _make_locus(
        flanking_strings=["CA|TT", "GG|TT"], counts=[70, 30],
        seqs=["GGGAGGG"] * 70 + ["GGGGGG"] * 30,
        up_width=2, dn_width=2,
    )
    panel = _panel("L1", locus)
    pp = PhasingPanel.from_panel(panel)
    df = build_haplotype_calls(panel, pp)

    assert list(df.columns) == [
        "locus_id", "mi", "strand", "count", "interrupter_pattern",
        "phase_label", "n_major", "n_minor", "n_other",
    ]
    assert len(df) == 100  # one row per (mi, strand); one row per family here
    assert (df["locus_id"] == "L1").all()
    assert set(df["phase_label"]) == {"hap_major", "hap_minor"}
    assert (df.loc[df["phase_label"] == "hap_major", "interrupter_pattern"] == "A").all()
    assert (df.loc[df["phase_label"] == "hap_minor", "interrupter_pattern"] == "-").all()


def test_haplotype_calls_unk_when_no_informative_snvs():
    # A single flanking string carries no minor allele anywhere -> no SNVs.
    locus = _make_locus(
        flanking_strings=["TT|TT"], counts=[50], seqs=["GGGGGG"] * 50,
        up_width=2, dn_width=2,
    )
    panel = _panel("L1", locus)
    pp = PhasingPanel.from_panel(panel)
    df = build_haplotype_calls(panel, pp)
    assert (df["phase_label"] == "unk").all()
    assert (df["n_major"] == 0).all() and (df["n_minor"] == 0).all()


# ---------------------------------------------------------------------------
# build_snv_calls
# ---------------------------------------------------------------------------


def test_snv_calls_phased_and_dropped_rows():
    # up[0] (C/G) and up[1] (A/G) both pass the coverage+VAF filter (coverage
    # 63 each), but only 8 of the 63 molecules cover both jointly (the rest
    # have '.' at one position) -- below min_linkage_families=10, so up[1] is
    # dropped while the anchor up[0] (tie broken to list order) survives.
    locus = _make_locus(
        flanking_strings=["C.|", ".A|", "CA|", "G.|", ".G|", "GG|"],
        counts=[40, 40, 5, 15, 15, 3],
        seqs=["GGGGGG"] * 118,
        up_width=2, dn_width=0,
    )
    panel = _panel("L1", locus)
    pp = PhasingPanel.from_panel(panel)
    df = build_snv_calls(pp)

    assert list(df.columns) == [
        "locus_id", "ref_pos", "segment", "string_index", "allele_major",
        "allele_minor", "vaf_minor", "coverage", "status", "hap_major_base",
        "hap_minor_base", "is_anchor", "pair_discordance",
        "pair_double_coverage",
    ]
    assert len(df) == 2
    phased = df[df["status"] == "phased"]
    dropped = df[df["status"] == "dropped_low_linkage"]
    assert len(phased) == 1 and len(dropped) == 1
    assert phased.iloc[0]["string_index"] == 0
    assert bool(phased.iloc[0]["is_anchor"]) is True
    assert phased.iloc[0]["hap_major_base"] == "C"
    assert phased.iloc[0]["hap_minor_base"] == "G"
    assert dropped.iloc[0]["string_index"] == 1
    assert pd.isna(dropped.iloc[0]["is_anchor"])
    assert pd.isna(dropped.iloc[0]["hap_major_base"])
    assert pd.isna(dropped.iloc[0]["pair_discordance"])
    assert pd.isna(dropped.iloc[0]["pair_double_coverage"])


def test_snv_calls_empty_for_zero_snv_panel():
    locus = _make_locus(
        flanking_strings=["TT|TT"], counts=[50], seqs=["GGGGGG"] * 50,
        up_width=2, dn_width=2,
    )
    panel = _panel("L1", locus)
    pp = PhasingPanel.from_panel(panel)
    df = build_snv_calls(pp)
    assert len(df) == 0


# ---------------------------------------------------------------------------
# build_locus_summary
# ---------------------------------------------------------------------------


def test_locus_summary_perfect_concordance():
    locus = _make_locus(
        flanking_strings=["CA|TT", "GG|TT"], counts=[70, 30],
        seqs=["GGGAGGG"] * 70 + ["GGGGGG"] * 30,
        up_width=2, dn_width=2,
    )
    panel = _panel("L1", locus)
    pp = PhasingPanel.from_panel(panel)
    row = build_locus_summary(panel, pp).iloc[0]

    assert row["locus_id"] == "L1"
    assert row["n_rfs"] == 100
    assert row["n_rows"] == 100
    assert row["hap_major_profile"] == "CA"
    assert row["hap_minor_profile"] == "GG"
    assert row["n_snvs_phased"] == 2
    assert row["n_snvs_dropped_low_linkage"] == 0
    assert row["max_discordance"] == 0.0
    assert row["n_anchor_families"] == 100
    assert row["reads_total"] == 100
    assert row["reads_major"] == 70
    assert row["reads_minor"] == 30
    assert row["reads_unk"] == 0
    assert row["phase_interrupter_concordance_major"] == 1.0
    assert row["phase_interrupter_concordance_minor"] == 1.0
    # admitted_patterns is a min_freq=0.1-filtered, freq-descending join of
    # format_pattern(h):freq -- both patterns clear the default threshold.
    assert row["admitted_patterns"] == "A:0.7;-:0.3"
    assert row["n_admitted_patterns"] == 2


def test_locus_summary_concordance_matches_haplotype_calls_join():
    # 2 of the 70 "major" families carry the "minor" group's interrupter
    # pattern -- imperfect concordance, and the number must equal what a
    # consumer gets by joining haplotype_calls.tsv on phase_label.
    locus = _make_locus(
        flanking_strings=["CA|TT", "GG|TT"], counts=[70, 30],
        seqs=["GGGAGGG"] * 68 + ["GGGGGG"] * 2 + ["GGGGGG"] * 30,
        up_width=2, dn_width=2,
    )
    panel = _panel("L1", locus)
    pp = PhasingPanel.from_panel(panel)
    hc = build_haplotype_calls(panel, pp)
    row = build_locus_summary(panel, pp).iloc[0]

    major_patterns = hc.loc[hc["phase_label"] == "hap_major", "interrupter_pattern"]
    expected_major = major_patterns.value_counts().iloc[0] / len(major_patterns)
    minor_patterns = hc.loc[hc["phase_label"] == "hap_minor", "interrupter_pattern"]
    expected_minor = minor_patterns.value_counts().iloc[0] / len(minor_patterns)

    assert row["phase_interrupter_concordance_major"] == expected_major
    assert row["phase_interrupter_concordance_minor"] == expected_minor
    assert row["phase_interrupter_concordance_major"] == 68 / 70
    assert row["phase_interrupter_concordance_minor"] == 1.0


def test_locus_summary_nan_concordance_when_label_absent():
    locus = _make_locus(
        flanking_strings=["TT|TT"], counts=[50], seqs=["GGGGGG"] * 50,
        up_width=2, dn_width=2,
    )
    panel = _panel("L1", locus)
    pp = PhasingPanel.from_panel(panel)
    row = build_locus_summary(panel, pp).iloc[0]
    assert np.isnan(row["phase_interrupter_concordance_major"])
    assert np.isnan(row["phase_interrupter_concordance_minor"])
    assert row["hap_major_profile"] == ""
    assert row["hap_minor_profile"] == ""
    assert row["reads_unk"] == 50


def test_locus_summary_min_pattern_freq_filters_admitted_patterns():
    # A third, rare pattern (2/102 families) clears min_freq=0.0 but not the
    # default 0.1.
    locus = _make_locus(
        flanking_strings=["CA|TT", "GG|TT", "AA|TT"], counts=[70, 30, 2],
        seqs=["GGGAGGG"] * 70 + ["GGGGGG"] * 30 + ["GGGCGGG"] * 2,
        up_width=2, dn_width=2,
    )
    panel = _panel("L1", locus)
    pp = PhasingPanel.from_panel(panel)
    row_default = build_locus_summary(panel, pp).iloc[0]
    row_all = build_locus_summary(panel, pp, min_pattern_freq=0.0).iloc[0]
    assert row_default["n_admitted_patterns"] == 2
    assert row_all["n_admitted_patterns"] == 3
