"""Tests for prolyg_phasing.ref_phasing."""
from __future__ import annotations

import numpy as np

from prolyg_phasing.io.panel import ExtractedLocus, ExtractedPanel, ExtractionProvenance
from prolyg_phasing.phasing import (
    LABEL_MAJOR,
    LABEL_MINOR,
    LABEL_UNK,
    FlankingHaplotype,
    SNVCandidate,
)
from prolyg_phasing.ref_phasing import (
    RefPhasingPanel,
    _remap_snv_to_target_position,
    assign_flanking_haplotypes_ref,
)


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
    up_width: int,
    dn_width: int,
    ref_orientation: str = "+",
    up_ref_start: int = 1000,
    dn_ref_start: int = 2000,
    bed_name: str = "LocusX",
) -> ExtractedLocus:
    """Synthesize ``counts[i]`` single-read families per unique flanking string.

    Every screen downstream counts read families, so ``counts`` is a family
    count: one row, one molecule, one read. Packing a string's whole count onto
    one row would make it a single molecule sequenced that many times, which
    carries one vote however deep it is — the fixture would then describe two
    molecules rather than the depth it names.
    """
    string_of_row = np.repeat(
        np.arange(len(flanking_strings), dtype=np.int32),
        np.asarray(counts, dtype=np.int64),
    )
    n_rows = int(string_of_row.size)
    mi = np.asarray([f"M{i}" for i in range(n_rows)], dtype=object)
    strand = np.asarray(["A"] * n_rows, dtype="U1")
    seq = np.asarray(["GGGGGG"] * n_rows, dtype=object)
    flanking_id = string_of_row
    count = np.ones(n_rows, dtype=np.int64)
    flanking_seq = np.asarray(flanking_strings, dtype=object)
    total = int(count.sum())

    return ExtractedLocus(
        mi=mi, strand=strand, seq=seq, count=count,
        n_runs=1, ref_inter_anchor_seq="GGGGGG", reference_run_lengths=(6,),
        chrom="chr1", bed_start=100, bed_end=106, bed_name=bed_name,
        ref_orientation=ref_orientation,
        upstream_anchor="ACGTACGTAC", downstream_anchor="TGCATGCATG",
        anchorability_status="anchorable",
        max_observed_run_length=6,
        n_alignments_overlap_locus=total * 2,
        n_alignments_drop_mapq=0, n_alignments_drop_sa=0,
        n_alignments_drop_tlen=0, n_alignments_drop_anchor=0,
        n_read_pairs=total, n_read_pairs_drop_disagree=0, n_reads=total,
        flanking_id=flanking_id, flanking_seq=flanking_seq,
        flanking_up_width=up_width, flanking_dn_width=dn_width,
        flanking_up_ref_pos_start=up_ref_start,
        flanking_dn_ref_pos_start=dn_ref_start,
    )


def _panel(locus_id: str, locus: ExtractedLocus) -> ExtractedPanel:
    return ExtractedPanel(
        loci={locus_id: locus}, n_alleles=7, provenance=_make_provenance(),
    )


def _reference_hap(ref_pos: int = 1002, major: str = "A", minor: str = "C"):
    """One-SNV reference FlankingHaplotype; string_index intentionally != target offset."""
    snv = SNVCandidate(
        locus_id="L1", ref_pos=ref_pos, segment="up", string_index=2,
        allele_major=major, allele_minor=minor, vaf_minor=0.4, coverage=60,
    )
    return FlankingHaplotype(
        locus_id="L1", snvs=(snv,), hap_major_profile=(major,),
        hap_minor_profile=(minor,), anchor_index=0, pair_discordance=(0.0,),
        pair_double_coverage=(60,), max_discordance=0.0, n_anchor_families=60,
        dropped_low_linkage_snvs=(),
    )


def test_ref_phase_remap_uses_ref_pos_not_string_index():
    # Target window shifted left and wider: ref_pos 1002 -> target offset 4
    # (reference string_index is 2 and must be ignored).
    target_locus = _make_locus(
        flanking_strings=["TTTTATT|", "TTTTCTT|", "TTTT.TT|", "TTTTGTT|"],
        counts=[1, 1, 1, 1], up_width=7, dn_width=0, up_ref_start=998,
    )
    out = assign_flanking_haplotypes_ref(
        _panel("L1", target_locus), {"L1": _reference_hap()},
        target_sample_id="TARGET", reference_sample_id="REF",
    )
    assert isinstance(out, RefPhasingPanel)
    a = out.assignments["L1"]
    assert list(a.label) == [LABEL_MAJOR, LABEL_MINOR, LABEL_UNK, LABEL_UNK]
    assert list(a.n_major) == [1, 0, 0, 0]
    assert list(a.n_minor) == [0, 1, 0, 0]
    assert list(a.n_other) == [0, 0, 0, 1]
    assert len(out.usable_snvs["L1"]) == 1
    assert out.usable_snvs["L1"][0].ref_pos == 1002


def test_ref_phase_drops_snv_outside_target_window():
    snv_cov = SNVCandidate("L1", 1002, "up", 2, "A", "C", 0.4, 60)
    snv_out = SNVCandidate("L1", 990, "up", 0, "A", "C", 0.4, 60)
    hap = FlankingHaplotype(
        locus_id="L1", snvs=(snv_cov, snv_out),
        hap_major_profile=("A", "A"), hap_minor_profile=("C", "C"),
        anchor_index=0, pair_discordance=(0.0, 0.0),
        pair_double_coverage=(60, 60), max_discordance=0.0,
        n_anchor_families=60, dropped_low_linkage_snvs=(),
    )
    target_locus = _make_locus(
        flanking_strings=["TTTTATT|"], counts=[1],
        up_width=7, dn_width=0, up_ref_start=998,
    )
    out = assign_flanking_haplotypes_ref(
        _panel("L1", target_locus), {"L1": hap},
        target_sample_id="TARGET", reference_sample_id="REF",
    )
    usable = out.usable_snvs["L1"]
    assert len(usable) == 1
    assert usable[0].ref_pos == 1002
    assert list(out.assignments["L1"].label) == [LABEL_MAJOR]


def test_ref_phase_missing_locus_all_unk():
    target_locus = _make_locus(
        flanking_strings=["TTTTATT|", "TTTTCTT|"], counts=[1, 1],
        up_width=7, dn_width=0, up_ref_start=998,
    )
    out = assign_flanking_haplotypes_ref(
        _panel("L1", target_locus), {},  # no reference phasing for L1
        target_sample_id="TARGET", reference_sample_id="REF",
    )
    assert list(out.assignments["L1"].label) == [LABEL_UNK, LABEL_UNK]
    assert out.usable_snvs["L1"] == ()


def test_remap_minus_orientation():
    locus = _make_locus(
        flanking_strings=["AAAAA|"], counts=[1],
        up_width=5, dn_width=0, ref_orientation="-",
        up_ref_start=2000, dn_ref_start=3000,
    )
    snv_in = SNVCandidate("L1", 1998, "up", 0, "A", "C", 0.4, 60)
    snv_out = SNVCandidate("L1", 2010, "up", 0, "A", "C", 0.4, 60)
    assert _remap_snv_to_target_position(snv_in, locus) == 2
    assert _remap_snv_to_target_position(snv_out, locus) is None


def test_ref_phase_matches_when_reference_loaded_from_snv_calls_table(tmp_path):
    """A `--reference-snv-calls` CLI run must match a `--reference-phasing-pkl` run."""
    import pandas as pd

    from prolyg_phasing.io.phasing_tables import (
        build_snv_calls,
        haplotypes_from_snv_calls_table,
    )
    from prolyg_phasing.phasing import PhasingPanel

    reference_locus = _make_locus(
        flanking_strings=["CA|TT", "GG|TT"], counts=[70, 30],
        up_width=2, dn_width=2,
    )
    reference_pp = PhasingPanel.from_panel(_panel("L1", reference_locus))
    assert len(reference_pp.haplotypes["L1"].snvs) == 2  # both up positions informative

    df = build_snv_calls(reference_pp)
    path = tmp_path / "snv_calls.tsv"
    df.to_csv(path, sep="\t", index=False)
    reloaded = pd.read_csv(path, sep="\t")
    from_table = haplotypes_from_snv_calls_table(reloaded)

    target_locus = _make_locus(
        flanking_strings=["TTCATTT|", "TTGGTTT|", "TTTTTTT|"],
        counts=[1, 1, 1], up_width=7, dn_width=0, up_ref_start=998,
    )
    target_panel = _panel("L1", target_locus)

    out_pkl = assign_flanking_haplotypes_ref(
        target_panel, reference_pp.haplotypes,
        target_sample_id="TARGET", reference_sample_id="REF",
    )
    out_table = assign_flanking_haplotypes_ref(
        target_panel, from_table,
        target_sample_id="TARGET", reference_sample_id="REF",
    )

    a_pkl, a_table = out_pkl.assignments["L1"], out_table.assignments["L1"]
    assert list(a_pkl.label) == [LABEL_MAJOR, LABEL_MINOR, LABEL_UNK]
    assert np.array_equal(a_pkl.label, a_table.label)
    assert np.array_equal(a_pkl.n_major, a_table.n_major)
    assert np.array_equal(a_pkl.n_minor, a_table.n_minor)
    assert np.array_equal(a_pkl.n_other, a_table.n_other)
    assert out_pkl.usable_snvs["L1"] == out_table.usable_snvs["L1"]
