"""Tests for prolyg_phasing.phasing."""
from __future__ import annotations

import numpy as np
import pytest

from prolyg_phasing.io.panel import ExtractedLocus, ExtractedPanel, ExtractionProvenance
from prolyg_phasing.phasing import (
    LABEL_MAJOR,
    LABEL_MINOR,
    LABEL_UNK,
    PhasingPanel,
    aggregate_position_frequencies,
    assign_flanking_haplotypes,
    determine_flanking_haplotypes,
    find_informative_snvs,
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


def _make_locus_from_rows(
    rows: list[tuple[str, str, int]],
    *,
    up_width: int,
    dn_width: int,
    bed_name: str = "L",
) -> ExtractedLocus:
    """Build a locus row by row: ``(flanking string, molecule id, read count)``.

    Unlike :func:`_make_locus`, which gives every molecule one read, this puts
    the depth of a molecule and the identity of a molecule under the caller's
    control — the only way to write a fixture where the read count and the
    family count of the same observation differ.
    """
    strings = sorted({s for s, _, _ in rows})
    index_of = {s: i for i, s in enumerate(strings)}
    total = sum(c for _, _, c in rows)
    return ExtractedLocus(
        mi=np.asarray([m for _, m, _ in rows], dtype=object),
        strand=np.asarray(["A"] * len(rows), dtype="U1"),
        seq=np.asarray(["GGGGGG"] * len(rows), dtype=object),
        count=np.asarray([c for _, _, c in rows], dtype=np.int64),
        n_runs=1, ref_inter_anchor_seq="GGGGGG", reference_run_lengths=(6,),
        chrom="chr1", bed_start=100, bed_end=106, bed_name=bed_name,
        ref_orientation="+",
        upstream_anchor="ACGTACGTAC", downstream_anchor="TGCATGCATG",
        anchorability_status="anchorable", max_observed_run_length=6,
        n_alignments_overlap_locus=total * 2,
        n_alignments_drop_mapq=0, n_alignments_drop_sa=0,
        n_alignments_drop_tlen=0, n_alignments_drop_anchor=0,
        n_read_pairs=total, n_read_pairs_drop_disagree=0, n_reads=total,
        flanking_id=np.asarray([index_of[s] for s, _, _ in rows], dtype=np.int32),
        flanking_seq=np.asarray(strings, dtype=object),
        flanking_up_width=up_width, flanking_dn_width=dn_width,
        flanking_up_ref_pos_start=1000, flanking_dn_ref_pos_start=2000,
    )


def _panel(locus_id: str, locus: ExtractedLocus) -> ExtractedPanel:
    return ExtractedPanel(
        loci={locus_id: locus}, n_alleles=7, provenance=_make_provenance(),
    )


def _row_of(counts: list[int], string_index: int) -> int:
    """Index of the first row ``_make_locus`` synthesized for a flanking string.

    Rows are emitted string by string, so a string's block starts at the sum of
    the counts before it.
    """
    return int(sum(counts[:string_index]))


def _blocks(counts: list[int], values: list) -> list:
    """``values[i]`` repeated ``counts[i]`` times — one entry per synthesized row."""
    return [v for c, v in zip(counts, values, strict=True) for _ in range(c)]


# ---------------------------------------------------------------------------
# aggregate_position_frequencies
# ---------------------------------------------------------------------------


def test_aggregate_handcomputed_3_strings():
    flanking_seq = np.asarray(["AA|CCG", "AT|CCG", "AT|CCC"], dtype=object)
    flanking_id = np.asarray([0, 1, 2], dtype=np.int32)
    count = np.asarray([10, 20, 5], dtype=np.int64)

    up_freqs, dn_freqs = aggregate_position_frequencies(
        flanking_seq, flanking_id, count,
        flanking_up_width=2, flanking_dn_width=3,
    )
    # up[0]: all three 'A' → A=35
    assert up_freqs[0, 0] == 35
    assert up_freqs[0].sum() == 35
    # up[1]: A=10, T=25
    assert up_freqs[1, 0] == 10
    assert up_freqs[1, 3] == 25
    # dn[0], dn[1]: all 'C' → 35 each
    assert dn_freqs[0, 1] == 35
    assert dn_freqs[1, 1] == 35
    # dn[2]: G=30 (10+20), C=5
    assert dn_freqs[2, 2] == 30
    assert dn_freqs[2, 1] == 5


def test_aggregate_with_dot_padding():
    flanking_seq = np.asarray(["A.|CG", "AT|CG"], dtype=object)
    flanking_id = np.asarray([0, 1], dtype=np.int32)
    count = np.asarray([10, 5], dtype=np.int64)
    up_freqs, _ = aggregate_position_frequencies(
        flanking_seq, flanking_id, count, 2, 2,
    )
    # up[1]: '.' weight 10, 'T' weight 5
    assert up_freqs[1, 5] == 10
    assert up_freqs[1, 3] == 5


def test_aggregate_multiple_rows_per_fid_summed():
    flanking_seq = np.asarray(["AT|CG", "GC|TA"], dtype=object)
    flanking_id = np.asarray([0, 1, 0, 1], dtype=np.int32)
    count = np.asarray([3, 5, 7, 2], dtype=np.int64)
    # fid 0 weight 10 (3+7), fid 1 weight 7 (5+2)
    up_freqs, _ = aggregate_position_frequencies(
        flanking_seq, flanking_id, count, 2, 2,
    )
    assert up_freqs[0, 0] == 10  # 'A' from fid 0
    assert up_freqs[0, 2] == 7   # 'G' from fid 1


def test_aggregate_asserts_on_negative_fid():
    flanking_seq = np.asarray(["AT|CG"], dtype=object)
    flanking_id = np.asarray([-1], dtype=np.int32)
    count = np.asarray([10], dtype=np.int64)
    with pytest.raises(AssertionError):
        aggregate_position_frequencies(flanking_seq, flanking_id, count, 2, 2)


def test_aggregate_raises_on_bad_string_length():
    flanking_seq = np.asarray(["AT|CG", "ATX|CG"], dtype=object)
    flanking_id = np.asarray([0, 1], dtype=np.int32)
    count = np.asarray([10, 5], dtype=np.int64)
    with pytest.raises(ValueError, match="char-array shape mismatch"):
        aggregate_position_frequencies(flanking_seq, flanking_id, count, 2, 2)


# ---------------------------------------------------------------------------
# find_informative_snvs
# ---------------------------------------------------------------------------


def test_single_snv_50_50():
    locus = _make_locus(
        flanking_strings=["AT|GC", "CT|GC"],
        counts=[100, 100],
        up_width=2, dn_width=2,
    )
    out = find_informative_snvs(_panel("L1", locus))
    [c] = out["L1"]
    assert c.locus_id == "L1"
    assert c.segment == "up"
    assert c.string_index == 0
    assert c.coverage == 200
    assert c.vaf_minor == pytest.approx(0.5)
    assert {c.allele_major, c.allele_minor} == {"A", "C"}


def test_vaf_below_min_excluded():
    # 95 / 5 split → minor VAF 0.05 < 0.2
    locus = _make_locus(
        flanking_strings=["AT|GC", "CT|GC"],
        counts=[95, 5],
        up_width=2, dn_width=2,
    )
    out = find_informative_snvs(_panel("L1", locus))
    assert out["L1"] == []


def test_coverage_floor_excluded():
    # ACGT family depth 20 < default min_coverage=24
    locus = _make_locus(
        flanking_strings=["AT|GC", "CT|GC"],
        counts=[10, 10],
        up_width=2, dn_width=2,
    )
    out = find_informative_snvs(_panel("L1", locus))
    assert out["L1"] == []


def test_ref_pos_plus_orientation():
    locus = _make_locus(
        flanking_strings=["AT|CG", "CT|CG"],
        counts=[100, 100],
        up_width=2, dn_width=2,
        ref_orientation="+",
        up_ref_start=1000, dn_ref_start=2000,
    )
    [c] = find_informative_snvs(_panel("L1", locus))["L1"]
    assert c.ref_pos == 1000  # up[0]: 1000 + 0


def test_ref_pos_minus_orientation():
    locus = _make_locus(
        flanking_strings=["AT|CG", "AT|CC"],  # SNV at dn[1]
        counts=[100, 100],
        up_width=2, dn_width=2,
        ref_orientation="-",
        up_ref_start=1000, dn_ref_start=2000,
    )
    [c] = find_informative_snvs(_panel("L1", locus))["L1"]
    assert c.segment == "dn"
    assert c.string_index == 1
    # - orientation: dn_ref_start - i = 2000 - 1
    assert c.ref_pos == 1999


def test_tri_allelic_drops_third():
    # 60 A, 30 C, 10 G at up[0]; major=A, minor=C, G silently dropped
    locus = _make_locus(
        flanking_strings=["AT|CG", "CT|CG", "GT|CG"],
        counts=[60, 30, 10],
        up_width=2, dn_width=2,
    )
    [c] = find_informative_snvs(_panel("L1", locus))["L1"]
    assert c.allele_major == "A"
    assert c.allele_minor == "C"
    assert c.coverage == 100
    assert c.vaf_minor == pytest.approx(0.3)


def test_n_invisible_to_depth_and_allele():
    # 50 A, 50 N at up[0]: ACGT-depth 50, no minor real base
    locus = _make_locus(
        flanking_strings=["AT|CG", "NT|CG"],
        counts=[50, 50],
        up_width=2, dn_width=2,
    )
    out = find_informative_snvs(_panel("L1", locus))
    # ACGT-depth 50 ≥ 50, but no minor base → no candidate
    assert out["L1"] == []


def test_dot_excluded_from_depth():
    # 50 A, 50 '.' at up[0]: ACGT-depth 50, no minor → no candidate
    locus = _make_locus(
        flanking_strings=["AT|CG", ".T|CG"],
        counts=[50, 50],
        up_width=2, dn_width=2,
    )
    out = find_informative_snvs(_panel("L1", locus))
    assert out["L1"] == []


def test_empty_flanking_locus_short_circuits():
    # Schema-v1 locus (no flanking) — up/dn widths = 0
    locus = _make_locus(
        flanking_strings=[], counts=[],
        up_width=0, dn_width=0,
    )
    out = find_informative_snvs(_panel("L1", locus))
    assert out == {"L1": []}


def test_every_locus_keyed_in_output():
    locus_with_snv = _make_locus(
        flanking_strings=["AT|CG", "CT|CG"], counts=[100, 100],
        up_width=2, dn_width=2, bed_name="L1",
    )
    locus_without = _make_locus(
        flanking_strings=["AT|CG"], counts=[100],
        up_width=2, dn_width=2, bed_name="L2",
    )
    panel = ExtractedPanel(
        loci={"L1": locus_with_snv, "L2": locus_without},
        n_alleles=7, provenance=_make_provenance(),
    )
    out = find_informative_snvs(panel)
    assert set(out.keys()) == {"L1", "L2"}
    assert len(out["L1"]) == 1
    assert out["L2"] == []


# ---------------------------------------------------------------------------
# determine_flanking_haplotypes
# ---------------------------------------------------------------------------


def test_phase_zero_snvs():
    locus = _make_locus(
        flanking_strings=["AT|CG"], counts=[100],
        up_width=2, dn_width=2,
    )
    panel = _panel("L1", locus)
    out = determine_flanking_haplotypes(panel, find_informative_snvs(panel))
    fh = out["L1"]
    assert fh.snvs == ()
    assert fh.hap_major_profile == ()
    assert fh.hap_minor_profile == ()
    assert fh.anchor_index == -1
    assert fh.max_discordance == 0.0
    assert fh.n_anchor_families == 0
    assert fh.dropped_low_linkage_snvs == ()


def test_phase_single_snv():
    locus = _make_locus(
        flanking_strings=["AT|CG", "CT|CG"], counts=[100, 100],
        up_width=2, dn_width=2,
    )
    panel = _panel("L1", locus)
    out = determine_flanking_haplotypes(panel, find_informative_snvs(panel))
    fh = out["L1"]
    assert len(fh.snvs) == 1
    assert fh.hap_major_profile == (fh.snvs[0].allele_major,)
    assert fh.hap_minor_profile == (fh.snvs[0].allele_minor,)
    assert fh.anchor_index == 0
    assert fh.max_discordance == 0.0
    assert fh.n_anchor_families == fh.snvs[0].coverage
    assert fh.pair_double_coverage == (fh.snvs[0].coverage,)


def test_phase_two_snvs_in_phase():
    # Both SNVs at 70/30. C and A co-segregate; G's co-segregate.
    # Profiles: hap_major=(C,A), hap_minor=(G,G)
    locus = _make_locus(
        flanking_strings=["CA|TT", "GG|TT"],
        counts=[70, 30],
        up_width=2, dn_width=2,
    )
    panel = _panel("L1", locus)
    snvs = find_informative_snvs(panel)
    assert len(snvs["L1"]) == 2
    fh = determine_flanking_haplotypes(panel, snvs)["L1"]
    assert fh.hap_major_profile == ("C", "A")
    assert fh.hap_minor_profile == ("G", "G")
    assert fh.max_discordance == 0.0
    assert fh.dropped_low_linkage_snvs == ()
    # Each non-anchor SNV's pair_double_coverage equals total in-phase reads (100)
    non_anchor = [
        c for i, c in enumerate(fh.pair_double_coverage) if i != fh.anchor_index
    ]
    assert all(c == 100 for c in non_anchor)


def test_phase_two_snvs_out_of_phase_flips():
    # Requires a 4-class joint distribution: per-position majors can be C and A
    # while the dominant cross-cooccurrence is (C,T) + (G,A) — forcing a flip.
    #   row weights: a=(C,T)=30, b=(G,A)=30, c=(C,A)=20, d=(G,T)=10
    #   up[0]: C=a+c=50, G=b+d=40 → major=C, minor=G
    #   up[1]: A=b+c=50, T=a+d=40 → major=A, minor=T
    #   in-phase  (C,A)+(G,T) = c+d = 30
    #   out-phase (C,T)+(G,A) = a+b = 60  ← dominates → flip up[1]
    locus = _make_locus(
        flanking_strings=["CT|GG", "GA|GG", "CA|GG", "GT|GG"],
        counts=[30, 30, 20, 10],
        up_width=2, dn_width=2,
    )
    panel = _panel("L1", locus)
    snvs = find_informative_snvs(panel)
    by_pos = {(s.segment, s.string_index): s for s in snvs["L1"]}
    assert by_pos[("up", 0)].allele_major == "C"
    assert by_pos[("up", 0)].allele_minor == "G"
    assert by_pos[("up", 1)].allele_major == "A"
    assert by_pos[("up", 1)].allele_minor == "T"
    fh = determine_flanking_haplotypes(panel, snvs)["L1"]
    # After flip: hap_major_profile[1] = up[1].minor = T (because anchor=C
    # co-segregates with T more than with A).
    assert fh.hap_major_profile == ("C", "T")
    assert fh.hap_minor_profile == ("G", "A")
    # Discordance: in-phase count = c+d = 30 over total 90 → 0.333...
    assert fh.max_discordance == pytest.approx(30 / 90)


def test_phase_discordance_value():
    # 90 in-phase, 10 cross-phase reads → discordance = 0.1
    # SNV A: up[0] C(major) vs G(minor). SNV B: up[1] A(major) vs T(minor).
    # In-phase: C+A (60), G+T (30) → 90.
    # Cross-phase: C+T (5), G+A (5) → 10.
    locus = _make_locus(
        flanking_strings=["CA|GG", "GT|GG", "CT|GG", "GA|GG"],
        counts=[60, 30, 5, 5],
        up_width=2, dn_width=2,
    )
    panel = _panel("L1", locus)
    snvs = find_informative_snvs(panel)
    fh = determine_flanking_haplotypes(panel, snvs)["L1"]
    # Anchor: up[0] with C=65 G=35; up[1] with A=65 T=35; tied → first.
    assert fh.hap_major_profile == ("C", "A")
    assert fh.hap_minor_profile == ("G", "T")
    assert fh.max_discordance == pytest.approx(0.10)
    # pair_double_coverage = 60 + 30 + 5 + 5 = 100 at the non-anchor SNV
    non_anchor = [
        c for i, c in enumerate(fh.pair_double_coverage) if i != fh.anchor_index
    ]
    assert non_anchor == [100]


def test_phase_low_linkage_snv_dropped():
    # Two SNVs but the (anchor, k) double-coverage falls below the floor passed
    # below → k is dropped. The floor is explicit so the test states the
    # mechanism rather than tracking the cohort-calibrated default.
    #
    # Construct so up[0] and dn[1] both pass the SNV discovery VAF + coverage filter, but
    # only 30 molecules cover both positions (the rest have '.' at one or the
    # other). Every molecule here carries one read, so the counts below read the
    # same in either unit; test_phase_linkage_counts_molecules separates them.
    locus = _make_locus(
        flanking_strings=[
            "CT|G.",  # 80: up[0]=C, dn[1]='.'         (up-only)
            "GT|G.",  # 60: up[0]=G, dn[1]='.'         (up-only)
            ".T|GT",  # 60: up[0]='.', dn[1]=T         (dn-only)
            ".T|GA",  # 40: up[0]='.', dn[1]=A         (dn-only)
            "CT|GT",  # 20: both                       (double-cov 20)
            "GT|GA",  # 10: both                       (double-cov 10)
        ],
        counts=[80, 60, 60, 40, 20, 10],
        up_width=2, dn_width=2,
    )
    panel = _panel("L1", locus)
    snvs = find_informative_snvs(panel)
    by_pos = {(s.segment, s.string_index): s for s in snvs["L1"]}
    # SNV discovery: up[0] C/G cov=170 vaf=70/170≈0.41; dn[1] T/A cov=130 vaf=50/130≈0.385
    assert ("up", 0) in by_pos
    assert ("dn", 1) in by_pos
    # Anchor = up[0] (higher coverage 170 > 130). Double-cov(up[0], dn[1]) = 30.
    fh = determine_flanking_haplotypes(panel, snvs, min_linkage_families=50)["L1"]
    assert len(fh.snvs) == 1
    assert (fh.snvs[0].segment, fh.snvs[0].string_index) == ("up", 0)
    assert len(fh.dropped_low_linkage_snvs) == 1
    assert (
        fh.dropped_low_linkage_snvs[0].segment,
        fh.dropped_low_linkage_snvs[0].string_index,
    ) == ("dn", 1)
    assert fh.hap_major_profile == ("C",)
    assert fh.hap_minor_profile == ("G",)


def test_phase_linkage_counts_molecules_not_reads():
    # 30 molecules carry both positions, each sequenced 5 times: 150 reads.
    # The gate is a count of molecules, so 30 is what it sees and the SNV drops
    # at a floor of 50. Weighting by reads would have passed it three times over.
    rows = (
        [("CA|GG", f"M{i}", 5) for i in range(15)]
        + [("GT|GG", f"M{100 + i}", 5) for i in range(15)]
    )
    locus = _make_locus_from_rows(rows, up_width=2, dn_width=2)
    panel = _panel("L1", locus)
    snvs = find_informative_snvs(panel, min_coverage=25, vaf_min=0.2)
    assert len(snvs["L1"]) == 2                      # up[0] C/G, up[1] A/T
    assert {s.coverage for s in snvs["L1"]} == {30}  # families, not the 150 reads

    fh = determine_flanking_haplotypes(panel, snvs, min_linkage_families=50)["L1"]
    assert len(fh.snvs) == 1
    assert len(fh.dropped_low_linkage_snvs) == 1

    # Same fixture, a floor the molecule count clears: the pair survives and its
    # double coverage is the molecule count.
    fh = determine_flanking_haplotypes(panel, snvs, min_linkage_families=30)["L1"]
    assert len(fh.snvs) == 2
    non_anchor = [
        c for i, c in enumerate(fh.pair_double_coverage) if i != fh.anchor_index
    ]
    assert non_anchor == [30]


def test_phase_linkage_family_votes_once_and_ties_abstain():
    # Three kinds of molecule at the two het positions:
    #   40 clean (C,A) and 40 clean (G,T), one read each;
    #   one molecule read 3 times, 2 saying (C,A) and 1 saying (G,T)  → votes (C,A);
    #   one molecule read twice, splitting 1-1                        → abstains.
    # Molecules: 40 + 40 + 1 = 81 with a call at both positions.
    # Reads:     40 + 40 + 3 + 2 = 85, every one of them an in-phase pair.
    rows = (
        [("CA|GG", f"C{i}", 1) for i in range(40)]
        + [("GT|GG", f"G{i}", 1) for i in range(40)]
        + [("CA|GG", "MAJ", 2), ("GT|GG", "MAJ", 1)]
        + [("CA|GG", "TIE", 1), ("GT|GG", "TIE", 1)]
    )
    locus = _make_locus_from_rows(rows, up_width=2, dn_width=2)
    panel = _panel("L1", locus)
    snvs = find_informative_snvs(panel, min_coverage=25, vaf_min=0.2)
    by_pos = {(s.segment, s.string_index): s for s in snvs["L1"]}
    assert by_pos[("up", 0)].coverage == 81
    assert by_pos[("up", 0)].allele_major == "C"     # 41 vs 40

    fh = determine_flanking_haplotypes(panel, snvs, min_linkage_families=50)["L1"]
    non_anchor = [
        c for i, c in enumerate(fh.pair_double_coverage) if i != fh.anchor_index
    ]
    assert non_anchor == [81]
    assert fh.max_discordance == 0.0
    assert fh.n_anchor_families == 81


def test_phase_anchor_is_highest_coverage():
    # up[0] coverage=100, dn[0] coverage=200. Anchor should be dn[0].
    # up[0]: 60 C, 40 G — major=C, minor=G
    # dn[0]: 120 T, 80 A — major=T, minor=A
    # All 100 up[0]-covered reads also cover dn[0] (T+A=200 total but only some
    # rows have up[0] covered).
    # Build so anchor = dn[0] and up[0] is the non-anchor.
    locus = _make_locus(
        flanking_strings=[
            "CT|TT",  # both up[0]=C and dn[0]=T (in-phase)
            "GA|AA",  # up[0]=G, dn[0]=A (in-phase: G+A)
            "..|TT",  # up[0] '.', dn[0]=T
            "..|AA",  # up[0] '.', dn[0]=A
        ],
        counts=[60, 40, 60, 40],
        up_width=2, dn_width=2,
    )
    panel = _panel("L1", locus)
    snvs = find_informative_snvs(panel)
    # Verify which SNVs we found (need both up[0] C/G and dn[0] T/A)
    by_pos = {(s.segment, s.string_index): s for s in snvs["L1"]}
    assert ("up", 0) in by_pos and ("dn", 0) in by_pos
    assert by_pos[("dn", 0)].coverage > by_pos[("up", 0)].coverage
    fh = determine_flanking_haplotypes(panel, snvs)["L1"]
    # Anchor is the index of the highest-coverage SNV in profile order.
    assert fh.snvs[fh.anchor_index].segment == "dn"
    # n_anchor_families = dn[0]'s coverage
    assert fh.n_anchor_families == by_pos[("dn", 0)].coverage


def test_phase_every_locus_keyed():
    locus_with = _make_locus(
        flanking_strings=["AT|CG", "CT|CG"], counts=[100, 100],
        up_width=2, dn_width=2, bed_name="L1",
    )
    locus_without = _make_locus(
        flanking_strings=["AT|CG"], counts=[100],
        up_width=2, dn_width=2, bed_name="L2",
    )
    panel = ExtractedPanel(
        loci={"L1": locus_with, "L2": locus_without},
        n_alleles=7, provenance=_make_provenance(),
    )
    out = determine_flanking_haplotypes(panel, find_informative_snvs(panel))
    assert set(out.keys()) == {"L1", "L2"}
    assert out["L1"].anchor_index == 0
    assert len(out["L1"].snvs) == 1
    assert out["L2"].anchor_index == -1
    assert out["L2"].snvs == ()


def test_phase_minus_orientation_consistent():
    # Same data as in-phase test but with ref_orientation='-'.
    # Phasing should be unaffected (operates in canonical-orientation space).
    locus = _make_locus(
        flanking_strings=["CA|TT", "GG|TT"],
        counts=[70, 30],
        up_width=2, dn_width=2,
        ref_orientation="-",
    )
    panel = _panel("L1", locus)
    fh = determine_flanking_haplotypes(panel, find_informative_snvs(panel))["L1"]
    assert fh.hap_major_profile == ("C", "A")
    assert fh.hap_minor_profile == ("G", "G")


# ---------------------------------------------------------------------------
# assign_flanking_haplotypes
# ---------------------------------------------------------------------------


def test_assign_zero_snvs_all_unk():
    locus = _make_locus(
        flanking_strings=["AT|CG", "AT|CG"], counts=[100, 50],
        up_width=2, dn_width=2,
    )
    panel = _panel("L1", locus)
    phased = determine_flanking_haplotypes(panel, find_informative_snvs(panel))
    assert phased["L1"].snvs == ()
    out = assign_flanking_haplotypes(panel, phased)["L1"]
    assert out.n_rows == 150
    assert (out.label == LABEL_UNK).all()
    assert (out.n_major == 0).all()
    assert (out.n_minor == 0).all()
    assert (out.n_other == 0).all()


def test_assign_single_snv_calls_by_base():
    # One SNV at up[0]: C=70 families (major), G=30 (minor).
    counts = [70, 30]
    locus = _make_locus(
        flanking_strings=["CT|GG", "GT|GG"], counts=counts,
        up_width=2, dn_width=2,
    )
    panel = _panel("L1", locus)
    phased = determine_flanking_haplotypes(panel, find_informative_snvs(panel))
    out = assign_flanking_haplotypes(panel, phased)["L1"]
    assert out.label.tolist() == _blocks(counts, [LABEL_MAJOR, LABEL_MINOR])
    assert out.n_major.tolist() == _blocks(counts, [1, 0])
    assert out.n_minor.tolist() == _blocks(counts, [0, 1])
    assert (out.n_other == 0).all()


def test_assign_single_snv_uncovered_is_unk():
    # Families with the SNV called (C=70 major, G=30 minor), and 30 with '.'.
    counts = [70, 30, 30]
    locus = _make_locus(
        flanking_strings=["CT|GG", "GT|GG", ".T|GG"], counts=counts,
        up_width=2, dn_width=2,
    )
    panel = _panel("L1", locus)
    phased = determine_flanking_haplotypes(panel, find_informative_snvs(panel))
    out = assign_flanking_haplotypes(panel, phased)["L1"]
    assert out.label.tolist() == _blocks(
        counts, [LABEL_MAJOR, LABEL_MINOR, LABEL_UNK],
    )
    assert out.n_major.tolist() == _blocks(counts, [1, 0, 0])
    assert out.n_minor.tolist() == _blocks(counts, [0, 1, 0])
    assert out.n_other.tolist() == _blocks(counts, [0, 0, 0])


def test_assign_multi_snv_full_coverage():
    # 2 in-phase SNVs: hap_major=(C,A), hap_minor=(G,G).
    # Row 0 matches major at both, row 1 matches minor at both.
    counts = [70, 30]
    locus = _make_locus(
        flanking_strings=["CA|TT", "GG|TT"], counts=counts,
        up_width=2, dn_width=2,
    )
    panel = _panel("L1", locus)
    phased = determine_flanking_haplotypes(panel, find_informative_snvs(panel))
    fh = phased["L1"]
    assert fh.hap_major_profile == ("C", "A")
    assert fh.hap_minor_profile == ("G", "G")
    out = assign_flanking_haplotypes(panel, phased)["L1"]
    assert out.label.tolist() == _blocks(counts, [LABEL_MAJOR, LABEL_MINOR])
    assert out.n_major.tolist() == _blocks(counts, [2, 0])
    assert out.n_minor.tolist() == _blocks(counts, [0, 2])


def test_assign_n_other_third_allele_ignored():
    # 2-SNV locus, hap_major=(C,A), hap_minor=(G,G). Row 2 has a T at up[1]
    # (third allele at that position) but still matches major at up[0]. Should
    # label major with n_other=1.
    counts = [70, 30, 10]
    locus = _make_locus(
        flanking_strings=["CA|TT", "GG|TT", "CT|TT"], counts=counts,
        up_width=2, dn_width=2,
    )
    panel = _panel("L1", locus)
    snvs = find_informative_snvs(panel)
    # Confirm up[1] passed SNV discovery with A/G — the T at row 2 is "other" relative
    # to the kept major/minor for that position.
    by_pos = {(s.segment, s.string_index): s for s in snvs["L1"]}
    assert by_pos[("up", 1)].allele_major == "A"
    assert by_pos[("up", 1)].allele_minor == "G"
    phased = determine_flanking_haplotypes(panel, snvs)
    out = assign_flanking_haplotypes(panel, phased)["L1"]
    # Third string: C at up[0] (major), T at up[1] (other). n_major=1, n_minor=0,
    # n_other=1 → vote rule says major (1 > 0).
    r = _row_of(counts, 2)
    assert out.label[r] == LABEL_MAJOR
    assert out.n_major[r] == 1
    assert out.n_minor[r] == 0
    assert out.n_other[r] == 1


def test_assign_tie_is_unk():
    # 2-SNV locus, hap_major=(C,A), hap_minor=(G,G). Row matches major at
    # up[0] (C) and minor at up[1] (G) → n_major=1, n_minor=1 → unk.
    counts = [70, 30, 50]
    locus = _make_locus(
        flanking_strings=["CA|TT", "GG|TT", "CG|TT"], counts=counts,
        up_width=2, dn_width=2,
    )
    panel = _panel("L1", locus)
    phased = determine_flanking_haplotypes(panel, find_informative_snvs(panel))
    out = assign_flanking_haplotypes(panel, phased)["L1"]
    r = _row_of(counts, 2)
    assert out.label[r] == LABEL_UNK
    assert out.n_major[r] == 1
    assert out.n_minor[r] == 1


def test_assign_min_informative_positions_param():
    # 2-SNV locus, one row covers only up[0] (the second position is '.').
    # Default min=1 → labeled major; min=2 → unk.
    counts = [70, 30, 40]
    locus = _make_locus(
        flanking_strings=["CA|TT", "GG|TT", "C.|TT"], counts=counts,
        up_width=2, dn_width=2,
    )
    panel = _panel("L1", locus)
    phased = determine_flanking_haplotypes(panel, find_informative_snvs(panel))
    default_out = assign_flanking_haplotypes(panel, phased)["L1"]
    strict_out = assign_flanking_haplotypes(
        panel, phased, min_informative_positions=2,
    )["L1"]
    partial = _row_of(counts, 2)
    assert default_out.label[partial] == LABEL_MAJOR
    assert strict_out.label[partial] == LABEL_UNK
    # The other two strings still get major/minor under strict (n_inf=2 ≥ 2)
    assert strict_out.label[_row_of(counts, 0)] == LABEL_MAJOR
    assert strict_out.label[_row_of(counts, 1)] == LABEL_MINOR


def test_assign_vote_margin_param():
    # 3-SNV in-phase locus. Build a row with n_major=2, n_minor=1.
    # vote_margin=0 → major; vote_margin=1 → unk (need diff > 1, have diff=1).
    # SNVs at up[0], up[1], dn[0]; profile major=(C,A,T), minor=(G,G,A).
    # Third string: C, A, A → n_major=2 (up[0],up[1]), n_minor=1 (dn[0]).
    counts = [100, 80, 30]
    locus = _make_locus(
        flanking_strings=["CA|TG", "GG|AG", "CA|AG"], counts=counts,
        up_width=2, dn_width=2,
    )
    panel = _panel("L1", locus)
    snvs = find_informative_snvs(panel)
    fh = determine_flanking_haplotypes(panel, snvs)["L1"]
    # Verify the locus phases as expected before checking the assignment.
    assert fh.hap_major_profile == ("C", "A", "T")
    assert fh.hap_minor_profile == ("G", "G", "A")
    phased = {"L1": fh}
    default_out = assign_flanking_haplotypes(panel, phased)["L1"]
    strict_out = assign_flanking_haplotypes(panel, phased, vote_margin=1)["L1"]
    r = _row_of(counts, 2)
    assert default_out.n_major[r] == 2
    assert default_out.n_minor[r] == 1
    assert default_out.label[r] == LABEL_MAJOR
    assert strict_out.label[r] == LABEL_UNK


def test_assign_every_locus_keyed():
    counts = [70, 30]
    locus_with = _make_locus(
        flanking_strings=["CT|GG", "GT|GG"], counts=counts,
        up_width=2, dn_width=2, bed_name="L1",
    )
    locus_without = _make_locus(
        flanking_strings=["AT|CG"], counts=[100],
        up_width=2, dn_width=2, bed_name="L2",
    )
    panel = ExtractedPanel(
        loci={"L1": locus_with, "L2": locus_without},
        n_alleles=7, provenance=_make_provenance(),
    )
    phased = determine_flanking_haplotypes(panel, find_informative_snvs(panel))
    out = assign_flanking_haplotypes(panel, phased)
    assert set(out.keys()) == {"L1", "L2"}
    assert out["L1"].label.tolist() == _blocks(counts, [LABEL_MAJOR, LABEL_MINOR])
    assert out["L2"].n_rows == 100
    assert (out["L2"].label == LABEL_UNK).all()


def test_assign_locus_missing_from_phased_is_all_unk():
    # If a locus is in panel.loci but absent from the phased dict (which
    # shouldn't happen via determine_flanking_haplotypes, but defensive
    # check), it should emit an all-unk assignment.
    locus = _make_locus(
        flanking_strings=["CT|GG", "GT|GG"], counts=[70, 30],
        up_width=2, dn_width=2,
    )
    panel = _panel("L1", locus)
    out = assign_flanking_haplotypes(panel, phased={})["L1"]
    assert out.n_rows == 100
    assert (out.label == LABEL_UNK).all()


def test_assign_array_dtypes_are_int8():
    locus = _make_locus(
        flanking_strings=["CT|GG", "GT|GG"], counts=[70, 30],
        up_width=2, dn_width=2,
    )
    panel = _panel("L1", locus)
    phased = determine_flanking_haplotypes(panel, find_informative_snvs(panel))
    out = assign_flanking_haplotypes(panel, phased)["L1"]
    assert out.label.dtype == np.int8
    assert out.n_major.dtype == np.int8
    assert out.n_minor.dtype == np.int8
    assert out.n_other.dtype == np.int8


# ---------------------------------------------------------------------------
# PhasingPanel
# ---------------------------------------------------------------------------


def test_phasing_panel_from_panel_runs_full_pipeline():
    counts = [70, 30]
    locus = _make_locus(
        flanking_strings=["CA|TT", "GG|TT"], counts=counts,
        up_width=2, dn_width=2,
    )
    panel = _panel("L1", locus)
    pp = PhasingPanel.from_panel(panel)
    assert set(pp.snvs.keys()) == {"L1"}
    assert set(pp.haplotypes.keys()) == {"L1"}
    assert set(pp.assignments.keys()) == {"L1"}
    assert pp.haplotypes["L1"].hap_major_profile == ("C", "A")
    assert pp.assignments["L1"].label.tolist() == _blocks(
        counts, [LABEL_MAJOR, LABEL_MINOR],
    )


def test_phasing_panel_save_load_roundtrip(tmp_path):
    locus = _make_locus(
        flanking_strings=["CA|TT", "GG|TT"], counts=[70, 30],
        up_width=2, dn_width=2,
    )
    panel = _panel("L1", locus)
    pp = PhasingPanel.from_panel(panel)
    out_path = tmp_path / "phasing.pkl"
    pp.save_pickle(out_path)
    loaded = PhasingPanel.load_pickle(out_path)
    assert loaded.haplotypes["L1"].hap_major_profile == ("C", "A")
    assert (
        loaded.assignments["L1"].label == pp.assignments["L1"].label
    ).all()
