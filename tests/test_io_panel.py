"""Tests for prolyg_phasing.io.panel — the read/family haplotype-labeling boundary.

Covers synthetic ExtractedPanel construction, the ``panel.db`` round-trip, and
every pure panel-shape helper (``interrupter_pattern``, pattern-frequency
screens). ``format_drop_report``/``format_panel_composition`` stay in prolyG
(``inference/to_loci_report.py``) — they format ``to_loci()``'s fit-machinery
output (torch ``LocusData`` objects), not anything this package produces; see
``tests/test_to_loci_report.py`` in prolyG. This file has no dependency on
``prolyG.inference.pcr_params``/``train`` either way — the ``to_loci()``
adapter into that fit machinery has its own coverage in prolyG's
``test_panel_to_loci.py``. pysam-free: BAM-walking tests live in
``test_io_bam.py``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from prolyg_phasing.io.panel import (
    ExtractedPanel,
    interrupter_pattern,
    loci_equal,
    majority_pattern_per_rf,
    observed_pattern_frequencies,
    panels_equal,
    parse_run_lengths,
    select_patterns_above_freq,
)

from ._panel_fixtures import _make_locus_single_run, _make_provenance

# ---------------------------------------------------------------------------
# parse_run_lengths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seq, expected",
    [
        ("GGGGGG", (6,)),                   # single-run
        ("GGGAGGG", (3, 3)),                # canonical interrupter
        ("GGGTGGG", (3, 3)),                # SNV in interrupter — same tuple
        ("GGGNGGG", (3, 3)),                # N as interrupter — same tuple
        ("GGGATGGG", (3, 3)),               # multi-base interrupter
        ("GGGAGGGCGG", (3, 3, 2)),          # S=3
        ("GGGGGGGGGGGGG", (13,)),           # interrupter deletion → S=1 (alt seq vs S=2 ref)
        ("", (0,)),                         # empty inter-anchor (anchors adjacent)
        ("AGGG", (0, 3)),                   # zero-length first run
        ("GGGA", (3, 0)),                   # zero-length last run
    ],
)
def test_parse_run_lengths(seq, expected):
    assert parse_run_lengths(seq) == expected


# ---------------------------------------------------------------------------
# SQLite (streaming) — save_db / load_db
# ---------------------------------------------------------------------------


def _two_locus_panel_with_flanking() -> ExtractedPanel:
    """A two-locus panel with flanking + multi-run fields set, for db tests."""
    locus1 = _make_locus_single_run(
        [("10004", "A", "GGGGGGG", 2), ("10004", "B", "GGGGGGG", 3),
         ("10007", "A", "GGGGGG", 1)],
        bed_name="Locus1",
    )
    locus1.flanking_id = np.asarray([0, 1, 2], dtype=np.int32)
    locus1.flanking_seq = np.asarray(["ACGT" * 5, "TGCA" * 5, "GATC" * 5], dtype=object)
    locus1.g_walk_up, locus1.g_walk_dn = 3, 2
    locus1.flanking_up_width = locus1.flanking_dn_width = 20
    locus1.flanking_up_ref_pos_start, locus1.flanking_dn_ref_pos_start = 90, 110

    locus2 = _make_locus_single_run(
        [("20001", "A", "GGGAGGG", 4)],
        n_runs=2, ref_inter_anchor_seq="GGGAGGG", reference_run_lengths=(3, 3),
        bed_name="Locus2", chrom="chr2", bed_start=200, bed_end=207,
    )
    return ExtractedPanel(
        loci={"Locus1": locus1, "Locus2": locus2},
        n_alleles=11,
        provenance=_make_provenance(),
    )


def test_save_load_db_roundtrip(tmp_path):
    panel = _two_locus_panel_with_flanking()
    db = tmp_path / "panel.db"
    panel.save_db(db)
    loaded = ExtractedPanel.load_db(db)

    assert loaded.n_alleles == panel.n_alleles
    assert loaded.provenance.bam_path == panel.provenance.bam_path
    assert loaded.provenance.min_mapq == panel.provenance.min_mapq
    assert set(loaded.loci.keys()) == {"Locus1", "Locus2"}

    rec = loaded.loci["Locus1"]
    np.testing.assert_array_equal(rec.mi, panel.loci["Locus1"].mi)
    np.testing.assert_array_equal(rec.strand, panel.loci["Locus1"].strand)
    np.testing.assert_array_equal(rec.seq, panel.loci["Locus1"].seq)
    np.testing.assert_array_equal(rec.count, panel.loci["Locus1"].count)
    np.testing.assert_array_equal(rec.flanking_seq, panel.loci["Locus1"].flanking_seq)
    assert rec.g_walk_up == 3
    assert rec.flanking_up_ref_pos_start == 90

    rec2 = loaded.loci["Locus2"]
    assert rec2.n_runs == 2
    assert rec2.reference_run_lengths == (3, 3)
    assert rec2.chrom == "chr2"


def test_load_db_loci_is_lazy_non_dict_mapping(tmp_path):
    panel = _two_locus_panel_with_flanking()
    db = tmp_path / "panel.db"
    panel.save_db(db)
    loaded = ExtractedPanel.load_db(db)

    assert not isinstance(loaded.loci, dict)
    assert isinstance(loaded.loci, Mapping)
    assert len(loaded.loci) == 2
    assert "Locus1" in loaded.loci
    assert "NoSuchLocus" not in loaded.loci
    with pytest.raises(KeyError):
        loaded.loci["NoSuchLocus"]
    # Repeated access re-decodes rather than erroring or aliasing.
    assert loaded.loci["Locus1"] is not loaded.loci["Locus1"]
    np.testing.assert_array_equal(
        loaded.loci["Locus1"].count, loaded.loci["Locus1"].count,
    )


def test_save_db_roundtrip_matches_source_via_panels_equal(tmp_path):
    """The format-equivalence oracle: save_db + load_db recovers the source panel exactly."""
    panel = _two_locus_panel_with_flanking()
    db = tmp_path / "panel.db"
    panel.save_db(db)

    db_loaded = ExtractedPanel.load_db(db)
    assert panels_equal(panel, db_loaded) == {}


def test_save_db_empty_panel(tmp_path):
    panel = ExtractedPanel(loci={}, n_alleles=8, provenance=_make_provenance())
    db = tmp_path / "panel.db"
    panel.save_db(db)
    loaded = ExtractedPanel.load_db(db)
    assert loaded.n_alleles == 8
    assert len(loaded.loci) == 0
    assert list(loaded.loci) == []


def test_loci_equal_detects_array_mismatch():
    locus_a = _make_locus_single_run([("10004", "A", "GGGGGGG", 2)])
    locus_b = _make_locus_single_run([("10004", "A", "GGGGGGG", 3)])  # count differs
    mismatches = loci_equal(locus_a, locus_b)
    assert "count" in mismatches


def test_loci_equal_no_mismatch_for_identical_loci():
    locus_a = _make_locus_single_run([("10004", "A", "GGGGGGG", 2)])
    locus_b = _make_locus_single_run([("10004", "A", "GGGGGGG", 2)])
    assert loci_equal(locus_a, locus_b) == []


def test_panels_equal_raises_on_locus_id_set_mismatch():
    locus = _make_locus_single_run([("10004", "A", "GGGGGGG", 2)])
    panel_a = ExtractedPanel(loci={"Locus1": locus}, n_alleles=8, provenance=_make_provenance())
    panel_b = ExtractedPanel(loci={"Locus2": locus}, n_alleles=8, provenance=_make_provenance())
    with pytest.raises(ValueError, match="locus_id sets differ"):
        panels_equal(panel_a, panel_b)


# ---------------------------------------------------------------------------
# ExtractedLocus.to_dataframe
# ---------------------------------------------------------------------------


def test_to_dataframe_view():
    locus = _make_locus_single_run([
        ("10004", "A", "GGGGG", 3),
        ("10007", "B", "GGGGGG", 2),
    ])
    df = locus.to_dataframe()
    assert list(df.columns) == ["mi", "strand", "seq", "count"]
    assert len(df) == 2
    assert int(df["count"].sum()) == 5


# ---------------------------------------------------------------------------
# interrupter_pattern
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seq, expected",
    [
        ("GGGGGG", ()),                    # all G, no interrupters
        ("", ()),                          # empty seq
        ("GGGAGGG", ("A",)),
        ("GGGTGGG", ("T",)),               # SNV — different pattern from canonical
        ("GGGATGGG", ("AT",)),             # multi-base interrupter
        ("GGGAGGGCGG", ("A", "C")),        # two interrupters
        ("AGGG", ("A",)),                  # zero-length first G-run
        ("GGGA", ("A",)),                  # zero-length last G-run
        ("GGGNGGG", ("N",)),               # N as interrupter
    ],
)
def test_interrupter_pattern(seq, expected):
    assert interrupter_pattern(seq) == expected


# ---------------------------------------------------------------------------
# observed_pattern_frequencies / majority_pattern_per_rf / select_patterns_above_freq
# ---------------------------------------------------------------------------


def test_observed_pattern_frequencies_single_pattern():
    """Locus with one canonical pattern → that pattern at frequency 1.0."""
    locus = _make_locus_single_run([
        ("10004", "A", "GGGAGGG", 5),
        ("10004", "B", "GGGAGGG", 3),
        ("10007", "A", "GGGAGGG", 2),
    ], n_runs=2, ref_inter_anchor_seq="GGGAGGG", reference_run_lengths=(3, 3))
    freqs = observed_pattern_frequencies(locus)
    assert freqs == {("A",): 1.0}


def test_observed_pattern_frequencies_two_patterns_per_mi():
    """Two MIs each carrying a distinct pattern → 0.5 / 0.5."""
    locus = _make_locus_single_run([
        ("10004", "A", "GGGAGGG", 5),  # MI 10004 → pattern A
        ("10007", "A", "GGGTGGG", 5),  # MI 10007 → pattern T
    ], n_runs=2, ref_inter_anchor_seq="GGGAGGG", reference_run_lengths=(3, 3))
    freqs = observed_pattern_frequencies(locus)
    assert freqs == {("A",): 0.5, ("T",): 0.5}


def test_observed_pattern_frequencies_per_mi_majority_resolves_within_mi_mix():
    """One MI with 4 reads at pattern A and 1 read at pattern T → MI counts as A only."""
    locus = _make_locus_single_run([
        ("10004", "A", "GGGAGGG", 4),  # majority pattern A
        ("10004", "B", "GGGTGGG", 1),  # minor pattern T within same MI
        ("10007", "A", "GGGTGGG", 3),  # MI 10007 is pure T
    ], n_runs=2, ref_inter_anchor_seq="GGGAGGG", reference_run_lengths=(3, 3))
    freqs = observed_pattern_frequencies(locus)
    # 1 MI assigned to A, 1 to T → each is 0.5 of n_rfs (2 MIs total).
    assert freqs == {("A",): 0.5, ("T",): 0.5}


def test_observed_pattern_frequencies_robust_to_amplification_bias():
    """Per-MI counting beats per-read: one over-amplified noisy MI cannot dominate."""
    locus = _make_locus_single_run([
        # MI 10004 carries pattern A but has 100 amplified reads (PCR bias)
        ("10004", "A", "GGGAGGG", 100),
        # 9 other MIs each carry the same pattern A with normal coverage
        *[(f"1000{i}", "A", "GGGAGGG", 1) for i in range(5, 14)],
        # MI 99999 is the noisy outlier with pattern T, only 1 read
        ("99999", "A", "GGGTGGG", 1),
    ], n_runs=2, ref_inter_anchor_seq="GGGAGGG", reference_run_lengths=(3, 3))
    freqs = observed_pattern_frequencies(locus)
    # 10 MIs at A, 1 MI at T; despite the 100:1 read ratio, T is still 1/11.
    assert freqs[("A",)] == pytest.approx(10 / 11)
    assert freqs[("T",)] == pytest.approx(1 / 11)


def test_observed_pattern_frequencies_empty_locus():
    locus = _make_locus_single_run([], n_runs=1)
    assert observed_pattern_frequencies(locus) == {}


def test_majority_pattern_per_rf():
    """One MI per (mi) majority — both strands aggregated."""
    locus = _make_locus_single_run([
        ("10004", "A", "GGGAGGG", 3),
        ("10004", "B", "GGGAGGG", 2),
        ("10007", "A", "GGGTGGG", 1),
    ], n_runs=2, ref_inter_anchor_seq="GGGAGGG", reference_run_lengths=(3, 3))
    maj = majority_pattern_per_rf(locus)
    assert maj == {"10004": ("A",), "10007": ("T",)}


def test_select_patterns_above_freq():
    locus = _make_locus_single_run([
        # 9 MIs at "A" (90% of MIs), 1 MI at "T" (10%)
        *[(f"1000{i}", "A", "GGGAGGG", 1) for i in range(9)],
        ("99999", "A", "GGGTGGG", 1),
    ], n_runs=2, ref_inter_anchor_seq="GGGAGGG", reference_run_lengths=(3, 3))
    # 0.0 keeps all
    assert select_patterns_above_freq(locus, min_freq=0.0) == {("A",), ("T",)}
    # 0.05 keeps both (T is 10%)
    assert select_patterns_above_freq(locus, min_freq=0.05) == {("A",), ("T",)}
    # 0.15 keeps only A (T at 10% drops)
    assert select_patterns_above_freq(locus, min_freq=0.15) == {("A",)}


# ---------------------------------------------------------------------------
# ExtractedLocus.pattern_breakdown
# ---------------------------------------------------------------------------


def test_pattern_breakdown_columns_and_values():
    locus = _make_locus_single_run([
        ("10004", "A", "GGGAGGG", 3),      # parsed (3,3) tuple, MI majority "A"
        ("10004", "B", "GGGAGGG", 2),
        ("10007", "A", "GGGGGGGGGGG", 1),  # fused S=1 → tuple None for S=2 locus
    ], n_runs=2, ref_inter_anchor_seq="GGGAGGG", reference_run_lengths=(3, 3))
    df = locus.pattern_breakdown()
    assert list(df.columns) == [
        "mi", "strand", "seq", "count", "pattern", "tuple", "tuple_str",
    ]
    assert len(df) == 3
    # MI 10004 rows: pattern "A", tuple (3, 3).
    mi_10004 = df[df["mi"] == "10004"]
    assert set(mi_10004["pattern"]) == {"A"}
    assert all(t == (3, 3) for t in mi_10004["tuple"])
    assert all(s == "(3, 3)" for s in mi_10004["tuple_str"])
    # MI 10007 row: parse failed (1 run, locus n_runs=2) → tuple None, tuple_str "—".
    mi_10007 = df[df["mi"] == "10007"]
    assert mi_10007["tuple"].iloc[0] is None
    assert mi_10007["tuple_str"].iloc[0] == "—"
    # MI 10007 majority is the empty pattern () → "-", the shared renderer's token
    # for "this allele has no interrupters". It used to render "", which in a table
    # column is indistinguishable from missing data -- and this row's neighbor is
    # genuinely missing ("—"), so the two were one column apart and told apart by
    # nothing. "-" is also what the published gexp_calls.tsv has always used.
    assert mi_10007["pattern"].iloc[0] == "-"


# ---------------------------------------------------------------------------
# Flanking schema-v2: backward compatibility with v1 artifacts
# ---------------------------------------------------------------------------


def test_v1_pickle_without_flanking_fields_round_trips(tmp_path):
    """An ExtractedLocus instantiated without flanking fields pickles + reloads cleanly."""
    import pickle as _pickle

    locus = _make_locus_single_run([("10004", "A", "GGGGGGG", 2)])
    # Sanity: defaults populated.
    assert len(locus.flanking_id) == 0
    assert len(locus.flanking_seq) == 0
    panel = ExtractedPanel(
        loci={"Locus1": locus},
        n_alleles=8,
        provenance=_make_provenance(),
    )
    pkl = tmp_path / "panel.pkl"
    with pkl.open("wb") as f:
        _pickle.dump(panel, f)
    with pkl.open("rb") as f:
        loaded = _pickle.load(f)
    rec = loaded.loci["Locus1"]
    assert len(rec.flanking_id) == 0
    assert rec.g_walk_up == 0
