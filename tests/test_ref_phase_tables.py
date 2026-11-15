"""ref-phase writes the phase tables for the target panel, and the viewer reads them."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

import prolyg_phasing.cli as cli
import prolyg_phasing.locus_view as lv
from prolyg_phasing.io.panel import ExtractedPanel
from prolyg_phasing.io.phasing_tables import haplotypes_from_snv_calls_table
from prolyg_phasing.phasing import LABEL_MAJOR, LABEL_MINOR
from prolyg_phasing.ref_phasing import assign_flanking_haplotypes_ref
from tests.test_phasing_tables import _make_locus, _make_provenance

# Reference: one SNV at upstream index 4 (C in 70 families, G in 30).
REFERENCE = (["AAAACAAT|TTTTTTTT", "AAAAGAAT|TTTTTTTT"], [70, 30])
# Target: the upstream window starts 2 bases farther out, so the SNV sits at
# index 6; 25 C, 15 G and 5 families with no base there.
TARGET = (["TTAAAACAAT|TTTTTTTT", "TTAAAAGAAT|TTTTTTTT", "TTAAAA.AAT|TTTTTTTT"], [25, 15, 5])


def _panel(path, strings, counts, up_width, up_start, mi_prefix):
    n = sum(counts)
    locus = _make_locus(flanking_strings=strings, counts=counts, seqs=["GGGGGG"] * n,
                        up_width=up_width, dn_width=8)
    locus.flanking_up_ref_pos_start = up_start
    locus.mi = np.asarray([f"{mi_prefix}{i}" for i in range(n)], dtype=object)
    ExtractedPanel(loci={"L1": locus}, n_alleles=8, provenance=_make_provenance()).save_db(path)
    return path


def _run(tmp_path):
    ref_db = _panel(tmp_path / "ref.panel.db", *REFERENCE, 8, 1000, "R")
    target_db = _panel(tmp_path / "target.panel.db", *TARGET, 10, 998, "T")
    ref_out, out = tmp_path / "ref_phasing", tmp_path / "ref_phase"
    cli.main(["phase", "--panel-db", str(ref_db), "--out-dir", str(ref_out)])
    cli.main(["ref-phase", "--target-panel-db", str(target_db),
              "--reference-snv-calls", str(ref_out / "snv_calls.tsv"),
              "--target-sample-id", "T", "--reference-sample-id", "R", "--out-dir", str(out)])
    return target_db, ref_out, out


def test_ref_phase_writes_the_phase_tables_for_the_target(tmp_path):
    target_db, ref_out, out = _run(tmp_path)
    target = ExtractedPanel.load_db(target_db)
    locus = target.loci["L1"]
    anchored = assign_flanking_haplotypes_ref(
        target, haplotypes_from_snv_calls_table(pd.read_csv(ref_out / "snv_calls.tsv",
                                                            sep="\t")),
        target_sample_id="T", reference_sample_id="R")

    calls = pd.read_csv(out / "haplotype_calls.tsv", sep="\t", dtype=str)
    reference_calls = pd.read_csv(ref_out / "haplotype_calls.tsv", sep="\t", dtype=str)
    assert list(calls.columns) == list(reference_calls.columns)
    assert list(calls.mi) == [str(m) for m in locus.mi]
    names = {LABEL_MAJOR: "hap_major", LABEL_MINOR: "hap_minor"}
    assert list(calls.phase_label) == \
        [names.get(int(x), "unk") for x in anchored.assignments["L1"].label]

    snvs = pd.read_csv(out / "snv_calls.tsv", sep="\t", keep_default_na=False)
    reference_snvs = pd.read_csv(ref_out / "snv_calls.tsv", sep="\t", keep_default_na=False)
    assert list(snvs.columns) == list(reference_snvs.columns)
    (row,) = snvs.to_dict("records")
    (ref_row,) = reference_snvs.to_dict("records")
    # The target's own position, the reference's alleles and haplotype bases.
    assert (row["segment"], row["string_index"], row["ref_pos"]) == ("up", 6, 1004)
    for field in ("allele_major", "allele_minor", "hap_major_base", "hap_minor_base",
                  "is_anchor"):
        assert row[field] == ref_row[field]
    # The target's own family counts: 25 C and 15 G families cover it.
    bases = Counter(s[6] for s, n in zip(*TARGET, strict=True) for _ in range(n))
    assert row["coverage"] == bases["C"] + bases["G"] == 40
    assert row["vaf_minor"] == bases[ref_row["allele_minor"]] / 40
    assert (row["pair_discordance"], row["pair_double_coverage"]) == ("", "")

    summary = pd.read_csv(out / "locus_summary.tsv", sep="\t", keep_default_na=False)
    reference_summary = pd.read_csv(ref_out / "locus_summary.tsv", sep="\t")
    assert list(summary.columns) == list(reference_summary.columns)
    (s,) = summary.to_dict("records")
    assert (s["hap_major_profile"], s["hap_minor_profile"], s["n_snvs_phased"]) == ("C", "G", 1)
    assert (s["reads_major"], s["reads_minor"], s["reads_unk"]) == (25, 15, 5)


def test_the_viewer_loads_a_ref_phase_directory(tmp_path):
    target_db, _, out = _run(tmp_path)
    server = lv.make_server([(target_db, out)], 0)  # runs the stale check
    server.server_close()
    phasing = lv.PhasingIndex.read(out)
    view = lv.locus_view_data("L1", ExtractedPanel.load_db(target_db).loci["L1"], "T",
                              phasing=phasing)
    assert [c["index"] for c in view["pileup"]["up"] if "snv" in c] == [6]
    assert [s["label"] for s in view["pileup"]["sections"]] == ["hap_major", "hap_minor", "unk"]
    assert [s["reads"] for s in view["pileup"]["sections"]] == [25, 15, 5]
