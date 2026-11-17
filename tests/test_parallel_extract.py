"""Extraction in worker processes, written to panel.db one locus at a time."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

import prolyg_phasing.cli as cli
from prolyg_phasing.io.bam import extract_panel, extract_panel_to_db
from prolyg_phasing.io.panel import ExtractedPanel
from tests.test_extract_hand_built import MOVED, _write_case


def _case_files(tmp_path):
    """PolyG5177 on a "+" and a "-" contig; the "-" locus has no flank pair."""
    bam, bed, flanks = _write_case(tmp_path, MOVED)
    table = pd.read_csv(flanks, sep="\t", dtype=str, keep_default_na=False)
    minus = table.locus == "PolyG5177-"
    table.loc[minus, "reference_window_status"] = "no_flank_pair"
    table.loc[minus, ["flank_bases", "flank_ref_positions"]] = ""
    table.to_csv(flanks, sep="\t", index=False)
    return bam, bed, flanks


def _assert_same_panel(a: ExtractedPanel, b: ExtractedPanel) -> None:
    assert a.n_alleles == b.n_alleles
    assert a.provenance == b.provenance
    assert sorted(a.loci) == sorted(b.loci)
    for name in a.loci:
        x, y = a.loci[name], b.loci[name]
        for field in dataclasses.fields(x):
            u, v = getattr(x, field.name), getattr(y, field.name)
            if isinstance(u, np.ndarray):
                assert u.dtype == v.dtype and np.array_equal(u, v), (name, field.name)
            else:
                assert u == v, (name, field.name)


@pytest.mark.parametrize("n_workers", [1, 3])
def test_the_parallel_panel_equals_the_single_process_panel(tmp_path, n_workers):
    bam, bed, flanks = _case_files(tmp_path)
    single = tmp_path / "single.db"
    extract_panel(bam, bed, flanks).save_db(single)
    parallel = tmp_path / f"parallel_{n_workers}.db"
    assert extract_panel_to_db(bam, bed, flanks, parallel, n_workers=n_workers) == 2
    a, b = ExtractedPanel.load_db(single), ExtractedPanel.load_db(parallel)
    _assert_same_panel(a, b)
    assert b.loci["PolyG5177-"].reference_window_status == "no_flank_pair"
    assert b.loci["PolyG5177+"].n_reads > 0


def test_a_failed_extraction_leaves_no_panel(tmp_path):
    bam, bed, flanks = _case_files(tmp_path)
    out = tmp_path / "panel.db"
    with pytest.raises(FileNotFoundError):
        extract_panel_to_db(tmp_path / "missing.bam", bed, flanks, out, n_workers=2)
    assert not out.exists() and not out.with_name("panel.db.tmp").exists()


def test_the_extract_cli_takes_a_worker_count(tmp_path, monkeypatch):
    bam, bed, flanks = _case_files(tmp_path)
    out = tmp_path / "panel.db"
    monkeypatch.setattr(cli, "worker_count", lambda: 1)
    base = ["extract", "--bam", str(bam), "--bed", str(bed), "--flanks", str(flanks),
            "--out-db", str(out)]
    assert cli.build_parser().parse_args(base).workers is None
    assert cli.build_parser().parse_args([*base, "--workers", "2"]).workers == 2
    cli.main([*base, "--workers", "2"])
    _assert_same_panel(ExtractedPanel.load_db(out),
                       _single(tmp_path, bam, bed, flanks))


def _single(tmp_path, bam, bed, flanks):
    path = tmp_path / "single.db"
    extract_panel(bam, bed, flanks).save_db(path)
    return ExtractedPanel.load_db(path)
