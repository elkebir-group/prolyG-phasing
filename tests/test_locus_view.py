"""The locus viewer's data, read from a panel built from hand-built reads."""

from __future__ import annotations

import json
import re
from collections import Counter

import pandas as pd
import pytest

import prolyg_phasing.cli as cli
import prolyg_phasing.locus_view as lv
from prolyg_phasing.io._flank_placement import FlankRecord
from prolyg_phasing.io.bam import extract_panel
from prolyg_phasing.io.panel import ExtractedPanel
from prolyg_phasing.locus_view import locus_blocks, locus_view_data
from tests.test_extract_hand_built import INVARIANT, MOVED, _write_case
from tests.test_flank_placement import flank_record

ORIENTATIONS = ("+", "-")


def _panel(tmp_path, case, *, records=None, edit=None):
    """A panel.db of `case` on a "+" and a "-" contig, loaded back by its own loader."""
    bam, bed, flanks = _write_case(tmp_path, case, flank_records=records)
    if edit is not None:
        table = pd.read_csv(flanks, sep="\t", dtype=str, keep_default_na=False)
        edit(table)
        table.to_csv(flanks, sep="\t", index=False)
    db = tmp_path / "panel.db"
    extract_panel(bam, bed, flanks).save_db(db)
    return db, ExtractedPanel.load_db(db)


def _panel_rows(locus):
    """(tract, upstream flanking, downstream flanking, MI, reads) of each panel row."""
    for mi, seq, fid, count in zip(locus.mi, locus.seq, locus.flanking_id, locus.count,
                                   strict=True):
        up, dn = str(locus.flanking_seq[int(fid)]).split("|")
        yield str(seq), up, dn, str(mi), int(count)


@pytest.mark.parametrize("case", [MOVED, INVARIANT], ids=["moved", "invariant"])
def test_the_molecules_are_the_panel_rows(tmp_path, case):
    _, panel = _panel(tmp_path, case)
    for o in ORIENTATIONS:
        name = f"{case.locus.name}{o}"
        locus = panel.loci[name]
        view = locus_view_data(name, locus, "S")
        rows = list(_panel_rows(locus))
        assert rows
        mols = view["molecules"]
        assert sum(m["reads"] for m in mols) == locus.n_reads == view["counters"]["n_reads"]
        keys = {(t, u, d) for t, u, d, _, _ in rows}
        assert {(m["tract"], m["up_flanking"], m["dn_flanking"]) for m in mols} == keys
        assert len(mols) == len(keys)
        for m in mols:
            mine = [r for r in rows if r[:3] == (m["tract"], m["up_flanking"], m["dn_flanking"])]
            assert m["reads"] == sum(r[4] for r in mine)
            assert m["molecules"] == len({r[3] for r in mine})
        assert [m["reads"] for m in mols] == sorted((m["reads"] for m in mols), reverse=True)
        # Each up flanking string spans the locus's upstream field.
        assert {len(m["up_flanking"]) for m in mols} == {locus.flanking_up_width}
        assert sum(v for _, v in view["tract_length_counts"]) == locus.n_reads
        assert sum(v for _, v in view["largest_run_counts"]) == locus.n_reads
        assert view["flanks"]["upstream"]["ref_positions"] == \
            list(locus.upstream_flank.ref_positions)
        counters = view["counters"]
        assert {"n_alignments_overlap_locus", "n_alignments_drop_no_mi",
                "n_alignments_drop_no_bed_base", "n_read_pairs",
                "n_read_pairs_merged"} <= set(counters)
        assert all(counters[k] == getattr(locus, k) for k in counters)


def test_the_moved_case_has_its_known_tract(tmp_path):
    """PolyG5177 with a G at both edge bases: the reference tract between
    the moved flanks is GGGT G11 A (see tests/test_extract_hand_built.py)."""
    _, panel = _panel(tmp_path, MOVED)
    view = locus_view_data("PolyG5177+", panel.loci["PolyG5177+"], "S")
    assert view["reference_tract"] == "GGGT" + "G" * 11 + "A"
    assert view["reference_run_lengths"] == [3, 11, 0]


def test_a_locus_with_no_reference_window_shows_its_status_only(tmp_path):
    def no_window(table):
        plus = table.locus == "PolyG151+"
        table.loc[plus, "reference_window_status"] = "ref_window_unrecoverable"
        table.loc[plus, ["flank_bases", "flank_ref_positions"]] = ""

    _, panel = _panel(tmp_path, INVARIANT, edit=no_window)
    view = locus_view_data("PolyG151+", panel.loci["PolyG151+"], "S")
    assert view["reference_window_status"] == "ref_window_unrecoverable"
    assert view["flanks"] is None and view["molecules"] == []
    assert view["counters"]["n_reads"] == 0
    assert view["flanking"]["up_ref_pos_start"] == view["flanking"]["dn_ref_pos_start"] == -1


def test_a_side_with_no_position(tmp_path):
    case = INVARIANT
    records = {f"PolyG151{o}": (flank_record(case, case.up, o),
                                FlankRecord(case.dn.bases, (), 0)) for o in ORIENTATIONS}
    _, panel = _panel(tmp_path, case, records=records)
    view = locus_view_data("PolyG151+", panel.loci["PolyG151+"], "S")
    dn = view["flanks"]["downstream"]
    assert (dn["status"], dn["ref_positions"], dn["run_side_position"]) == \
        ("no_reference_position", [], None)
    assert view["flanks"]["upstream"]["status"] == "recovered"
    assert view["flanking"]["dn_width"] == 0 and view["flanking"]["dn_ref_pos_start"] == -1
    assert {m["dn_flanking"] for m in view["molecules"]} == {""}
    assert sum(m["reads"] for m in view["molecules"]) == view["counters"]["n_reads"] > 0


def test_locus_blocks_label_each_panel_and_name_a_missing_locus(tmp_path):
    db, _ = _panel(tmp_path, INVARIANT)
    panels = [(str(db), ExtractedPanel.load_db(db))] * 2
    blocks = locus_blocks(panels, "PolyG151-")
    assert [b["sample"] for b in blocks] == ["sample", "sample"]
    with pytest.raises(KeyError, match="holds no locus PolyG9"):
        locus_blocks(panels, "PolyG9")


def test_the_rows_past_the_cap_are_counted(tmp_path, monkeypatch):
    """With a cap of 1 row, the other rows and their reads are counted, and the
    kept row is the one with the most reads."""
    _, panel = _panel(tmp_path, MOVED)
    locus = panel.loci["PolyG5177+"]
    full = locus_view_data("PolyG5177+", locus, "S")
    assert full["n_rows"] == len(full["molecules"]) >= 2
    assert (full["rows_not_shown"], full["reads_not_shown"]) == (0, 0)
    monkeypatch.setattr(lv, "ROWS_SHOWN", 1)
    capped = locus_view_data("PolyG5177+", locus, "S")
    assert capped["molecules"] == full["molecules"][:1]
    assert capped["n_rows"] == full["n_rows"]
    assert capped["rows_not_shown"] == full["n_rows"] - 1
    assert capped["reads_not_shown"] + capped["molecules"][0]["reads"] == locus.n_reads


def _page_blocks(page: str) -> dict:
    """The JSON elements of a written page, by element id."""
    return {m.group(1): json.loads(m.group(2)) for m in re.finditer(
        r'<script type="application/json" id="([^"]+)">(.*?)</script>', page, re.S)}


def test_the_cli_writes_a_page_with_each_requested_locus(tmp_path):
    db, panel = _panel(tmp_path, INVARIANT)
    out = tmp_path / "page.html"
    cli.main(["view-loci", "--panel", str(db), "--loci", "PolyG151+,PolyG151-",
              "--out", str(out)])
    page = out.read_text()
    elements = _page_blocks(page)
    # One data element per locus, and the index.
    assert set(elements) == {"locus-PolyG151+", "locus-PolyG151-", "index"}
    assert [name for name, _ in elements["index"]["loci"]] == ["PolyG151+", "PolyG151-"]
    for name in ("PolyG151+", "PolyG151-"):
        entry = elements[f"locus-{name}"]
        (block,) = entry["blocks"]
        assert elements["index"]["loci"][["PolyG151+", "PolyG151-"].index(name)][1] == \
            [[panel.loci[name].n_reads, "recovered"]]
        reads = Counter()
        for m in block["molecules"]:
            reads[m["tract"]] += m["reads"]
        assert sum(reads.values()) == panel.loci[entry["locus"]].n_reads
    assert "http" not in page.split("<script")[0]  # no external assets
    names = tmp_path / "loci.txt"
    names.write_text("PolyG151-\n")
    cli.main(["view-loci", "--panel", str(db), "--loci-file", str(names), "--out", str(out)])
    assert "PolyG151-" in out.read_text()
