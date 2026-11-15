"""The locus viewer's data, read from panels built from hand-built reads."""

from __future__ import annotations

import json
from collections import Counter

import pandas as pd
import pytest

import prolyg_phasing.cli as cli
import prolyg_phasing.locus_view as lv
from prolyg_phasing.io._flank_placement import FlankRecord
from prolyg_phasing.io.bam import extract_panel
from prolyg_phasing.io.panel import ExtractedPanel, parse_run_lengths
from prolyg_phasing.locus_view import locus_view_data
from prolyg_phasing.phasing import ref_pos_at_offset
from tests.test_extract_hand_built import INVARIANT, MOVED, _write_case
from tests.test_flank_placement import flank_record

ORIENTATIONS = ("+", "-")
N = lv.PILEUP_FLANKING


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


def _rows(locus):
    """(tract, upstream flanking, downstream flanking, MI, reads) of each panel row."""
    for mi, seq, fid, count in zip(locus.mi, locus.seq, locus.flanking_id, locus.count,
                                   strict=True):
        up, dn = str(locus.flanking_seq[int(fid)]).split("|")
        yield str(seq), up, dn, str(mi), int(count)


def _near(up: str, dn: str) -> tuple[str, str]:
    """The N flanking bases next to each flank, padded with "." where a side is short."""
    return up[-N:].rjust(N, ".") if up else "." * N, dn[:N].ljust(N, ".")


@pytest.mark.parametrize("case", [MOVED, INVARIANT], ids=["moved", "invariant"])
def test_tracts_pileup_and_base_counts_match_the_panel_rows(tmp_path, case):
    _, panel = _panel(tmp_path, case)
    for o in ORIENTATIONS:
        name = f"{case.locus.name}{o}"
        locus = panel.loci[name]
        view = locus_view_data(name, locus, "S")
        rows = list(_rows(locus))
        assert rows and view["counters"]["n_reads"] == locus.n_reads

        reads, families = Counter(), {}
        for t, _, _, mi, n in rows:
            reads[t] += n
            families.setdefault(t, set()).add(mi)
        assert {t: (n, m) for t, n, m in view["tracts"]} == \
            {t: (reads[t], len(families[t])) for t in reads}
        assert view["n_tracts"] == len(reads)
        largest = Counter()
        for t, n in reads.items():
            largest[max(parse_run_lengths(t))] += n
        assert dict(map(tuple, view["largest_run"])) == dict(largest)

        piled = Counter()
        for t, up, dn, _, n in rows:
            u, d = _near(up, dn)
            piled[(u, t, d)] += n
        assert {(u, t, d): n for u, t, d, n in view["pileup"]["rows"]} == dict(piled)
        assert view["pileup"]["reads_not_drawn"] == 0
        assert [r[3] for r in view["pileup"]["rows"]] == \
            sorted((r[3] for r in view["pileup"]["rows"]), reverse=True)

        flanking = view["flanking"]
        for side, column in (("up", 1), ("dn", 2)):
            counts = flanking[side]["counts"]
            assert len(counts) == flanking[side]["width"]
            for i, per_base in enumerate(counts):
                want = Counter()
                for row in rows:
                    base = row[column][i]
                    if base != ".":
                        want[base] += row[4]
                assert dict(zip(lv.FLANKING_BASES, per_base, strict=True)) == \
                    {b: want[b] for b in lv.FLANKING_BASES}


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_flanking_coordinates_start_next_to_each_flank(tmp_path, orientation):
    """The nearest flanking base sits one reference base past each flank's outer base,
    on a "+" and on a "-" locus."""
    _, panel = _panel(tmp_path, MOVED)
    name = f"PolyG5177{orientation}"
    view = locus_view_data(name, panel.loci[name], "S")
    up, dn = view["flanking"]["up"], view["flanking"]["dn"]
    step = 1 if orientation == "+" else -1
    nearest_up = ref_pos_at_offset(orientation, up["start"], up["width"] - 1)
    assert nearest_up == view["flanks"]["upstream"]["outer_position"] - step
    assert ref_pos_at_offset(orientation, dn["start"], 0) == \
        view["flanks"]["downstream"]["outer_position"] + step


def test_both_strands_give_one_pileup(tmp_path):
    _, panel = _panel(tmp_path, MOVED)
    plus = locus_view_data("PolyG5177+", panel.loci["PolyG5177+"], "S")
    minus = locus_view_data("PolyG5177-", panel.loci["PolyG5177-"], "S")
    assert plus["tracts"] == minus["tracts"]
    assert plus["pileup"]["rows"] == minus["pileup"]["rows"]


def test_the_rows_past_the_caps_are_counted(tmp_path, monkeypatch):
    _, panel = _panel(tmp_path, MOVED)
    locus = panel.loci["PolyG5177+"]
    full = locus_view_data("PolyG5177+", locus, "S")
    assert full["pileup"]["n_rows"] == len(full["pileup"]["rows"]) >= 2
    monkeypatch.setattr(lv, "PILEUP_ROWS", 1)
    monkeypatch.setattr(lv, "TRACT_ROWS", 1)
    capped = locus_view_data("PolyG5177+", locus, "S")
    assert capped["pileup"]["rows"] == full["pileup"]["rows"][:1]
    assert capped["pileup"]["rows"][0][3] + capped["pileup"]["reads_not_drawn"] == \
        locus.n_reads
    assert capped["tracts"] == full["tracts"][:1]
    assert capped["tracts"][0][1] + capped["tract_reads_not_shown"] == locus.n_reads


def test_a_side_with_no_position(tmp_path):
    case = INVARIANT
    records = {f"PolyG151{o}": (flank_record(case, case.up, o),
                                FlankRecord(case.dn.bases, (), 0)) for o in ORIENTATIONS}
    _, panel = _panel(tmp_path, case, records=records)
    view = locus_view_data("PolyG151-", panel.loci["PolyG151-"], "S")
    assert view["flanks"]["downstream"]["status"] == "no_reference_position"
    assert view["flanking"]["dn"] == {"start": -1, "width": 0, "counts": []}
    assert view["flanking"]["up"]["width"] > 0
    assert {r[2] for r in view["pileup"]["rows"]} == {"." * N}
    assert sum(r[3] for r in view["pileup"]["rows"]) == view["counters"]["n_reads"] > 0


def test_a_locus_with_no_reference_window(tmp_path):
    def no_window(table):
        plus = table.locus == "PolyG151+"
        table.loc[plus, "reference_window_status"] = "ref_window_unrecoverable"
        table.loc[plus, ["flank_bases", "flank_ref_positions"]] = ""

    _, panel = _panel(tmp_path, INVARIANT, edit=no_window)
    view = locus_view_data("PolyG151+", panel.loci["PolyG151+"], "S")
    assert view["reference_window_status"] == "ref_window_unrecoverable"
    assert view["flanks"] is None
    assert view["tracts"] == [] and view["pileup"]["rows"] == []
    assert view["flanking"]["up"] == {"start": -1, "width": 0, "counts": []}


def test_the_cli_writes_one_file_per_sample_and_locus(tmp_path):
    db, panel = _panel(tmp_path, INVARIANT)
    out = tmp_path / "view"
    cli.main(["view-loci", "--panel", str(db), "--panel", str(db), "--out-dir", str(out)])
    listing = json.loads((out / "loci.json").read_text())
    assert [e["locus"] for e in listing["loci"]] == ["PolyG151+", "PolyG151-"]
    assert [s["sample"] for s in listing["samples"]] == ["sample", "sample"]
    for e in listing["loci"]:
        for s in range(2):
            data = json.loads((out / "data" / str(s) / f"{e['file']}.json").read_text())
            assert data["locus"] == e["locus"]
            assert e["samples"][s]["n_reads"] == panel.loci[e["locus"]].n_reads
    page = (out / "index.html").read_text()
    assert "loci.json" in page and "http.server" in page
    assert "googleapis" not in page

    names = tmp_path / "loci.txt"
    names.write_text("PolyG151-\n")
    one = tmp_path / "one"
    cli.main(["view-loci", "--panel", str(db), "--loci-file", str(names), "--out-dir", str(one)])
    assert [e["locus"] for e in json.loads((one / "loci.json").read_text())["loci"]] == \
        ["PolyG151-"]
    assert sorted(p.name for p in (one / "data" / "0").iterdir()) == ["0.json"]
