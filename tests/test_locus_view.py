"""The locus viewer: its data, read from panels built from hand-built reads, and its server."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd
import pytest

import prolyg_phasing.locus_view as lv
from prolyg_phasing.io._flank_placement import FlankRecord
from prolyg_phasing.io.bam import extract_panel
from prolyg_phasing.io.panel import ExtractedPanel, parse_run_lengths
from prolyg_phasing.locus_view import locus_view_data
from prolyg_phasing.phasing import ref_pos_at_offset
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


def _rows(locus):
    """(tract, upstream flanking, downstream flanking, MI, reads) of each panel row."""
    for mi, seq, fid, count in zip(locus.mi, locus.seq, locus.flanking_id, locus.count,
                                   strict=True):
        up, dn = str(locus.flanking_seq[int(fid)]).split("|")
        yield str(seq), up, dn, str(mi), int(count)


# ---------------------------------------------------------------------------
# The data of one locus
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", [MOVED, INVARIANT], ids=["moved", "invariant"])
def test_tracts_and_base_counts_match_the_panel_rows(tmp_path, case):
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
        largest = Counter()
        for t, n in reads.items():
            largest[max(parse_run_lengths(t))] += n
        assert dict(map(tuple, view["largest_run"])) == dict(largest)
        for side, column in (("up", 1), ("dn", 2)):
            for i, per_base in enumerate(view["flanking"][side]["counts"]):
                want = Counter()
                for row in rows:
                    if row[column][i] != ".":
                        want[row[column][i]] += row[4]
                assert per_base == [want[b] for b in lv.FLANKING_BASES]
        assert sum(r["reads"] for r in view["pileup"]["rows"]) == locus.n_reads


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_column_coordinates_on_each_strand(tmp_path, orientation):
    """The nearest flanking base sits one reference base past each flank's outer
    base, and the near columns run to it."""
    _, panel = _panel(tmp_path, MOVED)
    name = f"PolyG5177{orientation}"
    view = locus_view_data(name, panel.loci[name], "S")
    step = 1 if orientation == "+" else -1
    up_cols, dn_cols = view["pileup"]["up"], view["pileup"]["dn"]
    assert [c["index"] for c in up_cols][-1] == view["flanking"]["up"]["width"] - 1
    assert up_cols[-1]["position"] == view["flanks"]["upstream"]["outer_position"] - step
    assert dn_cols[0]["position"] == view["flanks"]["downstream"]["outer_position"] + step
    assert up_cols[-1]["position"] == ref_pos_at_offset(
        orientation, view["flanking"]["up"]["start"], up_cols[-1]["index"])


def test_both_strands_give_one_pileup(tmp_path):
    _, panel = _panel(tmp_path, MOVED)
    plus = locus_view_data("PolyG5177+", panel.loci["PolyG5177+"], "S")
    minus = locus_view_data("PolyG5177-", panel.loci["PolyG5177-"], "S")
    assert plus["tracts"] == minus["tracts"]
    assert plus["pileup"]["rows"] == minus["pileup"]["rows"]


def test_a_locus_with_no_heterozygous_column_has_one_row_per_tract(tmp_path):
    _, panel = _panel(tmp_path, MOVED)
    view = locus_view_data("PolyG5177+", panel.loci["PolyG5177+"], "S", het_limit=0.5)
    assert not any(c["het"] for c in view["pileup"]["up"] + view["pileup"]["dn"])
    assert [r["tract"] for r in view["pileup"]["rows"]] == [t for t, _, _ in view["tracts"]]
    assert [r["reads"] for r in view["pileup"]["rows"]] == [n for _, n, _ in view["tracts"]]


def test_the_rows_past_the_caps_are_counted(tmp_path, monkeypatch):
    """With no limit, every covered flanking position is a heterozygous column,
    so the reads split into several rows; one row is kept and the rest counted."""
    _, panel = _panel(tmp_path, MOVED)
    locus = panel.loci["PolyG5177+"]
    full = locus_view_data("PolyG5177+", locus, "S", het_limit=0.0)
    assert full["pileup"]["n_rows"] == len(full["pileup"]["rows"]) >= 2
    monkeypatch.setattr(lv, "PILEUP_ROWS", 1)
    capped = locus_view_data("PolyG5177+", locus, "S", het_limit=0.0)
    assert capped["pileup"]["rows"] == full["pileup"]["rows"][:1]
    assert capped["pileup"]["rows"][0]["reads"] + capped["pileup"]["reads_not_drawn"] == \
        locus.n_reads


def test_a_side_with_no_position(tmp_path):
    case = INVARIANT
    records = {f"PolyG151{o}": (flank_record(case, case.up, o),
                                FlankRecord(case.dn.bases, (), 0)) for o in ORIENTATIONS}
    _, panel = _panel(tmp_path, case, records=records)
    view = locus_view_data("PolyG151-", panel.loci["PolyG151-"], "S")
    assert view["flanks"]["downstream"]["status"] == "no_reference_position"
    assert view["flanking"]["dn"] == {"start": -1, "width": 0, "counts": [], "het": []}
    assert view["pileup"]["dn"] == []
    assert all(r["dn"] == [] for r in view["pileup"]["rows"])
    assert sum(r["reads"] for r in view["pileup"]["rows"]) == view["counters"]["n_reads"] > 0


def test_a_locus_with_no_reference_window(tmp_path):
    def no_window(table):
        plus = table.locus == "PolyG151+"
        table.loc[plus, "reference_window_status"] = "ref_window_unrecoverable"
        table.loc[plus, ["flank_bases", "flank_ref_positions"]] = ""

    _, panel = _panel(tmp_path, INVARIANT, edit=no_window)
    view = locus_view_data("PolyG151+", panel.loci["PolyG151+"], "S")
    assert view["flanks"] is None and view["tracts"] == [] and view["pileup"]["rows"] == []


# ---------------------------------------------------------------------------
# Heterozygous columns and the grouping rule
# ---------------------------------------------------------------------------


def test_heterozygous_positions_use_the_second_base_share():
    counts = np.array([[80, 0, 20, 0, 0],    # second base 20 %: heterozygous
                       [81, 0, 19, 0, 0],    # 19 %: not
                       [0, 0, 0, 0, 0],      # no coverage
                       [40, 30, 30, 0, 0]])  # second 30 %
    assert lv.heterozygous(counts, 0.2).tolist() == [True, False, False, True]


def test_columns_are_the_near_bases_and_the_heterozygous_positions():
    het = np.zeros(20, dtype=bool)
    het[[2, 18]] = True
    assert lv._columns(20, het, 3, upstream=True) == [2, 17, 18, 19]
    assert lv._columns(20, het, 3, upstream=False) == [0, 1, 2, 18]
    assert lv._columns(2, np.zeros(2, dtype=bool), 6, upstream=True) == [0, 1]


def _group(items, near_cols=0):
    """`items`: (tract, het bases text, reads); near columns all empty."""
    tract = np.array([t for t, _, _ in items], dtype=object)
    het = np.array([[ord(c) for c in h] for _, h, _ in items], dtype=np.uint8)
    near = np.full((len(items), near_cols), ord("."), dtype=np.uint8)
    reads = np.array([n for _, _, n in items], dtype=np.int64)
    groups = lv._pileup_groups(tract, het, near, reads)
    return {(g["tract"], g["het"].decode()): (g["reads"], g["joined"]) for g in groups}


def test_a_read_with_an_empty_cell_joins_the_one_group_it_matches():
    assert _group([("GGG", "AC", 10), ("GGG", "TC", 5), ("GGG", "A.", 2)]) == \
        {("GGG", "AC"): (12, 2), ("GGG", "TC"): (5, 0)}


def test_of_two_matching_groups_the_read_joins_the_larger():
    assert _group([("GGG", "AC", 10), ("GGG", "AT", 30), ("GGG", "A.", 2)]) == \
        {("GGG", "AC"): (10, 0), ("GGG", "AT"): (32, 2)}


def test_a_read_that_matches_no_group_forms_its_own():
    """No group has its tract, or its bases disagree: it forms a group, shared by
    reads of the same pattern."""
    assert _group([("GGG", "AC", 10), ("GGGG", "A.", 2), ("GGG", "T.", 1),
                   ("GGG", "T.", 3)]) == \
        {("GGG", "AC"): (10, 0), ("GGGG", "A."): (2, 2), ("GGG", "T."): (4, 4)}


def test_with_no_heterozygous_column_each_tract_is_a_group():
    assert _group([("GGG", "", 4), ("GGG", "", 1), ("GG", "", 2)]) == \
        {("GGG", ""): (5, 0), ("GG", ""): (2, 0)}


# ---------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------


@pytest.fixture
def server(tmp_path):
    db, _ = _panel(tmp_path, MOVED)
    srv = lv.make_server([db, db], 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()
    thread.join()


def _get(url):
    with urllib.request.urlopen(url) as response:
        return response.status, response.read()


def test_the_server_serves_the_page_the_list_and_one_locus(server):
    status, page = _get(server + "/")
    assert status == 200 and b"/locus" not in page.split(b"<script")[0]
    status, body = _get(server + "/loci")
    listing = json.loads(body)
    assert [e["locus"] for e in listing["loci"]] == ["PolyG5177+", "PolyG5177-"]
    assert listing["loci"][0]["samples"][0]["n_reads"] == 5
    assert listing["loci"][0]["samples"][1]["status"] == "recovered"
    status, body = _get(server + "/locus?sample=1&locus=PolyG5177-")
    data = json.loads(body)
    assert data["locus"] == "PolyG5177-" and data["counters"]["n_reads"] == 5


def test_an_unknown_sample_or_locus_is_404(server):
    for query in ("sample=0&locus=PolyG9", "sample=5&locus=PolyG5177+"):
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(server + "/locus?" + query)
        assert err.value.code == 404
        assert b"No locus" in err.value.read()


def test_a_settings_change_changes_the_columns(server):
    def columns(query):
        data = json.loads(_get(server + "/locus?sample=0&locus=PolyG5177%2B&" + query)[1])
        return data["pileup"]["up"] + data["pileup"]["dn"]

    strict = columns("het=0.5")
    assert len(strict) == 2 * lv.NEAR_BASES and not any(c["het"] for c in strict)
    assert len(columns("het=0.5&near=2")) == 4
    default = columns("")
    assert sum(c["het"] for c in default) > 0
    assert sum(c["het"] for c in columns("het=0")) > sum(c["het"] for c in default)
