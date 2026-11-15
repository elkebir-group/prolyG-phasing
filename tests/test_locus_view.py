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
    assert [c["index"] for c in up_cols] == list(range(
        view["flanking"]["up"]["width"] - lv.NEAR_BASES, view["flanking"]["up"]["width"]))
    assert up_cols[-1]["position"] == view["flanks"]["upstream"]["outer_position"] - step
    assert dn_cols[0]["position"] == view["flanks"]["downstream"]["outer_position"] + step
    assert up_cols[-1]["position"] == ref_pos_at_offset(
        orientation, view["flanking"]["up"]["start"], up_cols[-1]["index"])


def test_without_phasing_rows_group_by_tract(tmp_path):
    _, panel = _panel(tmp_path, MOVED)
    for name in ("PolyG5177+", "PolyG5177-"):
        view = locus_view_data(name, panel.loci[name], "S", near=2)
        assert view["phasing"] is None
        assert len(view["pileup"]["up"]) == len(view["pileup"]["dn"]) == 2
        assert [(r["tract"], r["reads"], r["label"]) for r in view["pileup"]["rows"]] == \
            [(t, n, None) for t, n, _ in view["tracts"]]
        assert view["flanking"]["up"]["snv"] == view["flanking"]["dn"]["snv"] == []
    plus = locus_view_data("PolyG5177+", panel.loci["PolyG5177+"], "S")
    minus = locus_view_data("PolyG5177-", panel.loci["PolyG5177-"], "S")
    assert plus["pileup"]["rows"] == minus["pileup"]["rows"]


def test_a_cell_is_the_majority_of_the_rows_reads_and_never_invented(tmp_path):
    _, panel = _panel(tmp_path, MOVED)
    locus = panel.loci["PolyG5177+"]
    view = locus_view_data("PolyG5177+", locus, "S")
    rows = list(_rows(locus))
    for row in view["pileup"]["rows"]:
        mine = [r for r in rows if r[0] == row["tract"]]
        for k, col in enumerate(view["pileup"]["up"]):
            want = Counter()
            for r in mine:
                if r[1][col["index"]] != ".":
                    want[r[1][col["index"]]] += r[4]
            base, counts = row["up"][k]
            assert counts == [want[b] for b in lv.FLANKING_BASES]
            assert base == (max(want, key=lambda b: (want[b], -lv.FLANKING_BASES.index(b)))
                            if want else "")


def test_the_rows_past_the_cap_are_counted(tmp_path, monkeypatch):
    _, panel = _panel(tmp_path, MOVED)
    locus = panel.loci["PolyG5177+"]
    full = locus_view_data("PolyG5177+", locus, "S")
    monkeypatch.setattr(lv, "PILEUP_ROWS", 0)
    capped = locus_view_data("PolyG5177+", locus, "S")
    assert capped["pileup"]["rows"] == []
    assert capped["pileup"]["n_rows"] == full["pileup"]["n_rows"]
    assert capped["pileup"]["reads_not_drawn"] == locus.n_reads


def test_a_side_with_no_position(tmp_path):
    case = INVARIANT
    records = {f"PolyG151{o}": (flank_record(case, case.up, o),
                                FlankRecord(case.dn.bases, (), 0)) for o in ORIENTATIONS}
    _, panel = _panel(tmp_path, case, records=records)
    view = locus_view_data("PolyG151-", panel.loci["PolyG151-"], "S")
    assert view["flanks"]["downstream"]["status"] == "no_reference_position"
    assert view["flanking"]["dn"] == {"start": -1, "width": 0, "counts": [], "snv": []}
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
# With the saved phasing result
# ---------------------------------------------------------------------------

# One SNV at upstream index 4: C in 70 families, G in 30, no base in 10.
SNV_STRINGS = ["AAAACAAT|TTTTTTTT", "AAAAGAAT|TTTTTTTT", "AAAA.AAT|TTTTTTTT"]
SNV_COUNTS = [70, 30, 10]


def _snv_panel(tmp_path, *, mi_prefix="M"):
    """A panel with one heterozygous flanking SNV, and the phase output for it,
    made by the `phase` command."""
    from tests.test_phasing_tables import _make_locus, _make_provenance

    n = sum(SNV_COUNTS)
    seqs = ["GGGGGG" if i % 3 else "GGGGGGG" for i in range(n)]
    locus = _make_locus(flanking_strings=SNV_STRINGS, counts=SNV_COUNTS, seqs=seqs,
                        up_width=8, dn_width=8)
    locus.mi = np.asarray([f"{mi_prefix}{i}" for i in range(n)], dtype=object)
    # Rows 0 and 1 are one read family with two tracts: one (mi, strand) on two rows.
    locus.mi[1] = locus.mi[0]
    db = tmp_path / f"{mi_prefix}.panel.db"
    ExtractedPanel(loci={"L1": locus}, n_alleles=8, provenance=_make_provenance()).save_db(db)
    out = tmp_path / f"{mi_prefix}_phasing"
    cli.main(["phase", "--panel-db", str(db), "--out-dir", str(out)])
    return db, out


def test_with_phasing_the_snv_columns_and_label_rows(tmp_path):
    db, out = _snv_panel(tmp_path)
    phasing = lv.PhasingIndex.read(out)
    panel = ExtractedPanel.load_db(db)
    phasing.check(panel, db)
    view = locus_view_data("L1", panel.loci["L1"], "S", near=2, phasing=phasing)

    snv_cols = [c for c in view["pileup"]["up"] if "snv" in c]
    assert [c["index"] for c in view["pileup"]["up"]] == [4, 6, 7]
    assert [(c["index"], c["snv"]["status"], c["snv"]["allele_major"],
             c["snv"]["allele_minor"], c["snv"]["is_anchor"]) for c in snv_cols] == \
        [(4, "phased", "C", "G", True)]
    assert view["flanking"]["up"]["snv"] == [4]
    assert view["phasing"]["summary"]["reads_major"] == "70"

    labels = pd.read_csv(out / "haplotype_calls.tsv", sep="\t", dtype=str)
    label_of = dict(zip(labels.mi, labels.phase_label, strict=True))
    locus = panel.loci["L1"]
    want = Counter()
    for mi, seq, n in zip(locus.mi, locus.seq, locus.count, strict=True):
        want[(str(seq), label_of[str(mi)])] += int(n)
    rows = view["pileup"]["rows"]
    assert {(r["tract"], r["label"]): r["reads"] for r in rows} == dict(want)
    at_snv = view["pileup"]["up"].index(snv_cols[0])
    for r in rows:
        base, counts = r["up"][at_snv]
        expected = {"hap_major": "C", "hap_minor": "G", "unk": ""}[r["label"]]
        assert base == expected
        if r["label"] == "unk":
            assert counts == [0] * len(lv.FLANKING_BASES)


def test_labels_join_the_panel_rows_in_order(tmp_path):
    """haplotype_calls.tsv has one row per panel row, so one (mi, strand) can
    occur twice; the join is by row position, checked by (mi, strand, count)."""
    db, out = _snv_panel(tmp_path)
    phasing = lv.PhasingIndex.read(out)
    locus = ExtractedPanel.load_db(db).loci["L1"]
    labels = pd.read_csv(out / "haplotype_calls.tsv", sep="\t", dtype=str)
    assert (labels.mi == labels.mi.iloc[0]).sum() == 2
    assert list(labels.mi) == [str(m) for m in locus.mi]
    got = phasing.labels_of("L1", len(locus.mi))
    assert [lv.PHASE_LABELS[g] for g in got] == list(labels.phase_label)


def test_a_stale_pair_is_refused(tmp_path):
    db, _ = _snv_panel(tmp_path)
    _, other = _snv_panel(tmp_path, mi_prefix="X")
    with pytest.raises(ValueError, match=r"X_phasing does not belong to the panel .*M\.panel"
                                         r"\.db: the read families differ at L1"):
        lv.make_server([(db, other)], 0)


# ---------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------


@pytest.fixture
def server(tmp_path):
    db, out = _snv_panel(tmp_path)
    srv = lv.make_server([(db, out), (db, None)], 0)
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
    assert status == 200 and b"<html" in page
    listing = json.loads(_get(server + "/loci")[1])
    assert [e["locus"] for e in listing["loci"]] == ["L1"]
    assert [s["n_snvs"] for s in listing["loci"][0]["samples"]] == [1, None]
    assert listing["samples"][0]["phasing"] is not None and listing["samples"][1]["phasing"] is None
    with_phasing = json.loads(_get(server + "/locus?sample=0&locus=L1")[1])
    without = json.loads(_get(server + "/locus?sample=1&locus=L1")[1])
    assert {r["label"] for r in with_phasing["pileup"]["rows"]} == \
        {"hap_major", "hap_minor", "unk"}
    assert {r["label"] for r in without["pileup"]["rows"]} == {None}


def test_an_unknown_sample_or_locus_is_404(server):
    for query in ("sample=0&locus=L9", "sample=5&locus=L1"):
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(server + "/locus?" + query)
        assert err.value.code == 404
        assert b"No locus" in err.value.read()


def test_the_near_setting_changes_the_columns(server):
    def up_columns(query):
        data = json.loads(_get(server + "/locus?sample=0&locus=L1&" + query)[1])
        return [c["index"] for c in data["pileup"]["up"]]

    assert up_columns("near=2") == [4, 6, 7]
    assert up_columns("near=0") == [4]
    assert up_columns("") == list(range(2, 8))
