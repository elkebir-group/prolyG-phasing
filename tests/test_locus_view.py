"""The locus viewer: its data, read from panels built from hand-built reads, and its server."""

from __future__ import annotations

import dataclasses
import json
import re
import threading
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import prolyg_phasing.cli as cli
import prolyg_phasing.io._flank_placement as fp
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


def _all_rows(view):
    """The drawn pile-up rows of every section, in section order."""
    return [row for section in view["pileup"]["sections"] for row in section["rows"]]


def _section_sum(view, field):
    return sum(section[field] for section in view["pileup"]["sections"])


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
        assert sum(r["reads"] for r in _all_rows(view)) == locus.n_reads




@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_column_coordinates_on_each_strand(tmp_path, orientation):
    """The nearest flanking base sits one reference base past each flank's outer
    base, and the near columns run to it."""
    _, panel = _panel(tmp_path, MOVED)
    name = f"PolyG5177{orientation}"
    view = locus_view_data(name, panel.loci[name], "S", near=6)
    step = 1 if orientation == "+" else -1
    up_cols, dn_cols = view["pileup"]["up"], view["pileup"]["dn"]
    assert [c["index"] for c in up_cols] == list(range(
        view["flanking"]["up"]["width"] - 6, view["flanking"]["up"]["width"]))
    assert up_cols[-1]["position"] == view["flanks"]["upstream"]["outer_position"] - step
    assert dn_cols[0]["position"] == view["flanks"]["downstream"]["outer_position"] + step
    assert up_cols[-1]["position"] == ref_pos_at_offset(
        orientation, view["flanking"]["up"]["start"], up_cols[-1]["index"])


def _exact_rows(view, rows, label_of=None):
    """The pile-up rows from the panel rows, counted by hand: tract, label, and
    the read's own base at every shown column."""
    up_idx = [c["index"] for c in view["pileup"]["up"]]
    dn_idx = [c["index"] for c in view["pileup"]["dn"]]
    want = Counter()
    for t, up, dn, mi, n in rows:
        label = None if label_of is None else label_of[mi]
        want[(t, label, "".join(up[i] for i in up_idx), "".join(dn[i] for i in dn_idx))] += n
    return want


def _drawn(view):
    return {(r["tract"], r["label"], r["up"], r["dn"]): r["reads"]
            for r in _all_rows(view)}


def test_without_phasing_a_row_is_exact(tmp_path):
    """A row's cells are its reads' own bases, and reads that differ at any
    shown column are in different rows."""
    _, panel = _panel(tmp_path, MOVED)
    for name in ("PolyG5177+", "PolyG5177-"):
        locus = panel.loci[name]
        view = locus_view_data(name, locus, "S", near=2)
        assert view["phasing"] is None
        assert len(view["pileup"]["up"]) == len(view["pileup"]["dn"]) == 2
        assert view["flanking"]["up"]["snv"] == view["flanking"]["dn"]["snv"] == []
        want = _exact_rows(view, list(_rows(locus)))
        assert _drawn(view) == dict(want)
        assert len(_all_rows(view)) == _section_sum(view, "n_rows") == len(want)
    plus = locus_view_data("PolyG5177+", panel.loci["PolyG5177+"], "S")
    minus = locus_view_data("PolyG5177-", panel.loci["PolyG5177-"], "S")
    assert _all_rows(plus) == _all_rows(minus)


def test_the_rows_past_the_cap_are_counted(tmp_path, monkeypatch):
    _, panel = _panel(tmp_path, MOVED)
    locus = panel.loci["PolyG5177+"]
    full = locus_view_data("PolyG5177+", locus, "S")
    monkeypatch.setattr(lv, "PILEUP_ROWS", 0)
    capped = locus_view_data("PolyG5177+", locus, "S")
    assert _all_rows(capped) == []
    assert _section_sum(capped, "n_rows") == _section_sum(full, "n_rows")
    assert _section_sum(capped, "reads_not_drawn") == locus.n_reads


def test_a_side_with_no_position(tmp_path):
    case = INVARIANT
    records = {f"PolyG151{o}": (flank_record(case, case.up, o),
                                FlankRecord(case.dn.bases, (), 0)) for o in ORIENTATIONS}
    _, panel = _panel(tmp_path, case, records=records)
    view = locus_view_data("PolyG151-", panel.loci["PolyG151-"], "S")
    assert view["flanks"]["downstream"]["status"] == "no_reference_position"
    assert view["flanking"]["dn"] == {"start": -1, "width": 0, "counts": [], "snv": []}
    assert view["pileup"]["dn"] == []
    assert all(r["dn"] == "" for r in _all_rows(view))
    assert sum(r["reads"] for r in _all_rows(view)) == view["counters"]["n_reads"] > 0


def test_a_locus_with_no_reference_window(tmp_path):
    def no_window(table):
        plus = table.locus == "PolyG151+"
        table.loc[plus, "reference_window_status"] = "ref_window_unrecoverable"
        table.loc[plus, ["flank_bases", "flank_ref_positions"]] = ""

    _, panel = _panel(tmp_path, INVARIANT, edit=no_window)
    view = locus_view_data("PolyG151+", panel.loci["PolyG151+"], "S")
    assert view["flanks"] is None and view["tracts"] == [] and _all_rows(view) == []


# ---------------------------------------------------------------------------
# The counter help
# ---------------------------------------------------------------------------


def test_every_counter_has_a_help_text_and_every_text_a_counter(tmp_path):
    _, panel = _panel(tmp_path, INVARIANT)
    for name, locus in panel.loci.items():
        counters = locus_view_data(name, locus, "S")["counters"]
        assert set(lv.HELP_TEXTS["counters"]) == set(counters)
    texts = lv.view_help(panel.provenance)
    assert {g: set(t) for g, t in texts.items()} == \
        {g: set(t) for g, t in lv.HELP_TEXTS.items()}
    every = [t for group in texts.values() for t in group.values()]
    assert all(every) and not any("{" in t or "}" in t for t in every)


def test_every_status_label_and_filter_chip_has_a_help_text():
    # Every reference_window_status the extraction writes, the label the page
    # derives for a side with no position, and the label of a missing locus.
    written = {fp.WINDOW_RECOVERED, fp.WINDOW_UNRECOVERABLE, fp.NO_FLANK_PAIR}
    assert set(lv.HELP_TEXTS["status"]) == \
        written | {fp.NO_REFERENCE_POSITION, "not_in_panel"}
    page = (Path(lv.__file__).parent / "locus_view.html").read_text()
    chips = re.search(r"const F=\[(.*?)\];", page).group(1)
    assert set(re.findall(r'\["(\w+)",', chips)) == set(lv.HELP_TEXTS["filters"])
    for key in ("not_in_panel", "no_reference_position", "recovered"):
        assert f'"{key}"' in page


def test_the_status_help_names_the_stored_field_and_the_derived_labels():
    status = lv.HELP_TEXTS["status"]
    for stored in (fp.WINDOW_RECOVERED, fp.WINDOW_UNRECOVERABLE, fp.NO_FLANK_PAIR):
        assert "stored status" in status[stored]
    for derived in (fp.NO_REFERENCE_POSITION, "not_in_panel"):
        assert "The page derives this label" in status[derived]
    for no_flank in (fp.WINDOW_UNRECOVERABLE, fp.NO_FLANK_PAIR):
        assert "no reads" in status[no_flank]


def test_the_counter_help_states_the_panel_limits():
    from tests.test_phasing_tables import _make_provenance

    provenance = dataclasses.replace(_make_provenance(), min_mapq=17, max_tlen=777,
                                     drop_chimeric=False)
    texts = lv.view_help(provenance)
    counters = texts["counters"]
    assert "below 17." in counters["n_alignments_drop_mapq"]
    assert "MAPQ 17 or more" in texts["status"][fp.WINDOW_UNRECOVERABLE]
    assert "above 777," in counters["n_alignments_drop_tlen"]
    assert "--no-drop-chimeric" in counters["n_alignments_drop_sa"]
    assert "count is 0" in counters["n_alignments_drop_sa"]
    assert "drops them" in lv.view_help(_make_provenance())["counters"]["n_alignments_drop_sa"]
    # The placement value in whole units: G +1, other base -2, flank costs 1, 2, 2.5.
    value = counters["n_alignments_drop_negative_value"]
    assert all(x in value for x in ("+1 per G", "\u22122 per other", "1 per flank", "2 or 2.5"))


def test_the_list_carries_each_samples_help(server):
    listing = json.loads(_get(server + "/loci")[1])
    for sample in listing["samples"]:
        assert {g: set(t) for g, t in sample["help"].items()} == \
            {g: set(t) for g, t in lv.HELP_TEXTS.items()}


# ---------------------------------------------------------------------------
# With the saved phasing result
# ---------------------------------------------------------------------------

# One SNV at upstream index 4: C in 70 families, G in 30, no base in 10.
SNV_STRINGS = ["AAAACAAT|TTTTTTTT", "AAAAGAAT|TTTTTTTT", "AAAA.AAT|TTTTTTTT"]
SNV_COUNTS = [70, 30, 10]


def _snv_panel(tmp_path, *, mi_prefix="M", edit=None):
    """A panel with one heterozygous flanking SNV, and the phase output for it,
    made by the `phase` command. `edit` changes the locus before it is saved."""
    from tests.test_phasing_tables import _make_locus, _make_provenance

    n = sum(SNV_COUNTS)
    seqs = ["GGGGGG" if i % 3 else "GGGGGGG" for i in range(n)]
    locus = _make_locus(flanking_strings=SNV_STRINGS, counts=SNV_COUNTS, seqs=seqs,
                        up_width=8, dn_width=8)
    locus.mi = np.asarray([f"{mi_prefix}{i}" for i in range(n)], dtype=object)
    # Rows 0 and 1 are one read family with two tracts: one (mi, strand) on two rows.
    locus.mi[1] = locus.mi[0]
    if edit is not None:
        edit(locus)
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
    want = _exact_rows(view, list(_rows(locus)), label_of)
    assert _drawn(view) == dict(want)
    # The cell at the SNV is each read's own base: "." where it has none.
    at_snv = view["pileup"]["up"].index(snv_cols[0])
    assert {(r["label"], r["up"][at_snv]) for r in _all_rows(view)} == \
        {("hap_major", "C"), ("hap_minor", "G"), ("unk", ".")}


def test_the_list_shows_phasing_snvs_no_snv_or_no_phasing_result(tmp_path):
    """L1 has a phased SNV; L2's reads share one flanking string, so its phasing
    result has no SNV; the second sample is given without a phasing directory."""
    from tests.test_phasing_tables import _make_locus, _make_provenance

    db, _ = _snv_panel(tmp_path)
    one = ExtractedPanel.load_db(db).loci["L1"]
    flat = _make_locus(flanking_strings=["AAAACAAT|TTTTTTTT"], counts=[40],
                       seqs=["GGGGGG"] * 40, up_width=8, dn_width=8)
    both = tmp_path / "both.panel.db"
    ExtractedPanel(loci={"L1": one, "L2": flat}, n_alleles=8,
                   provenance=_make_provenance()).save_db(both)
    out = tmp_path / "both_phasing"
    cli.main(["phase", "--panel-db", str(both), "--out-dir", str(out)])
    panel = ExtractedPanel.load_db(both)
    listing = lv.locus_list([panel, panel], [lv.PhasingIndex.read(out), None],
                            [lv.molecule_counts(both, 1)] * 2)
    snvs = {e["locus"]: [s["snvs"] for s in e["samples"]] for e in listing["loci"]}
    assert snvs == {"L1": [{"phased": 1, "dropped": 0}, None],
                    "L2": [{"phased": 0, "dropped": 0}, None]}
    assert [s["phasing"] is not None for s in listing["samples"]] == [True, False]


def test_the_list_molecules_are_the_tract_tables_distinct_mi(tmp_path):
    """Six rows: M0 on both strands, M1 under two tracts, M2 once. Three molecules."""
    from tests.test_phasing_tables import _make_locus, _make_provenance

    locus = _make_locus(flanking_strings=["AAAA|TTTT"], counts=[6],
                        seqs=["GGGGGG", "GGGGGG", "GGGGGG", "GGGGGGG", "GGGGGG", "GGGGGG"],
                        up_width=4, dn_width=4)
    locus.mi = np.asarray(["M0", "M0", "M1", "M1", "M2", "M2"], dtype=object)
    locus.strand = np.asarray(["A", "B", "A", "A", "A", "A"], dtype="U1")
    db = tmp_path / "molecules.panel.db"
    ExtractedPanel(loci={"L1": locus}, n_alleles=8, provenance=_make_provenance()).save_db(db)
    panel = ExtractedPanel.load_db(db)
    view = locus_view_data("L1", panel.loci["L1"], "S")
    # The tract table: M0, M1 and M2 under GGGGGG; M1 alone under GGGGGGG.
    assert {t: m for t, _, m in view["tracts"]} == {"GGGGGG": 3, "GGGGGGG": 1}
    assert view["n_molecules"] == 3
    for n_workers in (1, 2):
        listing = lv.locus_list([panel], [None], [lv.molecule_counts(db, n_workers)])
        assert listing["loci"][0]["samples"][0]["molecules"] == view["n_molecules"]


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
    assert [s["snvs"] for s in listing["loci"][0]["samples"]] == \
        [{"phased": 1, "dropped": 0}, None]
    assert listing["samples"][0]["phasing"] is not None and listing["samples"][1]["phasing"] is None
    with_phasing = json.loads(_get(server + "/locus?sample=0&locus=L1")[1])
    without = json.loads(_get(server + "/locus?sample=1&locus=L1")[1])
    assert {r["label"] for r in _all_rows(with_phasing)} == \
        {"hap_major", "hap_minor", "unk"}
    assert {r["label"] for r in _all_rows(without)} == {None}


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
    assert up_columns("near=6") == list(range(2, 8))
    # The default shows no bases next to the flank: only the SNV column.
    assert lv.NEAR_BASES == 0 and up_columns("") == [4]


# ---------------------------------------------------------------------------
# Haplotype sections, headers, marks and per-label counts
# ---------------------------------------------------------------------------


def _phased_view(tmp_path, **kwargs):
    db, out = _snv_panel(tmp_path)
    phasing = lv.PhasingIndex.read(out)
    locus = ExtractedPanel.load_db(db).loci["L1"]
    labels = pd.read_csv(out / "haplotype_calls.tsv", sep="\t", dtype=str)
    return locus, list(labels.phase_label), locus_view_data("L1", locus, "S", phasing=phasing,
                                                            **kwargs)


def test_sections_come_in_label_order_each_with_its_own_cap(tmp_path, monkeypatch):
    locus, labels, full = _phased_view(tmp_path)
    sections = full["pileup"]["sections"]
    assert [s["label"] for s in sections] == ["hap_major", "hap_minor", "unk"]
    reads = Counter()
    for label, n in zip(labels, locus.count, strict=True):
        reads[label] += int(n)
    assert [s["reads"] for s in sections] == [reads[x] for x in lv.PHASE_LABELS]
    assert all(r["label"] == s["label"] for s in sections for r in s["rows"])
    monkeypatch.setattr(lv, "PILEUP_ROWS", 1)
    capped = locus_view_data("L1", locus, "S", phasing=lv.PhasingIndex.read(
        tmp_path / "M_phasing"))
    for s, f in zip(capped["pileup"]["sections"], sections, strict=True):
        assert s["rows"] == f["rows"][:1] and s["n_rows"] == f["n_rows"]
        assert s["reads_not_drawn"] == s["reads"] - s["rows"][0]["reads"]


def test_tract_and_largest_run_counts_by_label(tmp_path):
    locus, labels, view = _phased_view(tmp_path)
    by_tract, by_run = {}, {}
    for seq, label, n in zip(locus.seq, labels, locus.count, strict=True):
        k = lv.PHASE_LABELS.index(label)
        by_tract.setdefault(str(seq), [0, 0, 0])[k] += int(n)
        by_run.setdefault(max(parse_run_lengths(str(seq))), [0, 0, 0])[k] += int(n)
    assert {t: v for (t, _, _), v in zip(view["tracts"], view["tract_labels"], strict=True)} == \
        by_tract
    assert dict(map(tuple, ((k, tuple(v)) for k, v in view["largest_run_labels"]))) == \
        {k: tuple(v) for k, v in by_run.items()}


def test_without_phasing_there_is_one_section_and_no_label_counts(tmp_path):
    db, _ = _snv_panel(tmp_path)
    view = locus_view_data("L1", ExtractedPanel.load_db(db).loci["L1"], "S")
    assert [s["label"] for s in view["pileup"]["sections"]] == [None]
    assert view["tract_labels"] is None and view["largest_run_labels"] is None
    assert all("differs" not in r for r in _all_rows(view))


def test_the_header_bases_are_the_haplotype_bases(tmp_path):
    """At upstream index 1 the T is the less common allele, but it goes with
    the anchor's major base C, so the phasing makes T the major haplotype's
    base there; the column carries hap_major_base T."""
    from tests.test_phasing_tables import _make_locus, _make_provenance

    strings = ["ATAACAAT|TTTTTTTT", "A.AACAAT|TTTTTTTT", "AAAAGAAT|TTTTTTTT"]
    counts = [20, 50, 30]
    locus = _make_locus(flanking_strings=strings, counts=counts, seqs=["GGGGGG"] * 100,
                        up_width=8, dn_width=8)
    db = tmp_path / "flip.panel.db"
    ExtractedPanel(loci={"L1": locus}, n_alleles=8, provenance=_make_provenance()).save_db(db)
    out = tmp_path / "flip_phasing"
    cli.main(["phase", "--panel-db", str(db), "--out-dir", str(out)])
    view = locus_view_data("L1", ExtractedPanel.load_db(db).loci["L1"], "S", near=0,
                           phasing=lv.PhasingIndex.read(out))
    snv = {c["index"]: c["snv"] for c in view["pileup"]["up"]}
    assert (snv[1]["allele_major"], snv[1]["allele_minor"]) == ("A", "T")
    assert (snv[1]["hap_major_base"], snv[1]["hap_minor_base"]) == ("T", "A")
    assert snv[4]["is_anchor"] and snv[4]["hap_major_base"] == "C"
    page = (Path(lv.__file__).parent / "locus_view.html").read_text()
    assert "phased?v.hap_major_base:v.allele_major" in page


def test_the_mark_is_at_exactly_the_cells_that_differ_from_the_label_haplotype(tmp_path):
    """A hand-made phasing result: phased SNVs at upstream indices 1 (major A,
    minor T) and 4 (major C, minor G), and one dropped SNV at index 6."""
    from tests.test_phasing_tables import _make_locus

    strings = ["AAAACAAT|", "ATAACAAT|", "AAAAGAAT|", "A.AAGACT|", "AAAACAAT|"]
    labels = ["hap_major", "hap_major", "hap_minor", "hap_minor", "unk"]
    locus = _make_locus(flanking_strings=strings, counts=[1] * 5, seqs=["GGGGGG"] * 5,
                        up_width=8, dn_width=0)

    def snv(i, major, minor, status="phased"):
        return {"segment": "up", "string_index": i, "ref_pos": 1000 + i,
                "allele_major": major, "allele_minor": minor, "vaf_minor": 0.4,
                "coverage": 5, "status": status, "is_anchor": i == 4,
                "hap_major_base": major if status == "phased" else "",
                "hap_minor_base": minor if status == "phased" else ""}

    phasing = lv.PhasingIndex(
        "hand", {"L1": [snv(1, "A", "T"), snv(4, "C", "G"),
                        snv(6, "A", "C", "dropped_low_linkage")]},
        {"L1": {}}, {"L1": np.array([lv.PHASE_LABELS.index(x) for x in labels],
                                    dtype=np.uint8)}, {})
    view = locus_view_data("L1", locus, "S", near=0, phasing=phasing)
    columns = [c["index"] for c in view["pileup"]["up"]]
    assert columns == [1, 4, 6]
    marks = {(r["label"], r["up"]): r.get("differs") for r in _all_rows(view)}
    assert marks == {
        ("hap_major", "AC" + "A"): {"up": [], "dn": []},     # no difference
        ("hap_major", "TC" + "A"): {"up": [0], "dn": []},    # T at index 1: the minor base
        ("hap_minor", "AG" + "A"): {"up": [0], "dn": []},    # A at index 1: the major base
        ("hap_minor", ".G" + "C"): {"up": [], "dn": []},     # empty cell, dropped SNV: no mark
        ("unk", "AC" + "A"): None,                           # unk rows are never marked
    }


def test_largest_run_shares_are_per_label_hand_counts_over_label_totals(tmp_path):
    locus, labels, view = _phased_view(tmp_path)
    counts, totals = {}, Counter()
    for seq, label, n in zip(locus.seq, labels, locus.count, strict=True):
        run = max(parse_run_lengths(str(seq)))
        counts.setdefault(label, Counter())[run] += int(n)
        totals[label] += int(n)
    runs = sorted({run for c in counts.values() for run in c})
    for label in lv.PHASE_LABELS:
        want = [[run, counts[label][run] / totals[label]] for run in runs] if totals[label] \
            else []
        got = view["largest_run_shares"][label]
        assert [k for k, _ in got] == [k for k, _ in want]
        assert [v for _, v in got] == pytest.approx([v for _, v in want])
        if want:
            assert sum(v for _, v in view["largest_run_shares"][label]) == pytest.approx(1.0)


def test_each_label_counts_its_molecules_and_a_split_molecule_counts_in_each(tmp_path):
    """Rows 0 and 1 are one family. Row 100, a read with no SNV base, is the
    B strand of row 0's molecule: one molecule with major and unk reads."""
    def split_molecule(locus):
        locus.mi[100] = locus.mi[0]
        locus.strand[100] = "B"

    db, out = _snv_panel(tmp_path, edit=split_molecule)
    view = locus_view_data("L1", ExtractedPanel.load_db(db).loci["L1"], "S",
                           phasing=lv.PhasingIndex.read(out))
    by_label = {S["label"]: (S["reads"], S["molecules"]) for S in view["pileup"]["sections"]}
    # Major: 70 reads (C), M0 on rows 0 and 1. Minor: 30 reads (G). Unk: 10
    # reads, M0 again on row 100.
    assert by_label == {"hap_major": (70, 69), "hap_minor": (30, 30), "unk": (10, 10)}
    assert view["n_molecules"] == 108 < sum(m for _, m in by_label.values())
    # The histogram panels show the same reads per label.
    totals = np.sum([v for _, v in view["largest_run_labels"]], axis=0).tolist()
    assert totals == [70, 30, 10]


def test_without_phasing_the_one_section_counts_every_molecule(tmp_path):
    db, _ = _snv_panel(tmp_path)
    view = locus_view_data("L1", ExtractedPanel.load_db(db).loci["L1"], "S")
    [section] = view["pileup"]["sections"]
    assert (section["label"], section["reads"], section["molecules"]) == (None, 110, 109)
    assert section["molecules"] == view["n_molecules"]
