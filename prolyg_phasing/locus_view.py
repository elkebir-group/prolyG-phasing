"""A locus viewer that reads only ``panel.db``: one self-contained HTML page.

The page shows each requested locus as the fit and the phasing see it: the
flanks and their reference positions, the kept molecules, the tract-length
and largest-run counts, and every per-locus counter. It reads the panel, not
the BAM, so dropped reads appear only as counts.

:func:`locus_view_data` turns one `ExtractedLocus` into a plain dict;
:func:`write_locus_page` embeds those dicts as JSON in the page template
``locus_view.html`` (plain CSS and JS, no external assets). The page draws
every sequence in polyG orientation (the G strand).
"""

from __future__ import annotations

import dataclasses
import html
import json
from collections import Counter
from collections.abc import Sequence
from importlib import resources
from pathlib import Path

import numpy as np

from prolyg_phasing.io.panel import ExtractedLocus, ExtractedPanel, parse_run_lengths

# Molecule rows the page draws per locus and sample; the rest are counted.
ROWS_SHOWN = 150

_COUNTER_PREFIXES = ("n_alignments_", "n_read_pairs")


def panel_sample(panel: ExtractedPanel) -> str:
    """The sample name of a panel: the file stem of the BAM its provenance names.

    The provenance holds no sample field of its own; the BAM name is the
    sample's name in this cohort's files.
    """
    return Path(panel.provenance.bam_path).stem


def _flank_view(flank) -> dict:
    if flank.has_position:
        return {"bases": flank.bases, "status": "recovered",
                "ref_positions": list(flank.ref_positions),
                "outer_position": flank.outer_position,
                "run_side_position": flank.run_side_position}
    return {"bases": flank.bases, "status": "no_reference_position", "ref_positions": [],
            "outer_position": None, "run_side_position": None}


def locus_view_data(locus_id: str, locus: ExtractedLocus, sample: str) -> dict:
    """One locus of one sample as a plain dict for the page.

    Keys
    ----
    locus, sample : str
    chrom, bed_name : str
    bed_start, bed_end : int
        The BED interval, 0-based half-open.
    ref_orientation : str
        "+" or "-"; empty when the locus has no geometry.
    reference_window_status : str
    reference_tract : str
        The reference between the two run-side bases, polyG orientation;
        empty when a flank has no reference position.
    reference_run_lengths : list of int
    n_runs : int
    flanks : dict or None
        ``{"upstream": side, "downstream": side}``, None when the locus has
        no flank. A side holds ``bases``, ``status`` ("recovered" or
        "no_reference_position"), ``ref_positions`` (polyG read order,
        empty with no position), ``outer_position`` and
        ``run_side_position`` (None with no position).
    flanking : dict
        ``up_width``, ``dn_width``, ``up_ref_pos_start``,
        ``dn_ref_pos_start`` (-1 for a side with no position).
    counters : dict of str to int
        Every ``n_alignments_*`` and ``n_read_pairs*`` field, and
        ``n_reads``.
    tract_length_counts, largest_run_counts : list of [int, int]
        ``[value, reads]``, sorted by value, over the kept reads.
    molecules : list of dict
        The `ROWS_SHOWN` distinct (tract, upstream flanking string,
        downstream flanking string) rows with the most reads:
        ``tract``, ``up_flanking``, ``dn_flanking``, ``reads`` (summed read
        count) and ``molecules`` (distinct MI). Sorted by reads, most
        first, then by tract.
    n_rows : int
        The distinct rows.
    rows_not_shown, reads_not_shown : int
        The rows past the first `ROWS_SHOWN`, and their reads.
    """
    counters = {f.name: int(getattr(locus, f.name))
                for f in dataclasses.fields(locus)
                if f.name.startswith(_COUNTER_PREFIXES)}
    counters["n_reads"] = int(locus.n_reads)

    molecules, n_rows, rows_not_shown, reads_not_shown = _top_molecules(locus, ROWS_SHOWN)

    lengths: Counter = Counter()
    largest: Counter = Counter()
    for seq, count in zip(locus.seq, locus.count, strict=True):
        lengths[len(str(seq))] += int(count)
        largest[max(parse_run_lengths(str(seq)))] += int(count)

    flanks = None
    if locus.upstream_flank is not None:
        flanks = {"upstream": _flank_view(locus.upstream_flank),
                  "downstream": _flank_view(locus.downstream_flank)}
    return {
        "locus": locus_id, "sample": sample,
        "chrom": locus.chrom, "bed_start": int(locus.bed_start), "bed_end": int(locus.bed_end),
        "bed_name": locus.bed_name, "ref_orientation": locus.ref_orientation,
        "reference_window_status": locus.reference_window_status,
        "reference_tract": locus.reference_tract,
        "reference_run_lengths": [int(x) for x in locus.reference_run_lengths],
        "n_runs": int(locus.n_runs),
        "flanks": flanks,
        "flanking": {"up_width": int(locus.flanking_up_width),
                     "dn_width": int(locus.flanking_dn_width),
                     "up_ref_pos_start": int(locus.flanking_up_ref_pos_start),
                     "dn_ref_pos_start": int(locus.flanking_dn_ref_pos_start)},
        "counters": counters,
        "tract_length_counts": [[k, v] for k, v in sorted(lengths.items())],
        "largest_run_counts": [[k, v] for k, v in sorted(largest.items())],
        "molecules": molecules,
        "n_rows": n_rows,
        "rows_not_shown": rows_not_shown,
        "reads_not_shown": reads_not_shown,
    }


def _top_molecules(locus: ExtractedLocus, n_shown: int) -> tuple[list[dict], int, int, int]:
    """The `n_shown` distinct (tract, flanking) rows with the most reads, and the rest's size.

    A row's key is the tract and the flanking id; each flanking id names
    one (upstream, downstream) flanking pair. Rows are counted with numpy,
    and only the rows kept become dicts. Ties go to the smaller tract, then
    the smaller upstream and downstream flanking strings.

    Returns the kept rows, the number of distinct rows, and the rows and
    reads not kept.
    """
    if len(locus.seq) == 0:
        return [], 0, 0, 0
    tracts, tract_id = np.unique(np.asarray(locus.seq, dtype=str), return_inverse=True)
    sides = [str(s).split("|") for s in locus.flanking_seq]
    up_rank = np.unique([u for u, _ in sides], return_inverse=True)[1].ravel()
    dn_rank = np.unique([d for _, d in sides], return_inverse=True)[1].ravel()
    fid = np.asarray(locus.flanking_id, dtype=np.int64)
    key = tract_id.ravel().astype(np.int64) * len(sides) + fid
    keys, group = np.unique(key, return_inverse=True)
    group = group.ravel()
    reads = np.bincount(group, weights=locus.count, minlength=len(keys)).astype(np.int64)
    mi_id = np.unique(np.asarray(locus.mi, dtype=str), return_inverse=True)[1].ravel()
    families = np.unique(group.astype(np.int64) * (int(mi_id.max()) + 1) + mi_id)
    molecules = np.bincount(families // (int(mi_id.max()) + 1), minlength=len(keys))
    key_tract, key_fid = keys // len(sides), keys % len(sides)
    order = np.lexsort((dn_rank[key_fid], up_rank[key_fid], key_tract, -reads))
    top = order[:n_shown]
    rows = [{"tract": str(tracts[key_tract[i]]), "up_flanking": sides[key_fid[i]][0],
             "dn_flanking": sides[key_fid[i]][1], "reads": int(reads[i]),
             "molecules": int(molecules[i])} for i in top]
    rest = order[n_shown:]
    return rows, len(keys), len(rest), int(reads[rest].sum())


def locus_blocks(panels: Sequence[tuple[str, ExtractedPanel]], locus: str) -> list[dict]:
    """One :func:`locus_view_data` dict per panel for `locus`.

    `panels` are ``(path, panel)`` pairs. Raises KeyError naming the locus
    and the panel when a panel does not hold the locus.
    """
    blocks = []
    for path, panel in panels:
        try:
            extracted = panel.loci[locus]
        except KeyError:
            raise KeyError(f"{path}: the panel holds no locus {locus}") from None
        blocks.append(locus_view_data(locus, extracted, panel_sample(panel)))
    return blocks


def _json_script(element_id: str, value) -> str:
    text = json.dumps(value, separators=(",", ":")).replace("</", "<\\/")
    return f'<script type="application/json" id="{html.escape(element_id)}">{text}</script>\n'


def write_locus_page(panel_paths: Sequence[str | Path], loci: Sequence[str],
                     out: str | Path) -> Path:
    """Write the self-contained HTML page of `loci` over the panels; return its path.

    The page holds one JSON element per locus (id ``locus-<name>``) and one
    index element (id ``index``): per locus, each sample's read count and
    status. The file is written one locus at a time, so only one locus's
    data is in memory. The page draws a locus when it is opened.
    """
    panels = [(str(path), ExtractedPanel.load_db(path)) for path in panel_paths]
    page = resources.files("prolyg_phasing").joinpath("locus_view.html").read_text("utf-8")
    head, tail = page.split("__DATA__")
    title = html.escape(", ".join(loci[:4]) + (" …" if len(loci) > 4 else ""))
    index = []
    out = Path(out)
    with out.open("w", encoding="utf-8") as f:
        f.write(head.replace("__TITLE__", title))
        for locus in loci:
            blocks = locus_blocks(panels, locus)
            f.write(_json_script(f"locus-{locus}", {"locus": locus, "blocks": blocks}))
            index.append([locus, [[b["counters"]["n_reads"], b["reference_window_status"]]
                                  for b in blocks]])
        f.write(_json_script("index", {
            "samples": [{"sample": panel_sample(panel), "panel": path,
                         "flank_table_sample": panel.provenance.flank_table_sample}
                        for path, panel in panels],
            "loci": index, "rows_shown": ROWS_SHOWN}))
        f.write(tail)
    return out
