"""A locus viewer that reads only ``panel.db``: a directory of small files and one page.

The page shows each locus as the fit and the phasing see it: the flanks and
the reference tract on the G strand, the tracts with their read and molecule
counts, the largest G run per read, a pile-up of the tracts between the
flanks with six flanking bases on each side, and the base counts over the
full flanking width of each side. It reads the panel, not the BAM, so
dropped reads appear only as counts.

:func:`locus_view_data` turns one `ExtractedLocus` into the plain dict that
the page draws. :func:`write_locus_view` writes the directory: ``index.html``
(the page template ``locus_view.html``, plain CSS and JS), ``loci.json`` (the
locus list) and one JSON file per sample and locus, which the page fetches
when the locus is opened. The page needs an HTTP server, because browsers
block ``fetch`` from ``file://``.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from importlib import resources
from pathlib import Path

import numpy as np

from prolyg_phasing.io.panel import ExtractedLocus, ExtractedPanel, parse_run_lengths

# Pile-up rows drawn per locus and sample; the rest are counted.
PILEUP_ROWS = 150

# Tract rows kept per locus and sample; the rest are counted.
TRACT_ROWS = 200

# Flanking bases drawn outside each flank in the pile-up.
PILEUP_FLANKING = 6

# Characters counted at each flanking position. "." (no read base) is not
# counted.
FLANKING_BASES = "ACGTN"

_COUNTER_PREFIXES = ("n_alignments_", "n_read_pairs")


def panel_sample(panel: ExtractedPanel) -> str:
    """The sample name of a panel: the file stem of the BAM its provenance names.

    The provenance holds no sample field of its own.
    """
    return Path(panel.provenance.bam_path).stem


def _flank_view(flank) -> dict:
    if flank.has_position:
        return {"bases": flank.bases, "status": "recovered",
                "outer_position": flank.outer_position,
                "run_side_position": flank.run_side_position}
    return {"bases": flank.bases, "status": "no_reference_position",
            "outer_position": None, "run_side_position": None}


def _summed(keys: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distinct keys, their summed weights, and each row's index among them."""
    distinct, index = np.unique(keys, return_inverse=True)
    index = index.ravel()
    sums = np.bincount(index, weights=weights, minlength=len(distinct)).astype(np.int64)
    return distinct, sums, index


def _flanking_matrix(locus: ExtractedLocus) -> np.ndarray:
    """The flanking strings as a (n_flanking, up_width + 1 + dn_width) byte matrix."""
    width = int(locus.flanking_up_width) + 1 + int(locus.flanking_dn_width)
    if len(locus.flanking_seq) == 0:
        return np.zeros((0, width), dtype=np.uint8)
    text = "".join(str(s) for s in locus.flanking_seq).encode("ascii")
    return np.frombuffer(text, dtype=np.uint8).reshape(len(locus.flanking_seq), width)


def _base_counts(matrix: np.ndarray, weights: np.ndarray) -> list[list[int]]:
    """Per column, the weighted count of each base of `FLANKING_BASES`."""
    if matrix.shape[1] == 0:
        return []
    counts = np.stack([weights @ (matrix == ord(b)) for b in FLANKING_BASES], axis=1)
    return counts.astype(np.int64).tolist()


def locus_view_data(locus_id: str, locus: ExtractedLocus, sample: str) -> dict:
    """One locus of one sample as the plain dict the page draws.

    Keys
    ----
    locus, sample, chrom, bed_name : str
    bed_start, bed_end : int
        The BED interval, 0-based half-open.
    ref_orientation : str
        "+" or "-"; empty when the locus has no geometry.
    reference_window_status : str
    reference_tract : str
        The reference between the two run-side bases, G strand; empty when
        a flank has no reference position.
    reference_run_lengths : list of int
    flanks : dict or None
        ``{"upstream": side, "downstream": side}``, None when the locus has
        no flank. A side holds ``bases``, ``status`` ("recovered" or
        "no_reference_position"), ``outer_position`` and
        ``run_side_position`` (None with no position).
    counters : dict of str to int
        Every ``n_alignments_*`` and ``n_read_pairs*`` field, and ``n_reads``.
    tracts : list of [str, int, int]
        ``[tract, reads, molecules]`` for the `TRACT_ROWS` tracts with the
        most reads; molecules are distinct MI. Sorted by reads, then tract.
    n_tracts, tract_reads_not_shown : int
        The distinct tracts, and the reads of the tracts not kept.
    largest_run : list of [int, int]
        ``[largest G run, reads]`` over the kept reads.
    pileup : dict
        ``rows``: ``[upstream flanking, tract, downstream flanking, reads]``
        for the `PILEUP_ROWS` rows with the most reads. A row is a distinct
        tract with its `PILEUP_FLANKING` flanking bases on each side,
        outermost first upstream and nearest first downstream; "." is no
        read base. ``n_rows`` and ``reads_not_drawn`` count the rest.
    flanking : dict
        ``up`` and ``dn``: ``start`` (reference coordinate of index 0, -1
        with no flanking bases), ``width``, and ``counts``: per index, the
        reads with each base of `FLANKING_BASES`. Upstream index 0 is the
        base farthest from the flank; downstream index 0 is the nearest.
        Reference coordinates rise with the index on a "+" locus and fall
        on a "-" locus.
    """
    counters = {f.name: int(getattr(locus, f.name))
                for f in dataclasses.fields(locus)
                if f.name.startswith(_COUNTER_PREFIXES)}
    counters["n_reads"] = int(locus.n_reads)
    count = np.asarray(locus.count, dtype=np.int64)
    seq = np.asarray(locus.seq, dtype=str)

    # Tracts, with their reads and molecules (distinct MI).
    tracts, tract_reads, tract_of_row = _summed(seq, count)
    mi = np.unique(np.asarray(locus.mi, dtype=str), return_inverse=True)[1].ravel()
    span = len(mi) + 1
    pairs = np.unique(tract_of_row.astype(np.int64) * span + mi)
    tract_molecules = np.bincount(pairs // span, minlength=len(tracts))
    order = np.lexsort((tracts, -tract_reads))
    tract_rows = [[str(tracts[i]), int(tract_reads[i]), int(tract_molecules[i])]
                  for i in order[:TRACT_ROWS]]
    largest: dict[int, int] = {}
    for tract, reads in zip(tracts, tract_reads, strict=True):
        run = max(parse_run_lengths(str(tract)))
        largest[run] = largest.get(run, 0) + int(reads)

    # Pile-up rows: a tract with its flanking bases next to each flank.
    up_w, dn_w = int(locus.flanking_up_width), int(locus.flanking_dn_width)
    matrix = _flanking_matrix(locus)
    n = PILEUP_FLANKING
    near = np.full((len(matrix), 2 * n), ord("."), dtype=np.uint8)
    take_up, take_dn = min(n, up_w), min(n, dn_w)
    if take_up:
        near[:, n - take_up:n] = matrix[:, up_w - take_up:up_w]
    if take_dn:
        near[:, n:n + take_dn] = matrix[:, up_w + 1:up_w + 1 + take_dn]
    near_text = np.array([row.tobytes().decode("ascii") for row in near] or [""], dtype=str)
    fid = np.asarray(locus.flanking_id, dtype=np.int64)
    row_key = np.char.add(np.char.add(near_text[fid], "|"), seq) if len(fid) else seq
    row_keys, row_reads, _ = _summed(row_key, count)
    row_order = np.lexsort((row_keys, -row_reads))
    pile_rows = []
    for i in row_order[:PILEUP_ROWS]:
        sides, tract = str(row_keys[i]).split("|", 1)
        pile_rows.append([sides[:n], tract, sides[n:], int(row_reads[i])])

    # Base counts over the full flanking width of each side.
    weights = np.bincount(fid, weights=count, minlength=len(matrix)).astype(np.int64)
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
        "flanks": flanks,
        "counters": counters,
        "tracts": tract_rows,
        "n_tracts": len(tracts),
        "tract_reads_not_shown": int(tract_reads[order[TRACT_ROWS:]].sum()),
        "largest_run": [[k, v] for k, v in sorted(largest.items())],
        "pileup": {"rows": pile_rows, "n_rows": len(row_keys),
                   "reads_not_drawn": int(row_reads[row_order[PILEUP_ROWS:]].sum())},
        "flanking": {
            "up": {"start": int(locus.flanking_up_ref_pos_start), "width": up_w,
                   "counts": _base_counts(matrix[:, :up_w], weights)},
            "dn": {"start": int(locus.flanking_dn_ref_pos_start), "width": dn_w,
                   "counts": _base_counts(matrix[:, up_w + 1:], weights)},
        },
    }


def _all_loci(panels: Sequence[ExtractedPanel]) -> list[str]:
    """Every locus of the panels: the first panel's, then the others' new ones."""
    seen: dict[str, None] = {}
    for panel in panels:
        for locus in panel.loci:
            seen.setdefault(locus, None)
    return list(seen)


def write_locus_view(panel_paths: Sequence[str | Path], out_dir: str | Path,
                     loci: Sequence[str] | None = None) -> Path:
    """Write the viewer directory for `loci` (all loci of the panels when None).

    Writes ``index.html``, ``loci.json`` and ``data/<sample index>/<locus
    index>.json``, one locus at a time, so only one locus's data is in
    memory. A locus that a panel does not hold has no file for that sample.
    Returns the path of ``index.html``.
    """
    panels = [ExtractedPanel.load_db(path) for path in panel_paths]
    names = list(loci) if loci is not None else _all_loci(panels)
    out = Path(out_dir)
    for s in range(len(panels)):
        (out / "data" / str(s)).mkdir(parents=True, exist_ok=True)
    held = [set(panel.loci) for panel in panels]
    entries = []
    for i, name in enumerate(names):
        per_sample = []
        for s, panel in enumerate(panels):
            if name not in held[s]:
                per_sample.append(None)
                continue
            data = locus_view_data(name, panel.loci[name], panel_sample(panel))
            (out / "data" / str(s) / f"{i}.json").write_text(
                json.dumps(data, separators=(",", ":")), encoding="utf-8")
            flanks = data["flanks"]
            per_sample.append({
                "n_reads": data["counters"]["n_reads"], "n_tracts": data["n_tracts"],
                "status": data["reference_window_status"],
                "no_position": flanks is not None and any(
                    side["status"] != "recovered" for side in flanks.values())})
        entries.append({"locus": name, "file": i, "samples": per_sample})
    (out / "loci.json").write_text(json.dumps({
        "samples": [{"sample": panel_sample(panel), "panel": str(path),
                     "flank_table_sample": panel.provenance.flank_table_sample}
                    for path, panel in zip(panel_paths, panels, strict=True)],
        "loci": entries,
        "pileup_rows": PILEUP_ROWS, "pileup_flanking": PILEUP_FLANKING,
        "flanking_bases": FLANKING_BASES}, separators=(",", ":")), encoding="utf-8")
    page = resources.files("prolyg_phasing").joinpath("locus_view.html").read_text("utf-8")
    index = out / "index.html"
    index.write_text(page, encoding="utf-8")
    return index
