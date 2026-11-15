"""A locus viewer that reads only ``panel.db``, served on the local machine.

:func:`make_server` starts an HTTP server bound to 127.0.0.1. It answers
``GET /`` with the page (the template ``locus_view.html``, plain CSS and
JS), ``GET /loci`` with the locus list, and ``GET /locus`` with one locus
of one sample, built on request by :func:`locus_view_data`. It writes no
files. It reads the panels, not the BAMs, so dropped reads appear only as
counts.

Per locus the page shows the flanks and the reference tract on the G strand,
the tracts with their read and molecule counts, the largest G run per read,
a pile-up, and the base counts over the full flanking width of each side.

The pile-up shows the positions that matter for phasing. On each side its
columns are the bases next to the flank and every flanking position where
the second most common base has at least a set share of the covered reads
(a heterozygous position). A row is a group of reads with one tract and one
set of bases at the heterozygous columns; see :func:`_pileup_groups`.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from prolyg_phasing.io.panel import ExtractedLocus, ExtractedPanel, parse_run_lengths

# Pile-up rows drawn per locus and sample; the rest are counted.
PILEUP_ROWS = 150

# Tract rows kept per locus and sample; the rest are counted.
TRACT_ROWS = 200

# Default page settings: the share of the covered reads that the second most
# common base must reach at a heterozygous position, and the bases shown
# next to each flank.
HET_LIMIT = 0.20
NEAR_BASES = 6

# Characters counted at each flanking position. "." (no read base) is not
# counted.
FLANKING_BASES = "ACGTN"

# The empty cell of a flanking string: no read base there.
_EMPTY = ord(".")

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


def _base_counts(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per column, the weighted count of each base of `FLANKING_BASES`: (width, 5)."""
    return np.stack([weights @ (matrix == ord(b)) for b in FLANKING_BASES],
                    axis=1).astype(np.int64).reshape(matrix.shape[1], len(FLANKING_BASES))


def heterozygous(counts: np.ndarray, limit: float) -> np.ndarray:
    """The positions where the second most common base has at least `limit` of the coverage.

    `counts` is (positions, bases). The coverage is the reads with a base at
    the position.
    """
    if len(counts) == 0:
        return np.zeros(0, dtype=bool)
    coverage = counts.sum(axis=1)
    second = np.sort(counts, axis=1)[:, -2]
    return (coverage > 0) & (second >= limit * coverage)


def _columns(width: int, het: np.ndarray, near: int, *, upstream: bool) -> list[int]:
    """The flanking indices of one side's pile-up columns, in drawing order.

    The `near` positions next to the flank, and every heterozygous position.
    Upstream, index 0 is the farthest base, so the order is ascending and the
    flank follows; downstream, index 0 is the nearest base.
    """
    near_set = set(range(max(width - near, 0), width)) if upstream \
        else set(range(min(near, width)))
    return sorted(near_set | set(np.flatnonzero(het).tolist()))


def _column_view(i: int, counts: np.ndarray, het: bool, orient: str, start: int) -> dict:
    cov = int(counts.sum())
    order = np.argsort(-counts, kind="stable")
    view = {"index": i, "position": start + i if orient == "+" else start - i,
            "counts": counts.tolist(), "coverage": cov, "het": het,
            "majority": FLANKING_BASES[order[0]] if cov else ""}
    if het:
        view["bases"] = [[FLANKING_BASES[k], int(counts[k]) / cov] for k in order[:2]]
    return view


def _pileup_groups(tract: np.ndarray, het_bases: np.ndarray, near_bases: np.ndarray,
                   reads: np.ndarray) -> list[dict]:
    """Group the reads into pile-up rows.

    Each item is a distinct (tract, flanking string) with its reads:
    `het_bases` (items, het columns) and `near_bases` (items, near columns)
    hold its bytes, "." for no base. A group is one tract and one set of
    bases at the heterozygous columns.

    The items with a base at every heterozygous column form the groups.
    Then each item with an empty heterozygous cell joins the group with the
    same tract whose bases agree at every column where the item has a base;
    of several, the one with the most reads. An item that matches no group
    forms its own group, shared with the items of the same pattern.

    Returns, per group: ``tract``, ``het`` (bytes), ``reads``, ``joined``
    (reads that joined with empty cells) and ``near_counts`` (per near
    column, the reads with each base of `FLANKING_BASES`).
    """
    complete = (het_bases != _EMPTY).all(axis=1)
    groups: dict[tuple[str, bytes], dict] = {}
    by_tract: dict[str, list[tuple[str, bytes]]] = {}

    def group(key):
        if key not in groups:
            groups[key] = {"tract": key[0], "het": key[1], "reads": 0, "joined": 0,
                           "near_counts": np.zeros((near_bases.shape[1], len(FLANKING_BASES)),
                                                   dtype=np.int64)}
        return groups[key]

    def add(g, item, joined):
        g["reads"] += int(reads[item])
        g["joined"] += int(reads[item]) if joined else 0
        for c, b in enumerate(near_bases[item]):
            if b != _EMPTY:
                g["near_counts"][c, FLANKING_BASES.index(chr(b))] += int(reads[item])

    for item in np.flatnonzero(complete):
        key = (str(tract[item]), het_bases[item].tobytes())
        if key not in groups:
            by_tract.setdefault(key[0], []).append(key)
        add(group(key), item, joined=False)
    full_reads = {key: g["reads"] for key, g in groups.items()}
    for item in np.flatnonzero(~complete):
        t, bases = str(tract[item]), het_bases[item]
        has = bases != _EMPTY
        fits = [key for key in by_tract.get(t, [])
                if (np.frombuffer(key[1], dtype=np.uint8)[has] == bases[has]).all()]
        if fits:
            add(groups[max(fits, key=lambda k: (full_reads[k], k[1]))], item, joined=True)
        else:
            add(group((t, bases.tobytes())), item, joined=True)
    return list(groups.values())


def locus_view_data(locus_id: str, locus: ExtractedLocus, sample: str, *,
                    het_limit: float = HET_LIMIT, near: int = NEAR_BASES) -> dict:
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
    largest_run : list of [int, int]
        ``[largest G run, reads]`` over the kept reads.
    settings : dict
        ``het_limit`` and ``near``.
    flanking : dict
        ``up`` and ``dn``: ``start`` (reference coordinate of index 0, -1
        with no flanking bases), ``width``, ``counts`` (per index, the reads
        with each base of `FLANKING_BASES`) and ``het`` (per index, whether
        the position is heterozygous). Upstream index 0 is the base farthest
        from the flank; downstream index 0 is the nearest.
    pileup : dict
        ``up`` and ``dn``: the columns, in drawing order, each with
        ``index``, ``position``, ``counts``, ``coverage``, ``het``,
        ``majority`` and, at a heterozygous column, ``bases`` (the two most
        common bases and their shares). ``rows``: per group, ``tract``,
        ``reads``, ``joined``, ``up`` and ``dn`` (per column, the shown base
        and the group's base counts there; the counts are None at a
        heterozygous column). ``n_rows`` and ``reads_not_drawn`` count the
        groups past `PILEUP_ROWS`.
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

    # Base counts and heterozygous positions over each side's flanking width.
    up_w = int(locus.flanking_up_width)
    matrix = _flanking_matrix(locus)
    fid = np.asarray(locus.flanking_id, dtype=np.int64)
    weights = np.bincount(fid, weights=count, minlength=len(matrix)).astype(np.int64)
    orient = locus.ref_orientation
    sides = {
        "up": (matrix[:, :up_w], int(locus.flanking_up_ref_pos_start), True),
        "dn": (matrix[:, up_w + 1:], int(locus.flanking_dn_ref_pos_start), False),
    }
    flanking, columns = {}, {}
    for name, (block, start, upstream) in sides.items():
        counts = _base_counts(block, weights)
        het = heterozygous(counts, het_limit)
        flanking[name] = {"start": start, "width": block.shape[1],
                          "counts": counts.tolist(), "het": het.tolist()}
        columns[name] = [_column_view(i, counts[i], bool(het[i]), orient, start)
                         for i in _columns(block.shape[1], het, near, upstream=upstream)]

    # Pile-up groups over the distinct (tract, flanking string) items.
    item_keys, item_reads, _ = _summed(
        np.char.add(np.char.add(seq, "\t"), fid.astype(str)), count)
    item_tract = np.array([str(k).split("\t")[0] for k in item_keys], dtype=object)
    item_fid = np.array([int(str(k).split("\t")[1]) for k in item_keys], dtype=np.int64)

    def cells(name, het_only):
        block = sides[name][0]
        idx = [c["index"] for c in columns[name] if c["het"] == het_only]
        return block[np.ix_(item_fid, idx)] if len(item_fid) else \
            np.zeros((0, len(idx)), dtype=np.uint8)

    het_cells = np.concatenate([cells("up", True), cells("dn", True)], axis=1)
    near_cells = np.concatenate([cells("up", False), cells("dn", False)], axis=1)
    groups = _pileup_groups(item_tract, het_cells, near_cells, item_reads)
    groups.sort(key=lambda g: (-g["reads"], g["tract"], g["het"]))

    def row(g) -> dict:
        het_iter, near_iter = iter(g["het"]), iter(g["near_counts"])
        out = {"tract": g["tract"], "reads": g["reads"], "joined": g["joined"]}
        for name in ("up", "dn"):
            side = []
            for c in columns[name]:
                if c["het"]:
                    b = next(het_iter)
                    side.append([chr(b) if b != _EMPTY else "", None])
                else:
                    cnt = next(near_iter)
                    best = int(np.argmax(cnt))
                    side.append([FLANKING_BASES[best] if cnt[best] else "", cnt.tolist()])
            out[name] = side
        return out

    flanks = None
    if locus.upstream_flank is not None:
        flanks = {"upstream": _flank_view(locus.upstream_flank),
                  "downstream": _flank_view(locus.downstream_flank)}
    return {
        "locus": locus_id, "sample": sample,
        "chrom": locus.chrom, "bed_start": int(locus.bed_start), "bed_end": int(locus.bed_end),
        "bed_name": locus.bed_name, "ref_orientation": orient,
        "reference_window_status": locus.reference_window_status,
        "reference_tract": locus.reference_tract,
        "reference_run_lengths": [int(x) for x in locus.reference_run_lengths],
        "flanks": flanks,
        "counters": counters,
        "tracts": tract_rows,
        "n_tracts": len(tracts),
        "tract_reads_not_shown": int(tract_reads[order[TRACT_ROWS:]].sum()),
        "largest_run": [[k, v] for k, v in sorted(largest.items())],
        "settings": {"het_limit": het_limit, "near": near},
        "flanking": flanking,
        "pileup": {"up": columns["up"], "dn": columns["dn"],
                   "rows": [row(g) for g in groups[:PILEUP_ROWS]],
                   "n_rows": len(groups),
                   "reads_not_drawn": sum(g["reads"] for g in groups[PILEUP_ROWS:])},
    }


# ---------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------


class _Panels:
    """The panels, opened read-only once per server thread.

    A SQLite connection must not cross threads, so each thread loads its
    own `ExtractedPanel` of each path.
    """

    def __init__(self, paths: Sequence[str | Path]):
        self.paths = [str(p) for p in paths]
        self._local = threading.local()

    def get(self) -> list[ExtractedPanel]:
        panels = getattr(self._local, "panels", None)
        if panels is None:
            panels = self._local.panels = [ExtractedPanel.load_db(p) for p in self.paths]
        return panels


def _lacks_position(text: str | None) -> bool:
    """Whether a stored flank has no reference positions (None: no flank at all)."""
    return text is not None and not json.loads(text)["ref_positions"]


def locus_list(panels: Sequence[ExtractedPanel]) -> dict:
    """The locus list: every locus of the panels, with per-sample fields.

    Read from the panels' metadata columns, without decoding any locus's
    rows. Per sample: ``n_reads``, ``status``, ``largest_run`` (the largest
    observed G run) and ``no_position`` (a flank side with no reference
    position), or None when the panel does not hold the locus.
    """
    columns = []
    for panel in panels:
        up, dn = panel.locus_column("upstream_flank"), panel.locus_column("downstream_flank")
        columns.append({
            "n_reads": panel.locus_column("n_reads"),
            "status": panel.locus_column("reference_window_status"),
            "largest_run": panel.locus_column("max_observed_run_length"),
            "no_position": {k: _lacks_position(up[k]) or _lacks_position(dn[k]) for k in up},
        })
    names: dict[str, None] = {}
    for panel in panels:
        for name in panel.loci:
            names.setdefault(name, None)
    fields = ("n_reads", "status", "largest_run", "no_position")
    entries = [{"locus": name,
                "samples": [None if name not in c["n_reads"] else
                            {field: c[field][name] for field in fields} for c in columns]}
               for name in names]
    return {"samples": [{"sample": panel_sample(p),
                         "flank_table_sample": p.provenance.flank_table_sample}
                        for p in panels],
            "loci": entries, "pileup_rows": PILEUP_ROWS,
            "het_limit": HET_LIMIT, "near": NEAR_BASES,
            "flanking_bases": FLANKING_BASES}


class _Handler(BaseHTTPRequestHandler):
    panels: _Panels
    page: bytes
    listing: bytes

    def do_GET(self) -> None:  # noqa: N802 (the http.server method name)
        url = urlparse(self.path)
        if url.path == "/":
            self._send(200, self.page, "text/html; charset=utf-8")
        elif url.path == "/loci":
            self._send(200, self.listing, "application/json")
        elif url.path == "/locus":
            self._locus(parse_qs(url.query))
        else:
            self._send(404, b"Not found.\n", "text/plain; charset=utf-8")

    def _locus(self, query: dict) -> None:
        panels = self.panels.get()
        try:
            s = int(query["sample"][0])
            name = query["locus"][0]
            het_limit = float(query.get("het", [HET_LIMIT])[0])
            near = int(query.get("near", [NEAR_BASES])[0])
        except (KeyError, ValueError, IndexError):
            self._send(400, b"Give sample (an index), locus, and number settings.\n",
                       "text/plain; charset=utf-8")
            return
        if not 0 <= s < len(panels) or name not in panels[s].loci:
            self._send(404, f"No locus {name} in sample {s}.\n".encode(),
                       "text/plain; charset=utf-8")
            return
        panel = panels[s]
        data = locus_view_data(name, panel.loci[name], panel_sample(panel),
                               het_limit=min(max(het_limit, 0.0), 0.5),
                               near=min(max(near, 0), 50))
        self._send(200, json.dumps(data, separators=(",", ":")).encode(), "application/json")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args) -> None:  # noqa: A002 (the base method's name)
        pass


def make_server(panel_paths: Sequence[str | Path], port: int) -> ThreadingHTTPServer:
    """The viewer's HTTP server on 127.0.0.1:`port` (0: a free port), not yet serving.

    Call ``serve_forever()`` on it. The locus list is read once, here.
    """
    panels = _Panels(panel_paths)
    page = resources.files("prolyg_phasing").joinpath("locus_view.html").read_bytes()
    listing = json.dumps(locus_list(panels.get()), separators=(",", ":")).encode()
    handler = type("LocusViewHandler", (_Handler,),
                   {"panels": panels, "page": page, "listing": listing})
    return ThreadingHTTPServer(("127.0.0.1", port), handler)
