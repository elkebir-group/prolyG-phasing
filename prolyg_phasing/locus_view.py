"""A locus viewer that reads ``panel.db`` and, if given, the saved phasing result.

:func:`make_server` starts an HTTP server bound to 127.0.0.1. It answers
``GET /`` with the page (the template ``locus_view.html``, plain CSS and
JS), ``GET /loci`` with the locus list, and ``GET /locus`` with one locus
of one sample, built on request by :func:`locus_view_data`. It writes no
files. It reads the panels, not the BAMs, so dropped reads appear only as
counts.

Per locus the page shows the flanks and the reference tract on the G strand,
the tracts with their read and molecule counts, the largest G run per read,
a pile-up, and the base counts over the full flanking width of each side.

A panel can come with the output directory of ``prolyg-phasing phase`` for
it. The viewer then reads ``snv_calls.tsv``, ``haplotype_calls.tsv`` and
``locus_summary.tsv`` (:class:`PhasingIndex`): the pile-up gets one column
per SNV the phasing found, and its rows split by the read family's phase
label. The viewer makes no SNV or phasing decision of its own.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import threading
from array import array
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

# Default page setting: the flanking bases shown next to each flank.
NEAR_BASES = 6

# Characters counted at each flanking position. "." (no read base) is not
# counted.
FLANKING_BASES = "ACGTN"

# Phase labels of haplotype_calls.tsv, and their codes in the index.
PHASE_LABELS = ("hap_major", "hap_minor", "unk")

# The locus_summary.tsv fields the page shows.
SUMMARY_FIELDS = ("hap_major_profile", "hap_minor_profile", "n_snvs_phased",
                  "n_snvs_dropped_low_linkage", "reads_major", "reads_minor", "reads_unk")

_COUNTER_PREFIXES = ("n_alignments_", "n_read_pairs")


def panel_sample(panel: ExtractedPanel) -> str:
    """The sample name of a panel: the file stem of the BAM its provenance names.

    The provenance holds no sample field of its own.
    """
    return Path(panel.provenance.bam_path).stem


# ---------------------------------------------------------------------------
# The saved phasing result
# ---------------------------------------------------------------------------


def _row_digest() -> hashlib.blake2b:
    return hashlib.blake2b(digest_size=16)


def _add_row(digest: hashlib.blake2b, mi: str, strand: str, count: str | int) -> None:
    digest.update(f"{mi}\t{strand}\t{count}\n".encode())


@dataclasses.dataclass
class PhasingIndex:
    """The saved phasing result of one panel, read from its output directory.

    ``haplotype_calls.tsv`` holds one row per panel row, in the panel's row
    order within each locus: the same (mi, strand) can occur on several
    rows (one per tract and flanking string of the read family). So the
    index joins a label to a panel row by its position within the locus,
    and checks the join by a digest of every row's (mi, strand, count).

    Attributes
    ----------
    directory : str
    snvs : dict of str to list of dict
        Per locus, the rows of ``snv_calls.tsv`` (phased and dropped).
    summary : dict of str to dict
        Per locus, the `SUMMARY_FIELDS` of ``locus_summary.tsv``.
    labels : dict of str to ndarray
        Per locus, each row's phase label as an index into `PHASE_LABELS`
        (uint8), in row order.
    digests : dict of str to bytes
        Per locus, the digest of its rows' (mi, strand, count) in row order.
    """

    directory: str
    snvs: dict[str, list[dict]]
    summary: dict[str, dict]
    labels: dict[str, np.ndarray]
    digests: dict[str, bytes]

    @classmethod
    def read(cls, directory: str | Path) -> PhasingIndex:
        """Read the three TSVs; ``haplotype_calls.tsv`` streams into compact arrays.

        Raises ValueError naming the file when a label is unknown.
        """
        d = Path(directory)
        snvs: dict[str, list[dict]] = {}
        with open(d / "snv_calls.tsv", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                snvs.setdefault(row["locus_id"], []).append({
                    "segment": row["segment"], "string_index": int(row["string_index"]),
                    "ref_pos": int(row["ref_pos"]), "allele_major": row["allele_major"],
                    "allele_minor": row["allele_minor"],
                    "vaf_minor": float(row["vaf_minor"]), "coverage": int(row["coverage"]),
                    "status": row["status"], "is_anchor": row["is_anchor"] == "True",
                    "hap_major_base": row["hap_major_base"],
                    "hap_minor_base": row["hap_minor_base"]})
        with open(d / "locus_summary.tsv", newline="") as f:
            summary = {row["locus_id"]: {k: row[k] for k in SUMMARY_FIELDS}
                       for row in csv.DictReader(f, delimiter="\t")}
        code = {label: i for i, label in enumerate(PHASE_LABELS)}
        raw: dict[str, array] = {}
        digests: dict[str, hashlib.blake2b] = {}
        path = d / "haplotype_calls.tsv"
        with open(path, newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            at = [header.index(name) for name in ("locus_id", "mi", "strand", "count",
                                                  "phase_label")]
            for row in reader:
                locus, mi, strand, count, label = (row[i] for i in at)
                label_code = code.get(label)
                if label_code is None:
                    raise ValueError(f"{path}: unknown phase label {label!r}")
                if locus not in raw:
                    raw[locus], digests[locus] = array("B"), _row_digest()
                raw[locus].append(label_code)
                _add_row(digests[locus], mi, strand, count)
        labels = {locus: np.frombuffer(a, dtype=np.uint8) for locus, a in raw.items()}
        return cls(str(d), snvs, summary, labels,
                   {locus: h.digest() for locus, h in digests.items()})

    def labels_of(self, locus_id: str, n_rows: int) -> np.ndarray:
        """The phase label code of each of the `n_rows` panel rows of a locus."""
        labels = self.labels.get(locus_id, np.zeros(0, dtype=np.uint8))
        if len(labels) != n_rows:
            raise RuntimeError(f"{self.directory}: {locus_id} has {len(labels)} labels for "
                               f"{n_rows} panel rows")
        return labels

    def check(self, panel: ExtractedPanel, panel_path: str | Path) -> None:
        """Raise ValueError unless every locus's rows match the panel's, in order.

        A locus matches when its labeled rows carry the same (mi, strand,
        count) as the panel's rows, row by row. The message names the panel,
        the phasing directory and the first loci that differ. Decodes every
        locus of the panel once.
        """
        empty = _row_digest().digest()
        differ = sorted(set(self.labels) - set(panel.loci))
        for locus_id in panel.loci:
            locus = panel.loci[locus_id]
            digest = _row_digest()
            for mi, strand, count in zip(locus.mi, locus.strand, locus.count, strict=True):
                _add_row(digest, mi, strand, int(count))
            if digest.digest() != self.digests.get(locus_id, empty):
                differ.append(locus_id)
            if len(differ) >= 5:
                break
        if differ:
            raise ValueError(f"the phasing result {self.directory} does not belong to the "
                             f"panel {panel_path}: the read families differ at "
                             f"{', '.join(differ[:5])}")


# ---------------------------------------------------------------------------
# The data of one locus
# ---------------------------------------------------------------------------


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


def _columns(width: int, snv_indices: set[int], near: int, *, upstream: bool) -> list[int]:
    """The flanking indices of one side's pile-up columns, in drawing order.

    The `near` positions next to the flank, and every SNV position. Upstream,
    index 0 is the farthest base, so the order is ascending and the flank
    follows; downstream, index 0 is the nearest base.
    """
    near_set = set(range(max(width - near, 0), width)) if upstream \
        else set(range(min(near, width)))
    return sorted(near_set | {i for i in snv_indices if 0 <= i < width})


def locus_view_data(locus_id: str, locus: ExtractedLocus, sample: str, *,
                    near: int = NEAR_BASES, phasing: PhasingIndex | None = None) -> dict:
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
    near : int
        The flanking bases shown next to each flank.
    phasing : dict or None
        None without a phasing result. Else ``summary`` (the
        `SUMMARY_FIELDS` of the locus) and ``snvs`` (its SNV rows).
    flanking : dict
        ``up`` and ``dn``: ``start`` (reference coordinate of index 0, -1
        with no flanking bases), ``width``, ``counts`` (per index, the reads
        with each base of `FLANKING_BASES`) and ``snv`` (the indices of the
        phasing's SNVs on this side). Upstream index 0 is the base farthest
        from the flank; downstream index 0 is the nearest.
    pileup : dict
        ``up`` and ``dn``: the columns, in drawing order, each with
        ``index``, ``position``, ``counts``, ``coverage`` and, at an SNV,
        ``snv`` (its row of ``snv_calls.tsv``). ``rows``: one per distinct
        (tract, phase label, the read's own base or "." at every shown
        column), with ``tract``, ``label`` (None without phasing),
        ``reads``, ``up`` and ``dn`` (the bases at the side's columns, one
        character each). Sorted by reads. ``n_rows`` and
        ``reads_not_drawn`` count the rows past `PILEUP_ROWS`.
    """
    counters = {f.name: int(getattr(locus, f.name))
                for f in dataclasses.fields(locus)
                if f.name.startswith(_COUNTER_PREFIXES)}
    counters["n_reads"] = int(locus.n_reads)
    count = np.asarray(locus.count, dtype=np.int64)
    seq = np.asarray(locus.seq, dtype=str)

    # Tracts, with their reads and molecules (distinct MI).
    tracts, tract_reads, tract_of_row = _summed(seq, count)
    mi_id = np.unique(np.asarray(locus.mi, dtype=str), return_inverse=True)[1].ravel()
    span = len(mi_id) + 1
    pairs = np.unique(tract_of_row.astype(np.int64) * span + mi_id)
    tract_molecules = np.bincount(pairs // span, minlength=len(tracts))
    order = np.lexsort((tracts, -tract_reads))
    tract_rows = [[str(tracts[i]), int(tract_reads[i]), int(tract_molecules[i])]
                  for i in order[:TRACT_ROWS]]
    largest: dict[int, int] = {}
    for tract, reads in zip(tracts, tract_reads, strict=True):
        run = max(parse_run_lengths(str(tract)))
        largest[run] = largest.get(run, 0) + int(reads)

    # The phasing result of this locus, if any.
    snvs = phasing.snvs.get(locus_id, []) if phasing is not None else []
    row_label = (phasing.labels_of(locus_id, len(count)).astype(np.int64)
                 if phasing is not None else np.full(len(count), -1, dtype=np.int64))

    # Base counts over each side's flanking width, and the pile-up columns.
    up_w = int(locus.flanking_up_width)
    matrix = _flanking_matrix(locus)
    fid = np.asarray(locus.flanking_id, dtype=np.int64)
    weights = np.bincount(fid, weights=count, minlength=len(matrix)).astype(np.int64)
    orient = locus.ref_orientation
    sides = {"up": (matrix[:, :up_w], int(locus.flanking_up_ref_pos_start), True),
             "dn": (matrix[:, up_w + 1:], int(locus.flanking_dn_ref_pos_start), False)}
    flanking, columns = {}, {}
    for name, (block, start, upstream) in sides.items():
        counts = _base_counts(block, weights)
        snv_at = {s["string_index"]: s for s in snvs if s["segment"] == name}
        flanking[name] = {"start": start, "width": block.shape[1], "counts": counts.tolist(),
                          "snv": sorted(i for i in snv_at if i < block.shape[1])}
        cols = []
        for i in _columns(block.shape[1], set(snv_at), near, upstream=upstream):
            cov = int(counts[i].sum())
            col = {"index": i, "position": start + i if orient == "+" else start - i,
                   "counts": counts[i].tolist(), "coverage": cov}
            if i in snv_at:
                col["snv"] = snv_at[i]
            cols.append(col)
        columns[name] = cols

    # Pile-up rows: the reads with one tract, one phase label and the same
    # own base ("." for none) at every shown column, with their read count.
    def shown(name):
        block = sides[name][0]
        idx = [c["index"] for c in columns[name]]
        cells = block[np.ix_(fid, idx)] if len(fid) else np.zeros((0, len(idx)), np.uint8)
        return np.array([row.tobytes().decode("ascii") for row in cells] or [""], dtype=str)

    up_cells, dn_cells = shown("up"), shown("dn")
    row_key = seq
    for part in (row_label.astype(str), up_cells[:len(seq)], dn_cells[:len(seq)]):
        row_key = np.char.add(np.char.add(row_key, "\t"), part)
    keys, key_reads, _ = _summed(row_key, count)
    key_order = np.lexsort((keys, -key_reads))
    rows = []
    for k in key_order[:PILEUP_ROWS]:
        tract, label, up, dn = str(keys[k]).split("\t")
        rows.append({"tract": tract, "reads": int(key_reads[k]), "up": up, "dn": dn,
                     "label": PHASE_LABELS[int(label)] if int(label) >= 0 else None})

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
        "near": near,
        "phasing": None if phasing is None else {
            "summary": phasing.summary.get(locus_id), "snvs": snvs},
        "flanking": flanking,
        "pileup": {"up": columns["up"], "dn": columns["dn"], "rows": rows,
                   "n_rows": len(keys),
                   "reads_not_drawn": int(key_reads[key_order[PILEUP_ROWS:]].sum())},
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


def locus_list(panels: Sequence[ExtractedPanel],
               phasings: Sequence[PhasingIndex | None]) -> dict:
    """The locus list: every locus of the panels, with per-sample fields.

    Read from the panels' metadata columns, without decoding any locus's
    rows. Per sample: ``n_reads``, ``status``, ``largest_run`` (the largest
    observed G run), ``no_position`` (a flank side with no reference
    position) and ``snvs`` (``{"phased": n, "dropped": m}`` from the
    phasing result's ``snv_calls.tsv``, None when no phasing result was
    given), or None when the panel does not hold the locus.
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

    def entry(c, phasing, name):
        if name not in c["n_reads"]:
            return None
        out = {field: c[field][name] for field in fields}
        if phasing is None:
            out["snvs"] = None
        else:
            status = [s["status"] for s in phasing.snvs.get(name, [])]
            out["snvs"] = {"phased": status.count("phased"),
                           "dropped": len(status) - status.count("phased")}
        return out

    entries = [{"locus": name,
                "samples": [entry(c, ph, name) for c, ph in zip(columns, phasings, strict=True)]}
               for name in names]
    return {"samples": [{"sample": panel_sample(p),
                         "flank_table_sample": p.provenance.flank_table_sample,
                         "phasing": None if ph is None else ph.directory}
                        for p, ph in zip(panels, phasings, strict=True)],
            "loci": entries, "pileup_rows": PILEUP_ROWS, "near": NEAR_BASES,
            "flanking_bases": FLANKING_BASES}


class _Handler(BaseHTTPRequestHandler):
    panels: _Panels
    phasings: list[PhasingIndex | None]
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
            near = int(query.get("near", [NEAR_BASES])[0])
        except (KeyError, ValueError, IndexError):
            self._send(400, b"Give sample (an index), locus, and a number of near bases.\n",
                       "text/plain; charset=utf-8")
            return
        if not 0 <= s < len(panels) or name not in panels[s].loci:
            self._send(404, f"No locus {name} in sample {s}.\n".encode(),
                       "text/plain; charset=utf-8")
            return
        panel = panels[s]
        data = locus_view_data(name, panel.loci[name], panel_sample(panel),
                               near=min(max(near, 0), 50), phasing=self.phasings[s])
        self._send(200, json.dumps(data, separators=(",", ":")).encode(), "application/json")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args) -> None:  # noqa: A002 (the base method's name)
        pass


def make_server(panels: Sequence[tuple[str | Path, str | Path | None]],
                port: int) -> ThreadingHTTPServer:
    """The viewer's HTTP server on 127.0.0.1:`port` (0: a free port), not yet serving.

    `panels` are ``(panel.db, phasing directory or None)`` pairs. Each
    phasing directory is read once, here, and checked against its panel
    (:meth:`PhasingIndex.check`); a pair that does not belong together
    raises ValueError. The locus list is read once, here. Call
    ``serve_forever()`` on the server.
    """
    paths = [str(p) for p, _ in panels]
    opened = _Panels(paths)
    phasings = []
    for (path, directory), panel in zip(panels, opened.get(), strict=True):
        if directory is None:
            phasings.append(None)
            continue
        phasing = PhasingIndex.read(directory)
        phasing.check(panel, path)
        phasings.append(phasing)
    page = resources.files("prolyg_phasing").joinpath("locus_view.html").read_bytes()
    listing = json.dumps(locus_list(opened.get(), phasings), separators=(",", ":")).encode()
    handler = type("LocusViewHandler", (_Handler,),
                   {"panels": opened, "phasings": phasings, "page": page, "listing": listing})
    return ThreadingHTTPServer(("127.0.0.1", port), handler)
