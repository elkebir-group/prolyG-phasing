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


def family_key(mi: str, strand: str) -> int:
    """The index key of one read family strand: a 64-bit hash of ``(mi, strand)``.

    The hash is stable within one process, which is the life of an index.
    Two different keys share a hash with a chance near ``n**2 / 2**65`` per
    locus of ``n`` families.
    """
    return hash((mi, strand)) & 0xFFFF_FFFF_FFFF_FFFF


@dataclasses.dataclass
class PhasingIndex:
    """The saved phasing result of one panel, read from its output directory.

    Attributes
    ----------
    directory : str
    snvs : dict of str to list of dict
        Per locus, the rows of ``snv_calls.tsv`` (phased and dropped).
    summary : dict of str to dict
        Per locus, the `SUMMARY_FIELDS` of ``locus_summary.tsv``.
    keys, labels : dict of str to ndarray
        Per locus, the sorted :func:`family_key` of each ``(mi, strand)`` of
        ``haplotype_calls.tsv`` (uint64), and its phase label as an index
        into `PHASE_LABELS` (uint8).
    """

    directory: str
    snvs: dict[str, list[dict]]
    summary: dict[str, dict]
    keys: dict[str, np.ndarray]
    labels: dict[str, np.ndarray]

    @classmethod
    def read(cls, directory: str | Path) -> PhasingIndex:
        """Read the three TSVs; ``haplotype_calls.tsv`` streams into compact arrays.

        Raises ValueError naming the file when a ``(locus, mi, strand)``
        occurs twice in ``haplotype_calls.tsv`` or a label is unknown.
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
        raw: dict[str, tuple[array, array]] = {}
        path = d / "haplotype_calls.tsv"
        with open(path, newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            at = {name: header.index(name) for name in ("locus_id", "mi", "strand",
                                                        "phase_label")}
            for row in reader:
                label = code.get(row[at["phase_label"]])
                if label is None:
                    raise ValueError(f"{path}: unknown phase label {row[at['phase_label']]!r}")
                k, lab = raw.setdefault(row[at["locus_id"]], (array("Q"), array("B")))
                k.append(family_key(row[at["mi"]], row[at["strand"]]))
                lab.append(label)
        keys, labels = {}, {}
        for locus, (k, lab) in raw.items():
            k_arr, lab_arr = np.frombuffer(k, dtype=np.uint64), np.frombuffer(lab, dtype=np.uint8)
            order = np.argsort(k_arr, kind="stable")
            keys[locus], labels[locus] = k_arr[order], lab_arr[order]
            if len(keys[locus]) > 1 and (np.diff(keys[locus]) == 0).any():
                raise ValueError(f"{path}: {locus} has one (mi, strand) on two rows")
        return cls(str(d), snvs, summary, keys, labels)

    def labels_of(self, locus_id: str, mi: np.ndarray, strand: np.ndarray) -> np.ndarray:
        """The phase label code of each panel row of a locus, by its ``(mi, strand)``."""
        keys = self.keys[locus_id]
        row_keys = np.fromiter((family_key(str(m), str(s)) for m, s in zip(mi, strand,
                                                                            strict=True)),
                               dtype=np.uint64, count=len(mi))
        at = np.searchsorted(keys, row_keys)
        return self.labels[locus_id][at]

    def check(self, panel: ExtractedPanel, panel_path: str | Path) -> None:
        """Raise ValueError unless every locus's labeled ``(mi, strand)`` set equals the panel's.

        The message names the panel, the phasing directory and the first
        loci that differ. Decodes every locus of the panel once.
        """
        # A locus with no reads has no rows in haplotype_calls.tsv.
        differ = sorted(set(self.keys) - set(panel.loci))
        no_rows = np.zeros(0, dtype=np.uint64)
        for locus_id in panel.loci:
            locus = panel.loci[locus_id]
            panel_keys = np.unique(np.fromiter(
                (family_key(str(m), str(s)) for m, s in zip(locus.mi, locus.strand,
                                                            strict=True)),
                dtype=np.uint64, count=len(locus.mi)))
            if not np.array_equal(panel_keys, self.keys.get(locus_id, no_rows)):
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
        ``index``, ``position``, ``counts``, ``coverage``, ``majority`` and,
        at an SNV, ``snv`` (its row of ``snv_calls.tsv``). ``rows``: per
        group of reads with one tract and one phase label (the label is
        None without phasing), ``tract``, ``label``, ``reads``, ``up`` and
        ``dn`` (per column, the most common base of the group's reads, empty
        when none of them has a base there, and the group's base counts).
        ``n_rows`` and ``reads_not_drawn`` count the groups past
        `PILEUP_ROWS`.
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
    row_label = (phasing.labels_of(locus_id, locus.mi, locus.strand)
                 if phasing is not None and len(locus.mi) else
                 np.full(len(count), -1, dtype=np.int64))

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
                   "counts": counts[i].tolist(), "coverage": cov,
                   "majority": FLANKING_BASES[int(np.argmax(counts[i]))] if cov else ""}
            if i in snv_at:
                col["snv"] = snv_at[i]
            cols.append(col)
        columns[name] = cols

    # Pile-up groups: one tract and one phase label. Each cell is the most
    # common base of the group's reads there.
    group_key = np.char.add(np.char.add(seq, "\t"), row_label.astype(str))
    keys, group_reads, group_of_row = _summed(group_key, count)
    group_order = np.lexsort((keys, -group_reads))
    drawn = group_order[:PILEUP_ROWS]
    position = {g: k for k, g in enumerate(drawn)}
    pick = np.array([position.get(g, -1) for g in group_of_row], dtype=np.int64)
    rows = []
    for g in drawn:
        tract, label = str(keys[g]).split("\t")
        rows.append({"tract": tract, "reads": int(group_reads[g]),
                     "label": PHASE_LABELS[int(label)] if int(label) >= 0 else None})
    for name, (block, _, _) in sides.items():
        idx = [c["index"] for c in columns[name]]
        cells = np.zeros((len(drawn), len(idx), len(FLANKING_BASES)), dtype=np.int64)
        if idx and len(fid):
            keep = pick >= 0
            bases = block[np.ix_(fid[keep], idx)]
            for b, base in enumerate(FLANKING_BASES):
                np.add.at(cells[:, :, b], pick[keep],
                          (bases == ord(base)) * count[keep][:, None])
        for k, row in enumerate(rows):
            row[name] = [[FLANKING_BASES[int(np.argmax(c))] if c.sum() else "", c.tolist()]
                         for c in cells[k]]

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
                   "reads_not_drawn": int(group_reads[group_order[PILEUP_ROWS:]].sum())},
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
    position) and ``n_snvs`` (the SNVs of the phasing result, None without
    one), or None when the panel does not hold the locus.
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
        out["n_snvs"] = None if phasing is None else len(phasing.snvs.get(name, []))
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
