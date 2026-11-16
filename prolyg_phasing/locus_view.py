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
from multiprocessing import Pool
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from prolyg_phasing.io._flank_placement import (
    FLANK_LEN,
    G_SCORE,
    GAP_OPEN,
    MAX_GAP,
    NO_FLANK_PAIR,
    NO_REFERENCE_POSITION,
    NON_G_SCORE,
    SUBSTITUTION_COST,
    WINDOW_RECOVERED,
    WINDOW_UNRECOVERABLE,
)
from prolyg_phasing.io._workers import worker_count
from prolyg_phasing.io.panel import (
    ExtractedLocus,
    ExtractedPanel,
    ExtractionProvenance,
    parse_run_lengths,
)

# Pile-up rows drawn per locus and sample; the rest are counted.
PILEUP_ROWS = 150

# Tract rows kept per locus and sample; the rest are counted.
TRACT_ROWS = 200

# Default page setting: the flanking bases shown next to each flank. The page
# reads it from /loci, and /locus uses it when the request names none.
NEAR_BASES = 0

# Characters counted at each flanking position. "." (no read base) is not
# counted.
FLANKING_BASES = "ACGTN"

# Phase labels of haplotype_calls.tsv, and their codes in the index.
PHASE_LABELS = ("hap_major", "hap_minor", "unk")

# The locus_summary.tsv fields the page shows.
SUMMARY_FIELDS = ("hap_major_profile", "hap_minor_profile", "n_snvs_phased",
                  "n_snvs_dropped_low_linkage", "reads_major", "reads_minor", "reads_unk")

_COUNTER_PREFIXES = ("n_alignments_", "n_read_pairs")


# ---------------------------------------------------------------------------
# The page's help texts
# ---------------------------------------------------------------------------

# The one table of the page's hover texts, in three groups: "counters", one
# text per counter of `locus_view_data`, from where `io.bam` adds to it;
# "status", one text per status label of the locus list; "filters", one text
# per filter chip of the page. `view_help` fills the fields in braces from the
# panel's provenance and the flank-match constants. A counter, a status the
# extraction writes, or a chip with no text here fails a unit test.
_FLANK_MATCH = ("A match may differ from the {flank_len} flank bases by one substitution or by "
                "one gap of 1 or {max_gap} bases. ")
_COUNTER_HELP = {
    "n_alignments_overlap_locus":
        "Alignments that overlap the BED interval. We skip unmapped, secondary and "
        "supplementary alignments, and alignments with no sequence. "
        "n_alignments_overlap_locus = n_alignments_drop_mapq + n_alignments_drop_sa + "
        "n_alignments_drop_tlen + n_alignments_drop_no_mi + n_alignments_placed + "
        "n_alignments_drop_flank \u2212 n_alignments_drop_negative_value. "
        "Every counter is 0 at a locus with no flank.",
    "n_alignments_drop_mapq":
        "Alignments with mapping quality below {min_mapq}. We drop these alignments first. "
        "The limit is --min-mapq of the extraction.",
    "n_alignments_drop_sa":
        "{sa_text}",
    "n_alignments_drop_tlen":
        "Alignments with an insert size |TLEN| above {max_tlen}, among the alignments that "
        "pass the MAPQ and SA filters. We keep alignments with TLEN 0. The limit is "
        "--max-tlen of the extraction.",
    "n_alignments_drop_no_mi":
        "Alignments with no MI tag, among the alignments that pass the MAPQ, SA and TLEN "
        "filters. We need the MI tag to group reads into read families.",
    "n_alignments_placed":
        "Alignments with an upstream flank match, a downstream flank match after the "
        "upstream flank match, and a read base between the two matches that aligns inside "
        "the BED interval. We search only the alignments that pass the MAPQ, SA, TLEN and "
        "MI filters. We keep n_alignments_placed \u2212 n_alignments_drop_negative_value "
        "alignments.",
    "n_alignments_drop_flank":
        "Alignments that we drop in the flank search, among the alignments that pass the "
        "MAPQ, SA, TLEN and MI filters. Each dropped alignment counts once. The five "
        "counters below add up to n_alignments_drop_flank plus the alignments with no "
        "match of either flank.",
    "n_alignments_drop_upstream_flank_not_placed":
        "Alignments with no match of the upstream flank that ends at a non-G base "
        "followed by a G. " + _FLANK_MATCH +
        "An alignment with no match of either flank counts here and in "
        "n_alignments_drop_downstream_flank_not_placed.",
    "n_alignments_drop_downstream_flank_not_placed":
        "Alignments with no match of the downstream flank that starts at a non-G base "
        "after a G. " + _FLANK_MATCH +
        "An alignment with no match of either flank counts here and in "
        "n_alignments_drop_upstream_flank_not_placed.",
    "n_alignments_drop_no_placement":
        "Alignments with a match of each flank, but with no upstream flank match before a "
        "downstream flank match.",
    "n_alignments_drop_no_bed_base":
        "Alignments with an upstream flank match before a downstream flank match, but with "
        "no read base between the two matches that aligns inside the BED interval. A base "
        "in an insertion or a soft clip has no reference position.",
    "n_alignments_drop_negative_value":
        "Alignments with a negative value at every pair of flank matches. We add {g_score} "
        "per G between the two matches and {non_g_score} per other base. We subtract "
        "{sub_cost} per flank substitution and {gap1_cost} or {gap2_cost} per flank gap of "
        "1 or 2 bases. These alignments also count in n_alignments_placed.",
    "n_read_pairs":
        "Read names with a kept alignment, as a read pair or as a single read. We skip a "
        "read name whose MI tag does not end in /A or /B, a read name with more than two "
        "mates, and a read pair whose mates differ in MI. No counter holds the skipped "
        "read names. We use a mate outside the BED interval only if the mate covers a "
        "flanking base. n_read_pairs \u2264 n_alignments_placed \u2212 "
        "n_alignments_drop_negative_value.",
    "n_read_pairs_drop_disagree":
        "Read pairs with two mates over the BED interval and two different tracts that we "
        "cannot merge. The two stretches from flank match to flank match differ in length, "
        "a mate has no base qualities, the two mates differ at a base of equal quality "
        "inside either mate's tract, or the merged stretch has no tract. We drop these "
        "read pairs.",
    "n_read_pairs_merged":
        "Read pairs with two different tracts. In each mate we take the stretch from the "
        "outer base of one flank match to the outer base of the other. At each base where "
        "the two stretches differ, we keep the base of higher quality. At a flank base of "
        "equal quality, we write N. The tract is the "
        "best placement of the two flanks in the merged stretch. A read pair with two equal "
        "tracts counts neither here nor in n_read_pairs_drop_disagree.",
    "n_reads":
        "Reads at this locus in the panel. A read is a read name of n_read_pairs that we "
        "keep. n_reads = n_read_pairs \u2212 n_read_pairs_drop_disagree.",
}


def _half_units(score: int) -> str:
    """A half-unit placement score in whole units, with a true minus sign."""
    return f"{score / 2:g}".replace("-", "\u2212")


# The status labels: the panel's stored reference_window_status, and two the
# page derives. NO_REFERENCE_POSITION keys "side with no position", which the
# page shows for a stored "recovered" locus with a flank side that
# `_lacks_position`; "not_in_panel" is a locus that only the other sample's
# panel holds.
_STATUS_HELP = {
    WINDOW_RECOVERED:
        "Stored in the panel. We rebuilt the reference window of the locus from the MD tags "
        "of the reads. Both flanks have a reference position.",
    WINDOW_UNRECOVERABLE:
        "Stored in the panel. We could not rebuild the reference window of the locus from "
        "the reads of the matched normal, or from the reads of this sample with MAPQ "
        "{min_mapq} or more. The locus has no flank and no reads.",
    NO_FLANK_PAIR:
        "Stored in the panel. We rebuilt the reference window from the matched normal, but "
        "no flank pair has a positive value in build-flanks. The locus has no flank and no "
        "reads.",
    NO_REFERENCE_POSITION:
        "Not stored in the panel. The panel stores the status recovered. No read of the "
        "matched normal aligns at both end bases of one flank, and that flank has no "
        "reference position. We still use the flank to find the tract in each read. We show "
        "no flanking bases on that side, and phasing gets no SNVs from that side.",
    "not_in_panel":
        "Not stored in the panel. The panel of this sample has no row for the locus. The "
        "panel of the other sample has one.",
}

# The filter chips, by their key in the page.
_FILTER_HELP = {
    "all": "All loci of the panels.",
    "noflank": "Loci with the status ref window unrecoverable or no flank pair in a shown "
               "sample.",
    "noposition": "Loci with a flank that has no reference position in a shown sample.",
    "noreads": "Loci with no reads in a shown sample.",
}

HELP_TEXTS: dict[str, dict[str, str]] = {
    "counters": _COUNTER_HELP, "status": _STATUS_HELP, "filters": _FILTER_HELP}


def view_help(provenance: ExtractionProvenance) -> dict[str, dict[str, str]]:
    """`HELP_TEXTS` with the limits of the panel's provenance and the flank match filled in."""
    if provenance.drop_chimeric:
        sa_text = ("Alignments with an SA tag, among the alignments that pass the MAPQ "
                   "filter. We drop these chimeric alignments. --no-drop-chimeric keeps them.")
    else:
        sa_text = ("The extraction ran with --no-drop-chimeric. We kept the alignments "
                   "with an SA tag, and the count is 0.")
    fields = {
        "min_mapq": provenance.min_mapq, "max_tlen": provenance.max_tlen, "sa_text": sa_text,
        "flank_len": FLANK_LEN, "max_gap": MAX_GAP,
        "g_score": "+" + _half_units(G_SCORE), "non_g_score": _half_units(NON_G_SCORE),
        "sub_cost": _half_units(SUBSTITUTION_COST), "gap1_cost": _half_units(GAP_OPEN + 1),
        "gap2_cost": _half_units(GAP_OPEN + 2),
    }
    return {group: {name: text.format(**fields) for name, text in texts.items()}
            for group, texts in HELP_TEXTS.items()}


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


def molecule_ids(locus: ExtractedLocus) -> tuple[int, np.ndarray]:
    """The locus's molecules, and each row's molecule as an index among them.

    A molecule is a distinct MI. The panel's MI holds no strand (strand is
    its own field), so the A and B strands of a read family are one molecule.
    """
    distinct, index = np.unique(np.asarray(locus.mi, dtype=str), return_inverse=True)
    return len(distinct), index.ravel()


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
    n_molecules : int
        The locus's molecules, distinct MI (:func:`molecule_ids`).
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
        ``snv`` (its row of ``snv_calls.tsv``). ``sections``: one per phase
        label in `PHASE_LABELS` order (one, label None, without phasing),
        each with ``label``, ``reads``, ``molecules`` (distinct MI among
        the section's reads; a molecule with reads under two labels counts
        in each), ``rows``, ``n_rows`` and ``reads_not_drawn`` (the rows
        past `PILEUP_ROWS` in the section).
        A row is one distinct (tract, the read's own base or "." at every
        shown column) within the section, with ``tract``, ``label``,
        ``reads``, ``up`` and ``dn`` (the bases at the side's columns, one
        character each), sorted by reads. A major or minor row also has
        ``differs``: per side, the column numbers of the phased SNVs where
        the row's base is not its label's haplotype base in
        ``snv_calls.tsv`` (empty cells never count).
    tract_labels : list or None
        With phasing, per listed tract, its reads by phase label in
        `PHASE_LABELS` order.
    largest_run_labels : list or None
        With phasing, ``[largest G run, reads by phase label]``.
    largest_run_shares : dict or None
        With phasing, per phase label, ``[largest G run, share]``: the
        label's reads at that run over the label's reads (empty for a label
        with no reads).
    """
    counters = {f.name: int(getattr(locus, f.name))
                for f in dataclasses.fields(locus)
                if f.name.startswith(_COUNTER_PREFIXES)}
    counters["n_reads"] = int(locus.n_reads)
    count = np.asarray(locus.count, dtype=np.int64)
    seq = np.asarray(locus.seq, dtype=str)

    # Tracts, with their reads and molecules (distinct MI).
    tracts, tract_reads, tract_of_row = _summed(seq, count)
    n_molecules, mi_id = molecule_ids(locus)
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
    for part in (up_cells[:len(seq)], dn_cells[:len(seq)]):
        row_key = np.char.add(np.char.add(row_key, "\t"), part)
    # One section per phase label (one section, label None, without phasing).
    labels_present = list(range(len(PHASE_LABELS))) if phasing is not None else [-1]
    snv_cols = {name: [(k, c["snv"]) for k, c in enumerate(columns[name])
                       if "snv" in c and c["snv"]["status"] == "phased"]
                for name in ("up", "dn")}
    sections = []
    for code in labels_present:
        in_label = row_label == code
        keys, key_reads, key_of_row = _summed(row_key[in_label], count[in_label])
        # Molecules per pile-up row: distinct MI among the row's reads.
        key_mi = np.unique(np.stack([key_of_row, mi_id[in_label]]), axis=1)
        key_molecules = np.bincount(key_mi[0], minlength=len(keys))
        key_order = np.lexsort((keys, -key_reads))
        label = PHASE_LABELS[code] if code >= 0 else None
        rows = []
        for k in key_order[:PILEUP_ROWS]:
            tract, up, dn = str(keys[k]).split("\t")
            row = {"tract": tract, "reads": int(key_reads[k]),
                   "molecules": int(key_molecules[k]), "up": up, "dn": dn, "label": label}
            if label in ("hap_major", "hap_minor"):
                field = f"{label}_base"
                row["differs"] = {
                    name: [k2 for k2, snv in snv_cols[name]
                           if bases[k2] != "." and bases[k2] != snv[field]]
                    for name, bases in (("up", up), ("dn", dn))}
            rows.append(row)
        # A molecule with reads under two labels counts in each label.
        sections.append({"label": label, "rows": rows, "n_rows": len(keys),
                         "reads": int(key_reads.sum()),
                         "molecules": len(np.unique(mi_id[in_label])),
                         "reads_not_drawn": int(key_reads[key_order[PILEUP_ROWS:]].sum())})

    # Reads per tract and per largest G run, by phase label.
    tract_labels = largest_labels = largest_shares = None
    if phasing is not None:
        by_label = np.zeros((len(tracts), len(PHASE_LABELS)), dtype=np.int64)
        np.add.at(by_label, (tract_of_row, row_label), count)
        tract_labels = [by_label[i].tolist() for i in order[:TRACT_ROWS]]
        runs: dict[int, np.ndarray] = {}
        for i, tract in enumerate(tracts):
            run = max(parse_run_lengths(str(tract)))
            runs[run] = runs.get(run, 0) + by_label[i]
        largest_labels = [[k, v.tolist()] for k, v in sorted(runs.items())]
        totals = by_label.sum(axis=0)
        largest_shares = {
            label: ([[k, float(v[i]) / float(totals[i])] for k, v in sorted(runs.items())]
                    if totals[i] else [])
            for i, label in enumerate(PHASE_LABELS)}

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
        "tract_labels": tract_labels,
        "n_tracts": len(tracts),
        "n_molecules": n_molecules,
        "tract_reads_not_shown": int(tract_reads[order[TRACT_ROWS:]].sum()),
        "largest_run": [[k, v] for k, v in sorted(largest.items())],
        "largest_run_labels": largest_labels,
        "largest_run_shares": largest_shares,
        "near": near,
        "phasing": None if phasing is None else {
            "summary": phasing.summary.get(locus_id), "snvs": snvs},
        "flanking": flanking,
        "pileup": {"up": columns["up"], "dn": columns["dn"], "sections": sections},
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


_MOLECULE_PANEL: list[ExtractedPanel] = []


def _open_molecule_worker(path: str) -> None:
    _MOLECULE_PANEL[:] = [ExtractedPanel.load_db(path)]


def _locus_molecules(locus_id: str) -> int:
    return molecule_ids(_MOLECULE_PANEL[0].loci[locus_id])[0]


def molecule_counts(path: str | Path, n_workers: int) -> dict[str, int]:
    """The molecules of every locus of the panel at `path` (see :func:`molecule_ids`).

    It decodes the rows of every locus, which costs seconds per panel, so
    `n_workers` processes share the loci.
    """
    loci = list(ExtractedPanel.load_db(path).loci)
    n_workers = max(1, min(n_workers, len(loci)))
    with Pool(n_workers, initializer=_open_molecule_worker, initargs=(str(path),)) as pool:
        return dict(zip(loci, pool.map(_locus_molecules, loci, chunksize=1), strict=True))


def locus_list(panels: Sequence[ExtractedPanel], phasings: Sequence[PhasingIndex | None],
               molecules: Sequence[dict[str, int]]) -> dict:
    """The locus list: every locus of the panels, with per-sample fields.

    Read from the panels' metadata columns; `molecules` gives each panel's
    :func:`molecule_counts`. Per sample: ``n_reads``, ``molecules`` (distinct MI, see
    :func:`molecule_ids`), ``status``, ``largest_run`` (the largest observed
    G run), ``no_position`` (a flank side with no reference
    position) and ``snvs`` (``{"phased": n, "dropped": m}`` from the
    phasing result's ``snv_calls.tsv``, None when no phasing result was
    given), or None when the panel does not hold the locus. Each entry of
    ``samples`` also holds ``help``, the panel's :func:`view_help`.
    """
    columns = []
    for panel, counts in zip(panels, molecules, strict=True):
        up, dn = panel.locus_column("upstream_flank"), panel.locus_column("downstream_flank")
        columns.append({
            "n_reads": panel.locus_column("n_reads"),
            "status": panel.locus_column("reference_window_status"),
            "largest_run": panel.locus_column("max_observed_run_length"),
            "no_position": {k: _lacks_position(up[k]) or _lacks_position(dn[k]) for k in up},
            "molecules": counts,
        })
    names: dict[str, None] = {}
    for panel in panels:
        for name in panel.loci:
            names.setdefault(name, None)
    fields = ("n_reads", "molecules", "status", "largest_run", "no_position")

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
                         "help": view_help(p.provenance),
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
    raises ValueError. The locus list is read once, here, with the molecules
    of every locus (:func:`molecule_counts`). Call
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
    n_workers = worker_count()
    molecules = [molecule_counts(path, n_workers) for path in paths]
    listing = json.dumps(locus_list(opened.get(), phasings, molecules),
                         separators=(",", ":")).encode()
    handler = type("LocusViewHandler", (_Handler,),
                   {"panels": opened, "phasings": phasings, "page": page, "listing": listing})
    return ThreadingHTTPServer(("127.0.0.1", port), handler)
