"""`ExtractedPanel` — in-memory BAM-extraction artifact + serialization.

The data path is:

    BAM ──extract_panel──► ExtractedPanel
                              │
                              ├─► save_db(path)          ─► single SQLite file, one row per locus
                              └─► ExtractedPanel.load_db(path)   ─► lazy per-locus ``.loci``

`ExtractedPanel` holds faithful per-(MI, strand, seq) read counts in
deduped form (numpy arrays, not pandas — pickle is robust across pandas
versions).

`save_db`/`load_db` are the single serialization format and the streaming
runtime cache: one SQLite file, one row per locus, scalar metadata as
queryable columns and the per-row arrays gzip-pickled into a per-locus
BLOB. `load_db`'s `.loci` is a lazy, non-caching mapping — each locus is
decoded from disk on access, so a consumer iterating the panel never
holds more than one locus's arrays resident. This avoids the memory
blowup a whole-object gzip pickle has: a 0.408 GB gzipped panel would
unpickle to 13.27 GB resident, ~87% `flanking_seq` — a per-locus panel
never pays that, however deep it is.

The inference adapter that turns a panel into fittable loci lives in
prolyG (``prolyG.inference.to_loci``). This module keeps the read-level
primitives it and the phasing tables share: the run-length and
interrupter-pattern parse, and the per-read-family majority pattern.

Module is pysam-free so downstream code can load extraction artifacts
on machines without pysam installed.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import pickle
import re
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from prolyg_phasing.io.format import format_pattern

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ExtractionProvenance:
    """Panel-wide extraction kwargs + paths + version. One per panel."""

    bam_path: str
    bed_path: str
    extraction_version: str
    extraction_date: str
    min_mapq: int
    drop_chimeric: bool
    max_tlen: int
    anchor_hamming_max: int
    pair_disagreement_policy: str
    n_alleles_margin: int
    # Schema-v2 additions (flanking extraction). Defaults preserve
    # backward-compat for old provenance JSONs that lack these keys.
    min_base_q: int = 0
    extraction_schema_version: str = "1"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ExtractionProvenance:
        # Unknown future keys are dropped; missing legacy keys use defaults.
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})


@dataclasses.dataclass
class ExtractedLocus:
    """Per-locus extraction artifact: deduped (MI, strand, seq, count) rows + meta + QC.

    Rows are deduped on `(mi, strand, seq)` at extraction time; `count`
    is the aggregated read count for that triple. Reconstruct a pandas
    view via :meth:`to_dataframe` for inspection.

    Notes
    -----
    ``n_runs`` is derived from the **data** (max parsed tuple length
    across all accepted reads at this locus). It is recorded explicitly
    so the inference adapter doesn't need to re-walk the rows. The
    reference-derived ``reference_run_lengths`` is kept as diagnostic
    context but is no longer authoritative — interrupter-pattern
    polymorphisms can put a locus at multiple ``S`` values within one
    sample, and the inference adapter handles this via a per-pattern
    founder state space. Old pickles that carry the legacy
    reference-derived value on ``n_runs`` continue to load; ``to_loci``
    recomputes the per-pattern run counts from the rows so the loaded
    ``n_runs`` is informational on re-load.
    """

    # Per-row data (deduped).
    mi: np.ndarray              # (n_rows,) — read-family ID strings ("10004", etc.)
    strand: np.ndarray          # (n_rows,) 'U1' — 'A' / 'B'
    seq: np.ndarray             # (n_rows,) object — inter-anchor strings
    count: np.ndarray           # (n_rows,) int64 — aggregated read count

    # Structural meta (per-locus, constant across rows).
    n_runs: int
    ref_inter_anchor_seq: str
    reference_run_lengths: tuple[int, ...]

    # Locus geometry.
    chrom: str
    bed_start: int
    bed_end: int
    bed_name: str
    ref_orientation: str        # '+' (G-dominant) / '-' (C-dominant; revcomp'd)
    upstream_anchor: str
    downstream_anchor: str

    # QC counters.
    n_alignments_overlap_locus: int
    n_alignments_drop_mapq: int
    n_alignments_drop_sa: int
    n_alignments_drop_tlen: int
    n_alignments_drop_anchor: int
    n_read_pairs: int
    n_read_pairs_drop_disagree: int
    n_reads: int                 # = count.sum()
    max_observed_run_length: int
    anchorability_status: str

    # Schema-v2 additions (flanking extraction). All have defaults so
    # old pickles deserialize into a panel with empty flanking; a fresh
    # extraction repopulates them.
    flanking_id: np.ndarray = dataclasses.field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    flanking_seq: np.ndarray = dataclasses.field(
        default_factory=lambda: np.empty(0, dtype=object)
    )
    g_walk_up: int = 0
    g_walk_dn: int = 0
    flanking_up_width: int = 0
    flanking_dn_width: int = 0
    flanking_up_ref_pos_start: int = 0
    flanking_dn_ref_pos_start: int = 0

    def to_dataframe(self):
        """Materialize a pandas view of the (mi, strand, seq, count) rows."""
        import pandas as pd
        return pd.DataFrame({
            "mi": self.mi,
            "strand": self.strand,
            "seq": self.seq,
            "count": self.count,
        })

    def pattern_breakdown(self):
        """Per-row DataFrame with interrupter-pattern + parsed-tuple columns.

        Columns:

        - ``mi, strand, seq, count`` — the original per-row data.
        - ``pattern`` (str) — the read family's majority interrupter
          pattern as a ``"_"``-joined string (``""`` for no interrupters).
          Constant within an MI: same value on every row sharing that MI.
        - ``tuple`` (tuple[int, ...] | None) — parsed run lengths from
          ``seq``; ``None`` when ``len(parse) != self.n_runs`` (linker
          indel; not representable in the canonical state space).
        - ``tuple_str`` (str) — ``"(l_1, ..., l_S)"`` or ``"—"`` for
          parse-mismatched rows.

        For per-row pattern (vs. the read-family majority), call
        :func:`interrupter_pattern` on the ``seq`` column directly.
        """
        import pandas as pd

        mi_to_majority = majority_pattern_per_rf(self)

        patterns: list[str] = []
        tuples: list[tuple[int, ...] | None] = []
        tuple_strs: list[str] = []
        for i in range(len(self.mi)):
            mi = str(self.mi[i])
            patterns.append(format_pattern(mi_to_majority.get(mi, ())))
            parsed = parse_run_lengths(str(self.seq[i]))
            if len(parsed) == self.n_runs:
                tuples.append(parsed)
                tuple_strs.append("(" + ", ".join(str(x) for x in parsed) + ")")
            else:
                tuples.append(None)
                tuple_strs.append("—")

        return pd.DataFrame({
            "mi": self.mi,
            "strand": self.strand,
            "seq": self.seq,
            "count": self.count,
            "pattern": patterns,
            "tuple": tuples,
            "tuple_str": tuple_strs,
        })


@dataclasses.dataclass
class ExtractedPanel:
    """Full panel artifact: per-locus rows + panel-wide n_alleles + provenance."""

    loci: Mapping[str, ExtractedLocus]
    n_alleles: int
    provenance: ExtractionProvenance

    # ----- Serialization: SQLite (streaming) ---------------------------------

    def save_db(self, path: str | Path) -> None:
        """Write the panel as a single SQLite file, one row per locus.

        Per-locus scalar metadata is stored as queryable columns (see
        ``_PANEL_META_COLUMNS``); the per-row arrays
        (``mi``/``strand``/``seq``/``count``/``flanking_id``/``flanking_seq``)
        are gzip-pickled (``compresslevel=1``) into one ``row_blob`` column
        per locus. :meth:`load_db` decodes one
        locus's ``row_blob`` per access — the format exists so that streaming
        is possible, not because this write path itself saves memory (the
        panel is already fully built in memory by extraction time).

        Writes to a temporary path in the same directory and renames into
        place, so a crash mid-write cannot leave a partial file at ``path``.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.unlink(missing_ok=True)
        conn = sqlite3.connect(str(tmp))
        try:
            col_defs = ", ".join(
                f'"{c}" {"INTEGER" if c in _DB_INT_COLUMNS else "TEXT"}'
                for c in _DB_LOCI_COLUMNS[1:]
            )
            conn.execute(
                f'CREATE TABLE loci (locus_id TEXT PRIMARY KEY, {col_defs}, '
                f'row_blob BLOB)'
            )
            conn.execute("CREATE TABLE provenance (json TEXT)")
            conn.execute("CREATE TABLE panel_info (n_alleles INTEGER)")
            conn.execute(
                "INSERT INTO provenance (json) VALUES (?)",
                (json.dumps(self.provenance.to_dict()),),
            )
            conn.execute(
                "INSERT INTO panel_info (n_alleles) VALUES (?)", (self.n_alleles,)
            )
            placeholders = ", ".join("?" for _ in _DB_LOCI_COLUMNS)
            insert_sql = (
                f"INSERT INTO loci ({', '.join(_DB_LOCI_COLUMNS)}, row_blob) "
                f"VALUES ({placeholders}, ?)"
            )
            for locus_id, locus in self.loci.items():
                meta = _locus_meta_row(locus)
                values = [locus_id] + [meta[c] for c in _DB_LOCI_COLUMNS[1:]]
                values.append(_encode_row_blob(locus))
                conn.execute(insert_sql, values)
            conn.commit()
        finally:
            conn.close()
        tmp.replace(p)

    @classmethod
    def load_db(cls, path: str | Path) -> ExtractedPanel:
        """Load a panel written by :meth:`save_db`.

        ``.loci`` is a lazy, non-caching mapping (:class:`_LazyLociMap`)
        backed by a read-only connection to ``path``: each access re-reads
        and re-decodes exactly one locus, so iterating or subsetting the
        panel never holds more than one locus's arrays resident. Force an
        eager ``dict`` (e.g. ``dict(panel.loci)``) only when the whole panel
        is genuinely needed at once.
        """
        p = Path(path)
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        provenance_json = conn.execute("SELECT json FROM provenance").fetchone()[0]
        provenance = ExtractionProvenance.from_dict(json.loads(provenance_json))
        n_alleles = conn.execute("SELECT n_alleles FROM panel_info").fetchone()[0]
        return cls(loci=_LazyLociMap(conn), n_alleles=n_alleles, provenance=provenance)

    def locus_column(self, name: str) -> dict[str, object]:
        """One scalar metadata field of every locus, without decoding row arrays.

        ``name`` is one of the per-locus metadata columns ``save_db`` writes
        (``_PANEL_META_COLUMNS``). A panel loaded by :meth:`load_db` answers
        with one SQL query; an in-memory panel answers with the same
        serialized value ``save_db`` would store, so both return the same
        mapping for the same panel. Reading a scalar through ``.loci`` would
        decode each locus's whole ``row_blob`` for it.
        """
        if name not in _DB_LOCI_COLUMNS[1:]:
            raise ValueError(f"{name!r} is not a per-locus metadata column")
        if isinstance(self.loci, _LazyLociMap):
            return self.loci.column(name)
        return {lid: _locus_meta_row(locus)[name] for lid, locus in self.loci.items()}

# ---------------------------------------------------------------------------
# Tuple parse (uniform)
# ---------------------------------------------------------------------------


# One maximal non-G stretch: the run separator of every read parse. Compiled once;
# the parse runs on every read row, which on a deep panel is ~1e8 calls.
_NON_G_STRETCH = re.compile(r"[^G]+")


def parse_run_lengths(seq: str) -> tuple[int, ...]:
    """Split `seq` on maximal non-G stretches; return the tuple of G-run lengths.

    Interrupter base identity is not part of the parse: any non-G
    character (including N) is a run separator. SNVs in the linker
    region yield the same tuple as the canonical base.

    Edge cases:

    - All-G `seq` (single-run loci, no interrupters): returns
      `(len(seq),)`.
    - Empty `seq` (anchors adjacent in the read): returns `(0,)` —
      callers compare against the locus's `n_runs` to decide whether
      to accept.
    - `seq` starting or ending with a non-G base (zero-length boundary
      run): the boundary zero-run is preserved in the tuple, e.g.
      ``"AGGG"`` → ``(0, 3)``.
    """
    if seq == "":
        return (0,)
    # Splitting on non-G stretches yields one entry per G-run with
    # empty strings at boundaries when seq starts/ends with non-G.
    runs = _NON_G_STRETCH.split(seq)
    return tuple(len(r) for r in runs)


def reference_run_lengths_from_seq(ref_inter_anchor_seq: str) -> tuple[int, ...]:
    """Reference's run-length tuple, by the same parse used on reads."""
    return parse_run_lengths(ref_inter_anchor_seq)


def interrupter_pattern(seq: str) -> tuple[str, ...]:
    """Non-G stretches in `seq`, in order.

    The companion to :func:`parse_run_lengths`: where that function
    returns the G-run lengths, this returns the interrupter strings
    between them. Empty tuple for all-G `seq` (single-run loci) and
    for empty `seq`.

    Examples
    --------
    ``"GGGGGG"``      → ``()`` (no interrupters)
    ``"GGGAGGG"``     → ``("A",)``
    ``"GGGATGGG"``    → ``("AT",)`` (one multi-base interrupter)
    ``"GGGAGGGCGG"``  → ``("A", "C")``
    ``"AGGG"``        → ``("A",)`` (zero-length first G-run; interrupter retained)
    """
    if not seq:
        return ()
    return tuple(_NON_G_STRETCH.findall(seq))


# ---------------------------------------------------------------------------
# Per-read-family majority interrupter pattern and the pattern shares
#
# The phasing tables and plots read these, and so does prolyG's max-of-runs
# length histogram (the p^m anchor and the overdispersion width): a read
# enters that histogram only when its family's majority pattern clears the
# share floor. Founder admission is a separate, fit-time screen in prolyG.
# ---------------------------------------------------------------------------


def observed_pattern_frequencies(
    locus: ExtractedLocus,
) -> dict[tuple[str, ...], float]:
    """Per-read-family share at the locus, by majority interrupter pattern.

    Each read family $u$ (`n_rfs` total at the locus, in the glossary
    sense; one duplex molecule with both strands aggregated) is assigned
    the majority interrupter pattern across all its reads on both strands
    $\\tau \\in \\{A, B\\}$. Ties broken by lexicographic order of the
    pattern tuple.

    Returns
    -------
    dict[tuple[str, ...], float]
        Mapping interrupter pattern → (read families with that pattern) /
        `n_rfs`. Sums to 1.0 (modulo float). Empty dict when the locus
        has no reads.
    """
    return pattern_family_shares(majority_pattern_per_rf(locus))


def pattern_family_shares(
    majority: Mapping[str, tuple[str, ...]],
) -> dict[tuple[str, ...], float]:
    """Each pattern's share of read families, from a per-family majority map.

    ``majority`` maps each read family to its majority pattern, as
    :func:`majority_pattern_by_family` returns it. The share of pattern
    $h$ is (families whose majority is $h$) / (families). Empty dict for
    an empty map.
    """
    n_rfs = len(majority)
    if n_rfs == 0:
        return {}
    pattern_n_rfs: dict[tuple[str, ...], int] = defaultdict(int)
    for pattern in majority.values():
        pattern_n_rfs[pattern] += 1
    return {h: c / n_rfs for h, c in pattern_n_rfs.items()}


def select_patterns_above_freq(
    locus: ExtractedLocus,
    *,
    min_freq: float,
) -> set[tuple[str, ...]]:
    """Interrupter patterns with per-read-family share ≥ `min_freq`.

    Descriptive helper for plotting/diagnostics (the inference adapter no
    longer filters by pattern frequency). `min_freq = 0.0` keeps all
    observed patterns.
    """
    freqs = observed_pattern_frequencies(locus)
    return {h for h, f in freqs.items() if f >= min_freq}


def majority_pattern_per_rf(
    locus: ExtractedLocus,
) -> dict[str, tuple[str, ...]]:
    """Per-read-family majority interrupter pattern.

    Internal companion to :func:`observed_pattern_frequencies` — returns the
    per-MI assignment used to compute the frequencies. Useful for
    coloring plots and for grouping rows by their family's majority
    pattern rather than the row's own pattern.

    Returns
    -------
    dict[str, tuple[str, ...]]
        Mapping MI string → its majority interrupter pattern. Empty
        dict when the locus has no reads.
    """
    # A read string recurs across rows (tens of times per unique string on a
    # deep locus), so each unique string is parsed once.
    pattern_of: dict[str, tuple[str, ...]] = {}
    patterns: list[tuple[str, ...]] = []
    for seq in locus.seq:
        seq = str(seq)
        pattern = pattern_of.get(seq)
        if pattern is None:
            pattern = pattern_of[seq] = interrupter_pattern(seq)
        patterns.append(pattern)
    return majority_pattern_by_family(
        (str(m) for m in locus.mi), patterns, (int(c) for c in locus.count),
    )


def majority_pattern_by_family(
    mi: Iterable[str],
    pattern: Iterable[tuple[str, ...]],
    count: Iterable[int],
) -> dict[str, tuple[str, ...]]:
    """Per-read-family majority pattern, from rows already parsed.

    ``mi``, ``pattern`` and ``count`` are row-aligned: each row's read
    family, its own interrupter pattern and its read count. A family's
    majority is the pattern carrying the most reads over both strands;
    ties go to the lexicographically smallest pattern. The single
    definition of the majority vote: :func:`majority_pattern_per_rf`
    parses a locus and calls this, and a caller that already holds the
    parse calls it directly.
    """
    mi_pattern_count: dict[str, dict[tuple[str, ...], int]] = defaultdict(
        lambda: defaultdict(int),
    )
    for m, p, c in zip(mi, pattern, count, strict=True):
        mi_pattern_count[m][p] += c

    out: dict[str, tuple[str, ...]] = {}
    for m, pattern_count in mi_pattern_count.items():
        max_count = max(pattern_count.values())
        candidates = [h for h, c in pattern_count.items() if c == max_count]
        out[m] = min(candidates)
    return out


# ---------------------------------------------------------------------------
# Per-locus scalar-metadata column list (shared by the SQLite writer below)
# ---------------------------------------------------------------------------


_PANEL_META_COLUMNS = [
    "locus_id",
    "n_runs",
    "ref_inter_anchor_seq",
    "reference_run_lengths",
    "chrom", "bed_start", "bed_end", "bed_name",
    "ref_orientation",
    "upstream_anchor", "downstream_anchor",
    "anchorability_status",
    "max_observed_run_length",
    "n_alignments_overlap_locus",
    "n_alignments_drop_mapq",
    "n_alignments_drop_sa",
    "n_alignments_drop_tlen",
    "n_alignments_drop_anchor",
    "n_read_pairs",
    "n_read_pairs_drop_disagree",
    "n_reads",
    "n_alleles",
    # Schema-v2 flanking geometry.
    "g_walk_up", "g_walk_dn",
    "flanking_up_width", "flanking_dn_width",
    "flanking_up_ref_pos_start", "flanking_dn_ref_pos_start",
]


# ---------------------------------------------------------------------------
# SQLite panel format (streaming; save_db / load_db)
# ---------------------------------------------------------------------------

# The `loci` table's scalar-metadata columns: `_PANEL_META_COLUMNS` minus
# `n_alleles` (panel-wide, stored once in `panel_info` instead of repeated
# per row). `locus_id` stays first — it is the primary key.
_DB_LOCI_COLUMNS = [c for c in _PANEL_META_COLUMNS if c != "n_alleles"]

# Integer-typed columns among `_DB_LOCI_COLUMNS[1:]` (everything else is
# TEXT); mirrors the `int(...)` casts in `_read_panel_meta`.
_DB_INT_COLUMNS = {
    "n_runs", "bed_start", "bed_end", "max_observed_run_length",
    "n_alignments_overlap_locus", "n_alignments_drop_mapq",
    "n_alignments_drop_sa", "n_alignments_drop_tlen",
    "n_alignments_drop_anchor", "n_read_pairs",
    "n_read_pairs_drop_disagree", "n_reads",
    "g_walk_up", "g_walk_dn", "flanking_up_width", "flanking_dn_width",
    "flanking_up_ref_pos_start", "flanking_dn_ref_pos_start",
}

# The ExtractedLocus fields carried in `row_blob` rather than as SQL
# columns — the per-row arrays, keyed on the same locus as the scalar
# metadata columns.
_ROW_ARRAY_FIELDS = ("mi", "strand", "seq", "count", "flanking_id", "flanking_seq")


def _encode_row_blob(locus: ExtractedLocus) -> bytes:
    """Gzip-pickle a locus's per-row arrays (``compresslevel=1``, see ``save_db``)."""
    payload = {f: getattr(locus, f) for f in _ROW_ARRAY_FIELDS}
    return gzip.compress(
        pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), compresslevel=1,
    )


def _decode_row_blob(blob: bytes) -> dict:
    return pickle.loads(gzip.decompress(blob))


def _row_to_locus(locus_id: str, row: tuple) -> ExtractedLocus:
    """Reconstruct one ``ExtractedLocus`` from a ``loci`` table row.

    ``row`` is ``(*scalar metadata in _DB_LOCI_COLUMNS[1:] order, row_blob)``,
    matching the ``SELECT`` in :class:`_LazyLociMap`.
    """
    meta = dict(zip(_DB_LOCI_COLUMNS[1:], row[:-1], strict=True))
    meta["reference_run_lengths"] = (
        tuple(int(x) for x in meta["reference_run_lengths"].split(","))
        if meta["reference_run_lengths"] else tuple()
    )
    arrays = _decode_row_blob(row[-1])
    return ExtractedLocus(
        mi=arrays["mi"], strand=arrays["strand"], seq=arrays["seq"],
        count=arrays["count"], flanking_id=arrays["flanking_id"],
        flanking_seq=arrays["flanking_seq"],
        **meta,
    )


class _LazyLociMap(Mapping):
    """Read-only, non-caching ``locus_id -> ExtractedLocus`` view over a panel DB.

    Backs :meth:`ExtractedPanel.load_db`'s ``.loci``. Every access re-reads
    and re-decodes its own locus from SQLite; nothing here caches the
    decoded ``ExtractedLocus``, so a consumer iterating ``panel.loci.items()``
    never holds more than one locus's arrays resident at a time.
    """

    _SELECT_COLS = ", ".join(_DB_LOCI_COLUMNS[1:]) + ", row_blob"

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def __getitem__(self, locus_id: str) -> ExtractedLocus:
        row = self._conn.execute(
            f"SELECT {self._SELECT_COLS} FROM loci WHERE locus_id = ?", (locus_id,),
        ).fetchone()
        if row is None:
            raise KeyError(locus_id)
        return _row_to_locus(locus_id, row)

    def __iter__(self):
        for (locus_id,) in self._conn.execute(
            "SELECT locus_id FROM loci ORDER BY locus_id",
        ):
            yield locus_id

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM loci").fetchone()[0]

    def __contains__(self, locus_id) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM loci WHERE locus_id = ?", (locus_id,),
        ).fetchone()
        return row is not None

    def column(self, name: str) -> dict[str, object]:
        """``{locus_id: value}`` of one metadata column; ``name`` is pre-validated."""
        return dict(self._conn.execute(f'SELECT locus_id, "{name}" FROM loci'))


def read_locus_chroms(path: str | Path) -> dict[str, str]:
    """Per-locus ``chrom`` from a ``panel.db``, without decoding any ``row_blob``.

    One lightweight SQL query for a caller that needs only the chromosome
    per locus (e.g. the germline-sex hemizygous-locus check) — decoding the
    full ``row_blob`` (dominated by ``flanking_seq``) for a scalar field
    every locus already carries as a column would defeat the point of the
    streaming format.
    """
    p = Path(path)
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    try:
        return dict(conn.execute("SELECT locus_id, chrom FROM loci"))
    finally:
        conn.close()


def loci_equal(a: ExtractedLocus, b: ExtractedLocus) -> list[str]:
    """Field names where two ``ExtractedLocus`` records differ (empty if equal).

    Compares every dataclass field generically (``np.array_equal`` for
    array fields, ``==`` otherwise), so new fields are checked automatically
    without updating this function. Backs the format-equivalence oracle
    that verifies a panel-format change never silently changes a number.
    """
    mismatches = []
    for f in dataclasses.fields(ExtractedLocus):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
            if not np.array_equal(va, vb):
                mismatches.append(f.name)
        elif va != vb:
            mismatches.append(f.name)
    return mismatches


def panels_equal(a: ExtractedPanel, b: ExtractedPanel) -> dict[str, list[str]]:
    """Per-locus field mismatches between two panels (empty dict if fully equal).

    Keys are locus_ids with at least one differing field, mapped to
    :func:`loci_equal`'s mismatch list, plus synthetic keys
    ``"__n_alleles__"`` / ``"__provenance__"`` for panel-wide fields. Raises
    ``ValueError`` if the two panels don't carry the same locus_id set (a
    set mismatch is not a per-locus diff).
    """
    ids_a, ids_b = set(a.loci.keys()), set(b.loci.keys())
    if ids_a != ids_b:
        raise ValueError(
            f"locus_id sets differ: only in a = {ids_a - ids_b}, "
            f"only in b = {ids_b - ids_a}"
        )
    out: dict[str, list[str]] = {}
    for locus_id in ids_a:
        mism = loci_equal(a.loci[locus_id], b.loci[locus_id])
        if mism:
            out[locus_id] = mism
    if a.n_alleles != b.n_alleles:
        out["__n_alleles__"] = [f"{a.n_alleles} != {b.n_alleles}"]
    if a.provenance.to_dict() != b.provenance.to_dict():
        out["__provenance__"] = ["differs"]
    return out


def _locus_meta_row(locus: ExtractedLocus) -> dict:
    """Per-locus scalar metadata as a dict, keyed like ``_PANEL_META_COLUMNS``.

    Excludes ``locus_id`` (the caller's key) and ``n_alleles`` (panel-wide,
    not per-locus). Single source for the field list shared by the TSV
    writer (:func:`_write_panel_meta`) and the SQLite writer
    (:meth:`ExtractedPanel.save_db`).
    """
    return {
        "n_runs": locus.n_runs,
        "ref_inter_anchor_seq": locus.ref_inter_anchor_seq,
        "reference_run_lengths": ",".join(
            str(r) for r in locus.reference_run_lengths
        ),
        "chrom": locus.chrom,
        "bed_start": locus.bed_start,
        "bed_end": locus.bed_end,
        "bed_name": locus.bed_name,
        "ref_orientation": locus.ref_orientation,
        "upstream_anchor": locus.upstream_anchor,
        "downstream_anchor": locus.downstream_anchor,
        "anchorability_status": locus.anchorability_status,
        "max_observed_run_length": locus.max_observed_run_length,
        "n_alignments_overlap_locus": locus.n_alignments_overlap_locus,
        "n_alignments_drop_mapq": locus.n_alignments_drop_mapq,
        "n_alignments_drop_sa": locus.n_alignments_drop_sa,
        "n_alignments_drop_tlen": locus.n_alignments_drop_tlen,
        "n_alignments_drop_anchor": locus.n_alignments_drop_anchor,
        "n_read_pairs": locus.n_read_pairs,
        "n_read_pairs_drop_disagree": locus.n_read_pairs_drop_disagree,
        "n_reads": locus.n_reads,
        "g_walk_up": locus.g_walk_up,
        "g_walk_dn": locus.g_walk_dn,
        "flanking_up_width": locus.flanking_up_width,
        "flanking_dn_width": locus.flanking_dn_width,
        "flanking_up_ref_pos_start": locus.flanking_up_ref_pos_start,
        "flanking_dn_ref_pos_start": locus.flanking_dn_ref_pos_start,
    }


