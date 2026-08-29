"""`ExtractedPanel` — in-memory BAM-extraction artifact + serialization + inference adapter.

The data path is:

    BAM ──extract_panel──► ExtractedPanel ──to_loci──► ToLociResult
                                                       (locus_ids, loci,
                                                        n_alleles, drops)
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

`to_loci` is the inference adapter: it walks each row's `seq`, parses
it into a length tuple against the locus's reference run structure
(positions of interrupter bases), and emits `LocusData` objects
(single-run is the `n_runs == 1` special case) ready for
`prolyG.inference.train.fit_normal`.

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
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm

from prolyg_phasing.io.format import format_pattern

if TYPE_CHECKING:
    from prolyg_phasing.phasing import FlankingHaplotypeAssignment

# Founder length-state budget. The dense length kernel
# scores a founder over its **active** (frozen-run-reduced) length-state space
# K' = k_h ** n_active; a locus is dropped (drops['overflow']) when an admitted
# founder's K' exceeds this. The freeze (n_active <= |h|) is why the budget keys
# on n_active, not the nominal |h| — the nominal space is never materialized.
# (Defined module-top because it is a ``to_loci`` default, bound at class def.)
_DEFAULT_MAX_FOUNDER_LENGTH_STATES = 100_000
# Separate, fixed encoding ceiling: the full-|h| flat-tuple ell = Σ ℓ_s·k_h^s is
# stored as int64, so a pattern whose nominal k_h ** |h| exceeds this can't be
# encoded — its ell is sentinelled to 0 in _build_locus (lossless off-founder),
# and an admitted founder over it drops the locus (un-scorable). Not tunable —
# it's the int64 dtype limit, distinct from the feasibility budget above.
_MAX_INT64_FLAT = 2 ** 63

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
class ToLociResult:
    """Output of :meth:`ExtractedPanel.to_loci` — fittable loci + drop report.

    Attributes
    ----------
    locus_ids : list[str]
        Sorted IDs of loci that survived adaptation + filtering.
    loci : list[LocusData]
        Fittable per-locus inference objects, aligned with ``locus_ids``.
        Never contains ``None``.
    n_alleles : int
        Panel-wide ``n_alleles_max`` — the largest per-pattern axis $k_h$ over
        surviving loci. Sizes $\\lambda$ and the global model
        state; the per-pattern axes live on each ``LocusData.n_alleles_per_pattern``
        and the kernels slice $\\lambda$ to the relevant $k_h$.
    drops : dict[str, list[str]]
        Drop report keyed by reason → list of dropped locus_ids.
        Reasons: ``"unfittable"`` (no fittable
        :class:`LocusData` from the I/O
        adapter — ``max_n_runs`` cap exceeded or empty post-filter CSR;
        the two are indistinguishable by design) and
        ``"not_anchorable"`` (``ExtractedLocus.anchorability_status !=
        "anchorable"``; absent when ``require_anchorable=False``).
    ell_ref : list[float]
        Per-locus fit-free :math:`p^m` anchor length :math:`\\ell_{\\mathrm{ref},i}`
        (the max-of-runs read-histogram mode), aligned with ``loci``. The single
        source of truth for the anchor — derived here from the
        :class:`ExtractedLocus` reads via
        :func:`prolyG.inference.overdispersion.panel_ref_and_delta` (byte-identical
        to the overdispersion width statistic) and read directly by the fits.
    delta_up : list[float]
        Per-locus founder-region half-extent **above** the anchor,
        :math:`\\Delta_{\\mathrm{up},i} = \\ell_{\\mathrm{hi},i} -
        \\ell_{\\mathrm{ref},i}`, from the dominant basin of the max-of-runs
        histogram, aligned with ``loci``. Sets the upper edge of the
        two-sided :math:`p^m` box.
    delta_down : list[float]
        Per-locus founder-region half-extent **below** the anchor,
        :math:`\\Delta_{\\mathrm{down},i} = \\ell_{\\mathrm{ref},i} -
        \\ell_{\\mathrm{lo},i}`, from the same dominant basin, aligned with
        ``loci``. Sets the lower edge of the two-sided :math:`p^m` box.
    """

    locus_ids: list[str]
    loci: list
    n_alleles: int
    drops: dict[str, list[str]]
    ell_ref: list[float]
    delta_up: list[float]
    delta_down: list[float]
    # The founder-histogram pattern-share floor this build used for the anchor
    # (``panel_ref_and_delta``). The downstream regime partition
    # (``partition_by_overdispersion``) must use the SAME value so the p^m-box
    # anchor and the regime membership are built from the same histogram; the
    # fit threads this through rather than re-defaulting to 0.1.
    min_pattern_freq: float = 0.1


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

    # ----- Inference adapter ------------------------------------------------

    def to_loci(
        self,
        *,
        max_n_runs: int | None = None,
        min_pattern_freq: float = 0.1,
        min_pattern_families: int = 5,
        min_families_run: int = 5,
        max_founder_length_states: int = _DEFAULT_MAX_FOUNDER_LENGTH_STATES,
        require_anchorable: bool = True,
        phasing: dict[str, FlankingHaplotypeAssignment] | None = None,
        exclude_locus_ids: Collection[str] | None = None,
        verbose: bool = False,
    ) -> ToLociResult:
        """Adapt to inference inputs at per-pattern length axes.

        Two-pass: first, parse every read at each locus into its own
        interrupter pattern $h(r)$ and run-length vector (no equality
        filter, no majority assignment — only the ``max_n_runs`` cap), and
        record the per-row parses, tracking each pattern's max realized run.
        Then set the **per-pattern** length axis $k_h$ =
        ``n_alleles_per_pattern[h]``: multi-run patterns get
        ``max realized run + margin`` rounded up to a bucket; single-run
        patterns share a panel-wide axis = ``max run among S=1 loci + margin``.
        Finally, encode pattern-keyed CSRs at each pattern's radix, drop
        unfittable + over-budget + non-anchorable loci, and return a
        :class:`ToLociResult` with surviving loci plus a structured drop report.

        The returned ``n_alleles`` is ``n_alleles_max`` — the **largest**
        per-pattern axis on the panel — which sizes $\\lambda$ and the global
        state; each locus carries its own per-pattern axes on
        ``LocusData.n_alleles_per_pattern`` and the kernels slice $\\lambda$ to
        the relevant $k_h$. The panel's extraction-time ``self.n_alleles`` (set
        by ``bam.extract_panel``) is a loose panel-wide pre-filter and is not
        carried forward past this call; the per-pattern axes are materially
        tighter, shrinking the stutter matrix, ``pis``, and the multi-run
        candidate state space ($k_h^{|h|}$) directly.

        Interrupter base identity (SNVs / Ns in the linker) does not
        gate the parse: maximal G-runs define the tuple.

        Parameters
        ----------
        max_n_runs : int, optional
            Skip loci whose locus-level max run count
            $\\max_{h \\in \\mathcal{H}_i} |h|$ exceeds this cap. No longer
            needed to avoid overflow (the encoding sentinels unrepresentable
            off-pattern ``ell`` and ``min_pattern_freq`` gates the founder
            drop); retained as an optional locus filter.
        min_pattern_freq : float, default 0.1
            Founder-admission frequency threshold (matches ``fit_normal``'s
            default). Predicts which patterns the fit will admit as founders
            (``_founder_eligible_patterns``, mirroring
            ``admitted_founder_patterns``), so their length axes are kept
            lossless and they gate the over-budget drop.
        min_pattern_families : int, default 5
            Founder-admission **absolute read-count** threshold, the companion
            to ``min_pattern_freq`` (matches ``fit_normal``'s default). A
            pattern must clear both to be founder-eligible; this rejects
            high-$|h|$ error-tail patterns that clear the relative screen only
            because the locus is shallow (e.g. a 5-read locus where every
            singleton pattern clears ``min_pattern_freq``). Must match the
            value passed to the fit (``fit_normal``/``fit_normal_panel``) for
            the prediction to stay in sync with admission.
        min_families_run : int, default 5
            Frozen-run classification threshold. Within a
            multi-run founder pattern, a run is **active** (kept on the
            propagated length chain) only if $\\ge 2$ of its lengths each carry
            $\\ge$ ``min_families_run`` reads; otherwise it is **frozen**, pinned at
            its modal length and collapsed out of the scored support
            $K' = k_h^{n_{\\mathrm{active}}}$. Stored on the ``LocusData`` and is
            the single source of truth for the reduction, so it is fixed here at
            construction (the fit reads it back, never re-derives it). Higher →
            fewer runs freeze → less reduction (and a smaller length-channel
            approximation); lower → more aggressive collapse.
        max_founder_length_states : int, default 100_000
            Dense-kernel feasibility budget on the **scored** founder length-state
            count $K' = k_h^{n_{\\mathrm{active}}}$ (the frozen-run-reduced space
            the kernel actually materializes — *not* the nominal $k_h^{|h|}$). A
            locus is dropped to ``drops['overflow']`` when an admitted founder's
            $K'$ exceeds this. The default is the measured eager-kernel ceiling:
            scoring is bit-exact but compute-bound at
            $K' \\gtrsim 10^5$ (≈100–370 s/locus at $K' \\approx 2.5\\times10^5$),
            so the high-$|h|$ tail is dropped rather than fit. Raise it if a
            faster (e.g. compiled) kernel lands. (The int64 flat-tuple encoding
            ceiling is separate and fixed — see ``_MAX_INT64_FLAT``; a founder
            whose nominal $k_h^{|h|}$ exceeds it is un-encodable and also drops
            the locus.) Must match the value used to build the loci the fit
            consumes — this is a build-time drop with no fit-time re-check.
        phasing : dict[str, FlankingHaplotypeAssignment], optional
            Per-locus phasing assignments (output of
            :func:`prolyg_phasing.phasing.assign_flanking_haplotypes`),
            row-aligned to each locus's ``ExtractedLocus`` rows. When
            given, per-family maj/min read counts are folded into each
            ``LocusData`` (``Xphs_maj`` / ``Xphs_min``); absent loci and
            the ``None`` default leave those counts zero (unphased).
        require_anchorable : bool, default True
            Drop loci whose
            :attr:`ExtractedLocus.anchorability_status` is not
            ``"anchorable"``. Set ``False`` to retain non-anchorable
            loci (e.g., for diagnostic comparison against the
            anchorable subset). The ``ExtractedPanel`` itself is
            unchanged; ``panel.loci[lid]`` is the faithful BAM-side
            handle for any locus regardless of this flag.
        exclude_locus_ids : Collection[str], optional
            Locus ids to drop up front (before parsing/axis sizing),
            reported under ``drops["blacklisted"]``. The panel-level
            blacklist for loci in problematic genomic regions (ENCODE
            blacklist ∪ centromere/heterochromatin, from
            :func:`prolyG.cn.blacklist.loci_in_exclude`), where short-read
            mapping is unreliable in both the length and depth channels.
            Ids absent from the panel are ignored.
        verbose : bool, default False
            When True, show a tqdm bar over per-locus filtering.

        Returns
        -------
        ToLociResult
            ``locus_ids``, ``loci`` (never ``None``), ``n_alleles``,
            and a ``drops`` report keyed by reason (``"unfittable"`` /
            ``"overflow"`` / ``"insufficient_founder_reads"`` / ``"blacklisted"``,
            plus ``"not_anchorable"`` when ``require_anchorable``) → list of
            dropped locus_ids. Every listed key is present even when its list is
            empty.
        """
        # Panel-level blacklist: loci in problematic regions (ENCODE blacklist ∪
        # centromere/heterochromatin) are dropped up front — before parsing or axis
        # sizing — so they never enter the length channel. Short-read mapping is
        # unreliable there, which makes the locus's genotype untrustworthy. This is
        # the length channel's gate only: the CN depth track admits a locus on its
        # own family-count criterion and is not filtered by this list
        # (prolyG/cn/blacklist.py explains why not).
        exclude_set = set(exclude_locus_ids or ())
        all_ids = sorted(self.loci.keys())
        blacklisted_ids = [lid for lid in all_ids if lid in exclude_set]
        locus_ids = [lid for lid in all_ids if lid not in exclude_set]
        iterator = locus_ids
        if verbose:
            iterator = tqdm(locus_ids, desc="to_loci", unit="locus")
        parsed_per_locus: list[_ParsedLocusData | None] = []
        for locus_id in iterator:
            parsed_per_locus.append(
                _parse_locus(self.loci[locus_id], max_n_runs=max_n_runs)
            )

        # Single-run axis: max run of the single-run () pattern across all loci
        # + margin. Not the panel max-over-all-reads (a long multi-run/error
        # read must not inflate it) and not just the S=1 loci (a () founder at
        # an interrupter-het locus is single-run too and must be representable).
        single_run_max = max(
            (p.pattern_max_run.get((), 0) for p in parsed_per_locus if p is not None),
            default=0,
        )
        n_alleles_single = single_run_max + _N_ALLELES_MARGIN

        # Per-locus multi-run tight axes (k_h); n_alleles_max = max over the
        # single-run axis and the **founder-eligible** multi-run axes — the only
        # axes that ever size lam (off-pattern axes are storage-only). Single-run
        # patterns are pinned to n_alleles_max below so every |h|=1 slice shares
        # one uniform axis (the dominant single-run packed batch, K linear).
        multi_axes_per_locus: list[dict[tuple[str, ...], int] | None] = []
        n_alleles_max = n_alleles_single
        for parsed in parsed_per_locus:
            if parsed is None:
                multi_axes_per_locus.append(None)
                continue
            ma = _multi_run_pattern_axes(parsed)
            multi_axes_per_locus.append(ma)
            founders = _founder_eligible_patterns(
                parsed, min_pattern_freq, min_pattern_families,
            )
            for h in founders:
                if h in ma and ma[h] > n_alleles_max:
                    n_alleles_max = ma[h]

        # Every bucket is present whatever the inputs drop, so a caller can read a
        # count off any reason without first testing that the reason fired. The
        # blacklist bucket is the one that used to be input-dependent: it appeared
        # only when the file happened to name a locus this panel carries.
        drops: dict[str, list[str]] = {
            "unfittable": [], "overflow": [], "insufficient_founder_reads": [],
            "blacklisted": blacklisted_ids,
        }
        if require_anchorable:
            drops["not_anchorable"] = []
        kept_ids: list[str] = []
        kept_loci: list = []
        for lid, parsed, ma in zip(
            locus_ids, parsed_per_locus, multi_axes_per_locus, strict=True,
        ):
            if parsed is None:
                drops["unfittable"].append(lid)
                continue
            # Materialize the per-pattern axis dict: single-run () patterns at
            # the panel n_alleles_max; multi-run patterns at their tight axes.
            na_pp = {
                h: (n_alleles_max if len(h) + 1 == 1 else ma[h])
                for h in parsed.observed_patterns
            }
            locus = _build_locus(
                parsed,
                n_alleles_per_pattern=na_pp,
                min_families_run=min_families_run,
                min_pattern_freq=min_pattern_freq,
                min_pattern_families=min_pattern_families,
                phasing_assignment=phasing.get(lid) if phasing else None,
            )
            if locus is None:
                drops["unfittable"].append(lid)
                continue
            # No interrupter pattern cleared the founder screen (same
            # ``_founder_screen`` the fit uses, here via ``from_pattern_csr`` →
            # ``founder_cols``): the locus has no founder with sufficient reads,
            # so fitting it would fall back to a low-confidence modal pattern.
            # Drop rather than fit. (Single-pattern loci keep their sole founder
            # and never reach here.)
            if locus.founder_cols.numel() == 0:
                drops["insufficient_founder_reads"].append(lid)
                continue
            # Freeze-aware feasibility guard: an admitted founder whose scored
            # length-state count k_h^{n_active} exceeds the budget (or whose
            # full-|h| flat is un-encodable in int64) can't be scored. Keyed on
            # n_active, reusing the build's frozen classification — the nominal
            # k_h^{|h|} space is never materialized post-freeze.
            if _founder_overflow(locus, max_founder_length_states):
                drops["overflow"].append(lid)
                continue
            if (
                require_anchorable
                and self.loci[lid].anchorability_status != "anchorable"
            ):
                drops["not_anchorable"].append(lid)
                continue
            kept_ids.append(lid)
            kept_loci.append(locus)

        # Per-locus p^m anchor ℓ_ref,i / founder-region half-extents
        # Δ_up,i, Δ_down,i, derived once here from the
        # ExtractedLocus reads — the single source of truth for the anchor and
        # box, byte-identical to the overdispersion width statistic (same
        # max-of-runs emp_total_histogram, same dominant-basin walk, same
        # ``min_pattern_freq``). The fits read these off ToLociResult rather than
        # re-deriving from the inference-side CSR (whose MR ``ell`` is a
        # flat-tuple index, not a length). Lazy import: overdispersion imports
        # from this module.
        from prolyG.inference.overdispersion import panel_ref_and_delta

        ell_ref, delta_up, delta_down = panel_ref_and_delta(
            [self.loci[lid] for lid in kept_ids],
            n_alleles=n_alleles_max,
            min_pattern_freq=min_pattern_freq,
        )

        return ToLociResult(
            locus_ids=kept_ids,
            loci=kept_loci,
            n_alleles=n_alleles_max,
            drops=drops,
            ell_ref=ell_ref,
            delta_up=delta_up,
            delta_down=delta_down,
            min_pattern_freq=min_pattern_freq,
        )


# ---------------------------------------------------------------------------
# Tuple parse (uniform)
# ---------------------------------------------------------------------------


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
    # `re.split` on non-G stretches yields one entry per G-run with
    # empty strings at boundaries when seq starts/ends with non-G.
    runs = re.split(r"[^G]+", seq)
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
    return tuple(re.findall(r"[^G]+", seq))


# ---------------------------------------------------------------------------
# Observed interrupter-pattern frequencies (descriptive; plotting/diagnostics)
#
# These helpers summarize the per-read-family majority interrupter pattern.
# The inference adapter (``to_loci``) no longer consults them — per-read
# tabulation keeps every pattern and founder selection is a fit-time screen
# (the ``min_haplotype_freq`` extraction filter was retired).
# They remain as descriptive utilities for plotting and diagnostics.
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
    # Normalized value-count of the per-MI majority assignment (the single
    # source of the accumulation + majority-vote, shared with the companion).
    assignment = majority_pattern_per_rf(locus)
    n_rfs = len(assignment)
    if n_rfs == 0:
        return {}
    pattern_n_rfs: dict[tuple[str, ...], int] = defaultdict(int)
    for majority in assignment.values():
        pattern_n_rfs[majority] += 1
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
    mi_pattern_count: dict[str, dict[tuple[str, ...], int]] = defaultdict(
        lambda: defaultdict(int),
    )
    for i in range(len(locus.mi)):
        mi = str(locus.mi[i])
        pat = interrupter_pattern(str(locus.seq[i]))
        mi_pattern_count[mi][pat] += int(locus.count[i])

    out: dict[str, tuple[str, ...]] = {}
    for mi, pattern_count in mi_pattern_count.items():
        max_count = max(pattern_count.values())
        candidates = [h for h, c in pattern_count.items() if c == max_count]
        out[mi] = min(candidates)
    return out


# ---------------------------------------------------------------------------
# to_loci helpers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _ParsedLocusData:
    """Pass-1 result for one locus: every read parsed by (pattern, run vector).

    Holds per-row parsed interrupter patterns and run-length tuples
    (base-independent) so Pass 2 in :meth:`ExtractedPanel.to_loci` can
    assign per-locus pattern indices and encode flat-tuple length indices
    at the panel-tight ``n_alleles`` chosen after scanning every locus.
    Rows stay in the original ``ExtractedLocus`` order so a row-aligned
    phasing assignment can be aggregated in Pass 2.
    """
    bed_name: str                                   # locus identifier (errors)
    mi: list[str]                                   # per-row family ID
    strand: list[str]                               # per-row strand
    pattern: list[tuple[str, ...]]                  # per-row interrupter pattern h(r)
    runs: list[tuple[int, ...]]                     # per-row run-length vector
    count: list[int]                                # per-row count
    observed_patterns: tuple[tuple[str, ...], ...]  # sorted distinct patterns P_i
    n_runs: int                                     # max |h| over observed patterns
    max_run_length: int                             # max over all rows' run components
    pattern_max_run: dict[tuple[str, ...], int]     # per-pattern max run component (k_h source)


def _parse_locus(
    locus: ExtractedLocus,
    *,
    max_n_runs: int | None,
) -> _ParsedLocusData | None:
    """Pass 1: parse every read into (interrupter pattern, run-length vector).

    No filtering beyond the locus-level ``max_n_runs`` cap: each accepted
    read is tabulated by its own parsed pattern $h(r)$ and run-length
    vector $\\boldsymbol{\\ell}(r)$. The pre-Scope-08 per-family equality
    filter and majority-pattern assignment are gone — off-founder reads are
    kept and scored by the interrupter channel at fit time. Returns
    ``None`` for an empty locus or one whose max run count exceeds
    ``max_n_runs``.

    No flat-index encoding here — the parse is base-independent, so the
    panel-tight ``n_alleles`` can be chosen after every locus has been
    scanned.
    """
    n_rows = len(locus.mi)
    if n_rows == 0:
        return None

    mis_out: list[str] = []
    strands_out: list[str] = []
    patterns_out: list[tuple[str, ...]] = []
    runs_out: list[tuple[int, ...]] = []
    counts_out: list[int] = []
    distinct: set[tuple[str, ...]] = set()
    max_run_length = 0
    pattern_max_run: dict[tuple[str, ...], int] = defaultdict(int)
    for i in range(n_rows):
        seq = str(locus.seq[i])
        pattern = interrupter_pattern(seq)
        runs = parse_run_lengths(seq)
        mis_out.append(str(locus.mi[i]))
        strands_out.append(str(locus.strand[i]))
        patterns_out.append(pattern)
        runs_out.append(runs)
        counts_out.append(int(locus.count[i]))
        distinct.add(pattern)
        local = max(runs) if runs else 0
        if local > max_run_length:
            max_run_length = local
        if local > pattern_max_run[pattern]:
            pattern_max_run[pattern] = local

    observed_patterns = tuple(sorted(distinct))
    n_runs = max((len(h) + 1 for h in observed_patterns), default=1)
    if max_n_runs is not None and n_runs > max_n_runs:
        return None

    return _ParsedLocusData(
        bed_name=locus.bed_name,
        mi=mis_out, strand=strands_out, pattern=patterns_out,
        runs=runs_out, count=counts_out,
        observed_patterns=observed_patterns, n_runs=n_runs,
        max_run_length=max_run_length,
        pattern_max_run=dict(pattern_max_run),
    )


def _axis_binned_rows(
    locus: ExtractedLocus,
    *,
    n_alleles: int,
    min_pattern_freq: float,
    max_n_runs: int | None,
):
    """The on-pattern, on-axis reads of one locus, as ``(mi, pattern, allele_len, count)``.

    The single binning convention behind both the pattern-keyed histogram
    (:func:`_build_empirical`) and the family-keyed one
    (:func:`_build_empirical_by_family`), so a caller that needs reads grouped by
    read family does not re-derive which reads are on the axis. A read is kept iff
    its family's majority pattern clears ``min_pattern_freq``, its run count matches
    that pattern, and its longest run lands below ``n_alleles``.

    Returns ``(rows, parsed, pattern_freqs, dom_pattern)``, or ``None`` when the
    locus is dropped by the parse or has no observed pattern. ``rows`` may be empty.
    """
    pf = _parse_locus(locus, max_n_runs=max_n_runs)
    if pf is None:
        return None

    pat_freqs = observed_pattern_frequencies(locus)
    if not pat_freqs:
        return None
    dom_pattern = max(pat_freqs, key=pat_freqs.get)
    majority = majority_pattern_per_rf(locus)
    kept = select_patterns_above_freq(locus, min_freq=min_pattern_freq)

    rows: list[tuple[str, tuple[str, ...], int, int]] = []
    for mi, run_tuple, count in zip(pf.mi, pf.runs, pf.count, strict=True):
        pattern = majority[mi]
        if pattern not in kept:
            continue
        # On-pattern reads only: the read's run count matches its family's
        # majority pattern (off-pattern reads ride the interrupter channel).
        if len(run_tuple) != len(pattern) + 1:
            continue
        if not run_tuple:
            continue
        allele_len = int(max(run_tuple))
        if allele_len >= n_alleles:
            continue
        rows.append((mi, pattern, allele_len, int(count)))
    return rows, pf, pat_freqs, dom_pattern


def _build_empirical_by_family(
    locus: ExtractedLocus,
    *,
    n_alleles: int,
    min_pattern_freq: float,
    max_n_runs: int | None,
) -> np.ndarray | None:
    """Family-keyed counterpart of :func:`_build_empirical`: one row per read family.

    Returns an ``(n_families, n_alleles)`` count matrix whose **column sums are
    exactly** the ``emp_total`` histogram (pinned by test), or ``None`` when no read
    is binned. The rows carry the within-family length composition that summing to
    ``emp_total`` destroys: reads of one family repeat a length, so a length
    histogram is a clustered sample, not the independent one a plain multinomial
    assumes. Each family maps to a single pattern (its majority), so no family is
    split across rows.
    """
    binned = _axis_binned_rows(
        locus, n_alleles=n_alleles, min_pattern_freq=min_pattern_freq,
        max_n_runs=max_n_runs,
    )
    if binned is None:
        return None
    rows = binned[0]
    if not rows:
        return None

    row_of: dict[str, int] = {}
    for mi, _pattern, _allele_len, _count in rows:
        if mi not in row_of:
            row_of[mi] = len(row_of)
    mat = np.zeros((len(row_of), n_alleles), dtype=np.float64)
    for mi, _pattern, allele_len, count in rows:
        mat[row_of[mi], allele_len] += count
    return mat


def _build_empirical(
    locus: ExtractedLocus,
    *,
    n_alleles: int,
    min_pattern_freq: float,
    max_n_runs: int | None,
):
    """Per-pattern max-of-runs read histograms on the founder length axis.

    Returns ``(emp_per_pattern, pattern_freqs, dom_pattern, n_reads, n_runs)``
    or ``None`` if the locus is dropped / has no displayable reads. Each
    ``emp_per_pattern[pattern]`` is a ``(n_alleles,)`` count histogram binned by
    the read's **longest run**; the panel-wide total ``sum(emp_per_pattern)`` is
    the empirical stutter cloud used by the plotting views (``emp_total``) and
    the overdispersion partition (:func:`prolyG.inference.overdispersion`).

    Only **on-pattern** reads are binned — a read whose run count matches its
    family's majority pattern (off-pattern reads are scored by the interrupter
    channel at fit time, not the per-pattern length histogram). ``min_pattern_freq``
    is a per-read-family share threshold that omits low-share patterns; reads
    whose longest run lands at or beyond ``n_alleles`` are off-axis and dropped.

    Single source of truth for the empirical histogram (consumed by both
    ``plotting`` and ``inference.overdispersion``) — do not re-derive the
    binning convention elsewhere. The reads are selected by
    :func:`_axis_binned_rows`; this function only chooses the grouping.
    """
    binned = _axis_binned_rows(
        locus, n_alleles=n_alleles, min_pattern_freq=min_pattern_freq,
        max_n_runs=max_n_runs,
    )
    if binned is None:
        return None
    rows, pf, pat_freqs, dom_pattern = binned

    emp_per_pattern: dict[tuple[str, ...], np.ndarray] = {}
    for _mi, pattern, allele_len, count in rows:
        arr = emp_per_pattern.get(pattern)
        if arr is None:
            arr = np.zeros(n_alleles, dtype=np.float64)
            emp_per_pattern[pattern] = arr
        arr[allele_len] += count

    if not emp_per_pattern:
        return None

    n_reads = int(sum(arr.sum() for arr in emp_per_pattern.values()))
    return emp_per_pattern, pat_freqs, dom_pattern, n_reads, int(pf.n_runs)


# Per-pattern length-axis (k_h) tuning.
_N_ALLELES_MARGIN = 4   # headroom above a pattern's max realized run length
_AXIS_BUCKET = 4        # multi-run k_h rounded up to this, capping compile slots


def _bucket_up(n: int, step: int) -> int:
    """Round ``n`` up to the next multiple of ``step`` (``step >= 1``)."""
    return ((n + step - 1) // step) * step


def _founder_eligible_patterns(
    parsed: _ParsedLocusData, min_pattern_freq: float, min_pattern_families: int = 1,
) -> set[tuple[str, ...]]:
    """Patterns clearing the family-share and family-count screens (empty if none).

    Mirrors :func:`prolyG.inference.pcr_params.admitted_founder_patterns`
    (read-family share over both strands **and** absolute family count; a
    multi-pattern locus where the joint screen clears nothing yields the empty
    set and is dropped at build, not fit on a fallback) on the pass-1
    parse, so ``to_loci`` can predict which patterns the fit will admit as
    founders — the patterns whose length axis must be lossless.
    ``min_pattern_families`` is the absolute-count companion to
    ``min_pattern_freq`` (default ``1`` ⇒ no-op; the policy default lives in
    :meth:`ExtractedPanel.to_loci`). Returns the set of founder-eligible
    patterns (empty for a read-free locus, or a multi-pattern locus where no
    pattern clears the screen).

    Each family votes once, for its own read-weighted majority pattern — the
    same rule as :func:`observed_pattern_frequencies` and as the fit-time
    counterpart, so the two agree family for family on the same rows.

    The pattern-admission screen only (the relative + absolute count screen) —
    this is a build-time *prediction*; the fit additionally applies
    the frozen-run reduction re-screen on the built classification
    (:func:`prolyG.inference.pcr_params.admitted_founder_patterns`). Both share
    the single :func:`prolyG.inference.pcr_params._founder_screen` core, so the
    prediction stays a superset of the fit's admitted set.
    """
    from prolyG.inference.pcr_params import _founder_screen

    patterns = parsed.observed_patterns
    pattern_to_index = {p: i for i, p in enumerate(patterns)}
    # Per-family read weight over patterns, then one vote each at its argmax.
    per_family: dict[str, np.ndarray] = {}
    for mi, pattern, count in zip(
        parsed.mi, parsed.pattern, parsed.count, strict=True,
    ):
        w = per_family.get(mi)
        if w is None:
            w = per_family[mi] = np.zeros(len(patterns), dtype=np.float64)
        w[pattern_to_index[pattern]] += count
    counts = np.zeros(len(patterns), dtype=np.float64)
    for w in per_family.values():
        if w.sum() > 0:
            counts[int(w.argmax())] += 1     # ties → lower index (patterns sorted)
    if counts.sum() <= 0:
        return set()
    return {
        patterns[i]
        for i in _founder_screen(counts, min_pattern_freq, min_pattern_families)
    }


def _multi_run_pattern_axes(
    parsed: _ParsedLocusData,
) -> dict[tuple[str, ...], int]:
    """Per-pattern length-axis size $k_h$ for the locus's multi-run patterns.

    Each multi-run pattern ($|h| > 1$) gets ``max realized run of the pattern +
    margin``, rounded up to ``_AXIS_BUCKET`` (the bucket caps the distinct
    $(|h|, k_h)$ count that keys the compiled kernel, hence the
    ``torch.compile`` warm-up count). Lossless for every pattern's own reads by
    construction (radix > its max run); the budget sentinel in
    :func:`_build_locus` handles a pattern whose dense $K = k_h^{|h|}$ is
    infeasible (off-pattern on any kept locus). Single-run patterns are **not**
    sized here — they take the panel ``n_alleles_max`` (see
    :meth:`ExtractedPanel.to_loci`) so every $|h| = 1$ slice shares one uniform
    axis and the single-run packed batch stays one group ($K$ linear there).
    """
    return {
        h: _bucket_up(
            parsed.pattern_max_run[h] + _N_ALLELES_MARGIN, _AXIS_BUCKET,
        )
        for h in parsed.observed_patterns
        if len(h) + 1 > 1
    }


def _founder_overflow(locus, max_founder_length_states: int) -> bool:
    """True if an admitted founder cannot be scored — feasibility or encoding.

    Checked **post-build** on the admitted founders (``locus.founder_cols``, the
    same two-phase ``_founder_screen`` the fit uses) so it reuses the build's
    frozen classification rather than recomputing it. A locus is dropped to
    ``drops['overflow']`` when, for any admitted founder $h$:

    - **Feasibility** — the *scored* length-state count
      $K' = k_h^{n_{\\mathrm{active}}}$ exceeds ``max_founder_length_states``.
      Keyed on the frozen-run-reduced ``n_active``, not the nominal $|h|$: the
      dense kernel only ever materializes the reduced ``FounderSlice``
      ($K'$), so the nominal $k_h^{|h|}$ is irrelevant to feasibility (this is
      the freeze-aware fix — the nominal test was too aggressive). Or
    - **Encoding** — the *nominal* full-$|h|$ flat $k_h^{|h|}$ exceeds
      ``_MAX_INT64_FLAT``, so the founder's $\\ell$ is un-encodable in the int64
      ``XA_ell`` (and was sentinelled to 0 in :func:`_build_locus`) — it can't be
      decoded back to a reduced slice. A fixed dtype ceiling, not the budget.
    """
    fc = locus.founder_cols
    for j in range(fc.numel()):
        h = int(fc[j].item())
        k_h = int(locus.n_alleles_per_pattern[h].item())
        n_active = int(locus.n_active_runs_per_pattern[h].item())
        n_runs = int(locus.n_runs_per_pattern[h].item())  # |h|
        if k_h ** n_active > max_founder_length_states:
            return True
        if k_h ** n_runs > _MAX_INT64_FLAT:
            return True
    return False


def _build_locus(
    parsed: _ParsedLocusData,
    *,
    n_alleles_per_pattern: dict[tuple[str, ...], int],
    min_families_run: int = 5,
    min_pattern_freq: float = 0.0,
    min_pattern_families: int = 0,
    phasing_assignment=None,
):
    """Pass 2: assign pattern indices, encode flat indices, emit ``LocusData``.

    Each read's flat-tuple length index is ``flat = sum_s ell_s * k_h**s`` for
    ``s ∈ [0, |h|)`` in its pattern $h$'s own subspace, at that pattern's own
    radix $k_h$ = ``n_alleles_per_pattern[h]``. Builds a
    pattern-keyed per-strand CSR over families (sorted MI order), the
    $(H \\times H)$ interrupter edit table (inside
    :meth:`LocusData.from_pattern_csr`), and per-family phasing counts from
    a row-aligned ``phasing_assignment`` (a
    :class:`prolyg_phasing.phasing.FlankingHaplotypeAssignment`; zeros
    when ``None``). Returns ``None`` when no rows carry counts.

    A pattern whose nominal full-$|h|$ flat $k_h^{|h|}$ exceeds the int64 encoding
    ceiling (``_MAX_INT64_FLAT``) has its reads' ``ell`` **sentinelled to 0** —
    the flat is otherwise un-representable. Lossless on every kept locus: such a
    pattern is off-founder there (an admitted founder over the ceiling drops the
    locus, :func:`_founder_overflow`), and no channel reads a non-founder
    pattern's length tuple (the length channel is founder-gated via
    ``_slice_pattern_csr``; the interrupter channel reads only pattern counts).
    Note this is the **encoding** ceiling, not the feasibility budget — the
    latter is a post-build founder check keyed on the reduced $K'$.
    """
    from prolyG.inference.pcr_params import LocusData

    if not parsed.mi:
        return None

    patterns = parsed.observed_patterns
    pattern_to_index = {h: idx for idx, h in enumerate(patterns)}

    # Patterns whose nominal full-|h| flat k_h**|h| overflows int64 can't be
    # encoded in XA_ell (a read with more runs than its founder carries
    # interrupter miscalls — PCR slippage cannot add or remove a run, glossary).
    # On a kept locus such a pattern is off-founder (an admitted founder over the
    # ceiling drops the locus post-build via ``_founder_overflow``), so the
    # founder-gated length channel never reads its ``ell`` and the interrupter
    # channel reads only pattern counts. We sentinel its ``ell`` to 0 rather
    # than overflowing: lossless on every kept locus, encode-safe at any |h|.
    overflow_pidx = {
        pattern_to_index[h]
        for h in patterns
        if n_alleles_per_pattern[h] ** (len(h) + 1) > _MAX_INT64_FLAT
    }

    # Per-family per-strand cells keyed by (pattern index, flat-tuple index).
    mi_to_cells_a: dict[str, dict[tuple[int, int], int]] = defaultdict(
        lambda: defaultdict(int)
    )
    mi_to_cells_b: dict[str, dict[tuple[int, int], int]] = defaultdict(
        lambda: defaultdict(int)
    )
    all_mis_set: set[str] = set()
    for mi_str, strand, pattern, runs, count in zip(
        parsed.mi, parsed.strand, parsed.pattern, parsed.runs, parsed.count,
        strict=True,
    ):
        all_mis_set.add(mi_str)
        h_idx = pattern_to_index[pattern]
        if h_idx in overflow_pidx:
            flat = 0  # sentinel: off-pattern-only, never read (see above)
        else:
            radix = n_alleles_per_pattern[pattern]
            flat = 0
            for s, r in enumerate(runs):
                flat += r * (radix ** s)
        if strand == "A":
            mi_to_cells_a[mi_str][(h_idx, flat)] += count
        elif strand == "B":
            mi_to_cells_b[mi_str][(h_idx, flat)] += count

    all_mis = sorted(all_mis_set)
    mi_to_row = {m: i for i, m in enumerate(all_mis)}

    def _csr(mi_to_cells: dict[str, dict[tuple[int, int], int]]):
        offsets = np.zeros(len(all_mis) + 1, dtype=np.int64)
        h_chunks: list[np.ndarray] = []
        ell_chunks: list[np.ndarray] = []
        v_chunks: list[np.ndarray] = []
        for i, m in enumerate(all_mis):
            cells = mi_to_cells.get(m, {})
            keys = sorted(cells.keys())  # lex sort by (pattern index, flat)
            offsets[i + 1] = offsets[i] + len(keys)
            if keys:
                h_chunks.append(np.asarray([k[0] for k in keys], dtype=np.int64))
                ell_chunks.append(np.asarray([k[1] for k in keys], dtype=np.int64))
                v_chunks.append(np.asarray([cells[k] for k in keys], dtype=np.int64))
        if h_chunks:
            return (
                offsets,
                np.concatenate(h_chunks),
                np.concatenate(ell_chunks),
                np.concatenate(v_chunks),
            )
        empty = np.empty(0, dtype=np.int64)
        return offsets, empty, empty.copy(), empty.copy()

    XA_offsets, XA_h, XA_ell, XA_v = _csr(mi_to_cells_a)
    XB_offsets, XB_h, XB_ell, XB_v = _csr(mi_to_cells_b)

    # Per-family phasing counts (maj/min reads; unk excluded), aggregated
    # from the row-aligned assignment. Zeros when no phasing was supplied.
    n_rfs = len(all_mis)
    Xphs_maj = np.zeros(n_rfs, dtype=np.int64)
    Xphs_min = np.zeros(n_rfs, dtype=np.int64)
    if phasing_assignment is not None:
        from prolyg_phasing.phasing import LABEL_MAJOR, LABEL_MINOR
        labels = np.asarray(phasing_assignment.label)
        if len(labels) != len(parsed.mi):
            raise ValueError(
                "phasing_assignment.label is not row-aligned to the parsed "
                f"rows: {len(labels)} labels vs {len(parsed.mi)} rows. The "
                "assignment must be built from the same locus passed to "
                "to_loci (one label per ExtractedLocus row, in order)."
            )
        for i, mi_str in enumerate(parsed.mi):
            lbl = int(labels[i])
            row = mi_to_row[mi_str]
            if lbl == LABEL_MAJOR:
                Xphs_maj[row] += parsed.count[i]
            elif lbl == LABEL_MINOR:
                Xphs_min[row] += parsed.count[i]

    return LocusData.from_pattern_csr(
        patterns,
        XA_offsets, XA_h, XA_ell, XA_v,
        XB_offsets, XB_h, XB_ell, XB_v,
        n_alleles_per_pattern=[n_alleles_per_pattern[h] for h in patterns],
        min_families_run=min_families_run,
        min_pattern_freq=min_pattern_freq,
        min_pattern_families=min_pattern_families,
        Xphs_maj=Xphs_maj,
        Xphs_min=Xphs_min,
    )


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


