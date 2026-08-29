"""BAM panel extraction — `extract_panel` entry point.

Walks a BAM against a BED panel and produces an in-memory
`ExtractedPanel` of faithful per-(MI, strand, seq) read counts.

Two passes:

1. **Anchorability pass** — per BED interval, reconstruct a `±pad` bp
   reference window from MD tags and try to construct upstream +
   downstream anchors (`_anchors.build_anchors`).
2. **Extraction pass** — per anchorable locus, fetch overlapping
   alignments, apply parser-side QC filters (MAPQ, SA, TLEN, anchor
   placement), group by `query_name` for pair-collapse, drop pair
   disagreements, and emit per-(MI, strand, seq) read counts.

Per-locus ``n_runs`` is set from the **data** (max parsed tuple length
across accepted reads), not from the reference inter-anchor sequence.
This handles interrupter-pattern polymorphisms where the reference
predicts one ``S`` but the sample carries reads at multiple ``S``
values (e.g. PolyG3298: ~49% of reads parse to 2-tuples, ~46% to
1-tuples, both biologically real). The reference-derived
``reference_run_lengths`` is retained as diagnostic context.

`n_alleles` is panel-wide and decided at extraction time:
`n_alleles = max_observed_run_length + n_alleles_margin`, where
`max_observed_run_length` is the maximum G-run length seen in any
parsed read across the panel.
"""

from __future__ import annotations

import datetime
from collections import Counter, defaultdict
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

import numpy as np
import pysam
from tqdm.auto import tqdm

from prolyg_phasing.io._anchors import (
    AnchorResult,
    build_anchors,
    get_reference_window,
    hamming_search,
    revcomp,
)
from prolyg_phasing.io.panel import (
    ExtractedLocus,
    ExtractedPanel,
    ExtractionProvenance,
    parse_run_lengths,
)

FLANKING_SEQ_LEN = 10
REF_WINDOW_PAD = 50

EXTRACTION_SCHEMA_VERSION = "2"
DEFAULT_MIN_FLANKING_BASE_Q = 20

_COMP_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


# ---------------------------------------------------------------------------
# BED loading
# ---------------------------------------------------------------------------


def _load_bed(bed_path: Path) -> list[tuple[str, str, int, int]]:
    """Load a 4-column BED: `(name, chrom, start, end)` per row."""
    rows: list[tuple[str, str, int, int]] = []
    with bed_path.open() as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            chrom, start, end, name = line.rstrip("\n").split("\t")[:4]
            rows.append((name, chrom, int(start), int(end)))
    return rows


# ---------------------------------------------------------------------------
# Per-locus anchorability + extraction
# ---------------------------------------------------------------------------


def _build_anchors_for_locus(
    bam: pysam.AlignmentFile,
    chrom: str,
    bed_start: int,
    bed_end: int,
    *,
    pad: int,
    k: int,
    min_mapq: int,
) -> AnchorResult:
    """Anchor extraction for one BED interval: MD-tag reference + anchor walk."""
    win_start = bed_start - pad
    win_end = bed_end + pad
    ref = get_reference_window(bam, chrom, win_start, win_end, min_mapq=min_mapq)
    return build_anchors(
        ref_window=ref,
        bed_offset=pad,
        bed_len=bed_end - bed_start,
        k=k,
    )


def _alignment_flanking(
    r: pysam.AlignedSegment,
    *,
    bed_start: int,
    bed_end: int,
    g_walk_up: int,
    g_walk_dn: int,
    anchor_k: int,
    ref_orientation: str,
    min_base_q: int,
) -> tuple[dict[int, str], dict[int, str]]:
    """Flanking bases for one alignment, keyed by canonical-distance from the anchor.

    Returns ``({up_dist: base}, {dn_dist: base})``. ``up_dist=0`` is the
    base immediately adjacent to the upstream anchor going outward (away
    from the polyG) in canonical orientation; ``dn_dist=0`` is the same
    for the downstream side. Bases are returned in canonical orientation
    (revcomp'd for ``"-"`` orientation loci). Indels are skipped
    (positions where either ``qpos`` or ``refpos`` is ``None``). Bases
    with ``query_qualities[qpos] < min_base_q`` are skipped.
    """
    if r.query_sequence is None or r.query_qualities is None:
        return {}, {}
    seq = r.query_sequence
    quals = r.query_qualities

    if ref_orientation == "+":
        up_anchor_ref_start = bed_start - g_walk_up - anchor_k
        dn_anchor_ref_end = bed_end + g_walk_dn + anchor_k

        def to_up(refpos: int) -> int | None:
            return (up_anchor_ref_start - 1 - refpos
                    if refpos < up_anchor_ref_start else None)

        def to_dn(refpos: int) -> int | None:
            return (refpos - dn_anchor_ref_end
                    if refpos >= dn_anchor_ref_end else None)

        def canonicalize(b: str) -> str:
            return b
    else:
        up_anchor_ref_end = bed_end + g_walk_up + anchor_k
        dn_anchor_ref_start = bed_start - g_walk_dn - anchor_k

        def to_up(refpos: int) -> int | None:
            return (refpos - up_anchor_ref_end
                    if refpos >= up_anchor_ref_end else None)

        def to_dn(refpos: int) -> int | None:
            return (dn_anchor_ref_start - 1 - refpos
                    if refpos < dn_anchor_ref_start else None)

        def canonicalize(b: str) -> str:
            return b.translate(_COMP_TABLE)

    up: dict[int, str] = {}
    dn: dict[int, str] = {}
    for qpos, refpos in r.get_aligned_pairs(with_seq=False):
        if qpos is None or refpos is None:
            continue
        if quals[qpos] < min_base_q:
            continue
        base = seq[qpos].upper()
        d = to_up(refpos)
        if d is not None:
            up[d] = canonicalize(base)
            continue
        d = to_dn(refpos)
        if d is not None:
            dn[d] = canonicalize(base)
    return up, dn


def _merge_pair_flanking(
    mates: list[tuple[dict[int, str], dict[int, str]]],
) -> tuple[dict[int, str], dict[int, str]]:
    """Per-position consensus over up-to-2 mates' flanking dicts.

    Agreement keeps the base; disagreement drops the position. Singleton
    input returns its single mate's dicts unchanged. Empty input returns
    ``({}, {})``.
    """
    if not mates:
        return {}, {}
    if len(mates) == 1:
        return mates[0]
    (up_a, dn_a), (up_b, dn_b) = mates

    def _consensus(a: dict[int, str], b: dict[int, str]) -> dict[int, str]:
        out: dict[int, str] = {}
        for pos in set(a) | set(b):
            ba = a.get(pos)
            bb = b.get(pos)
            if ba is None:
                out[pos] = bb  # type: ignore[assignment]
            elif bb is None:
                out[pos] = ba
            elif ba == bb:
                out[pos] = ba
        return out

    return _consensus(up_a, up_b), _consensus(dn_a, dn_b)


def _extract_locus(
    bam: pysam.AlignmentFile,
    chrom: str,
    bed_start: int,
    bed_end: int,
    *,
    ref_orientation: str,
    anchor_up: str,
    anchor_dn: str,
    anchor_k: int,
    g_walk_up: int,
    g_walk_dn: int,
    min_mapq: int,
    drop_chimeric: bool,
    max_tlen: int,
    anchor_hamming_max: int,
    min_base_q: int,
) -> tuple[Counter, list[str], int, int, dict]:
    """Extract reads at one locus.

    Returns ``(read_counts, flanking_seqs, flanking_up_width,
    flanking_dn_width, qc)``. ``read_counts`` is a
    ``Counter[(mi, strand, seq, flanking_id)]`` of accepted reads
    (post-pair-collapse); ``flanking_seqs`` is the per-locus deduped
    flanking string list (indexed by ``flanking_id``), each of the form
    ``up_seq + "|" + dn_seq`` with ``.`` for uncovered/low-Q positions.
    ``flanking_up_width`` / ``flanking_dn_width`` are the per-locus
    extents observed.
    """
    qc = {
        "n_alignments_overlap_locus": 0,
        "n_alignments_drop_mapq": 0,
        "n_alignments_drop_sa": 0,
        "n_alignments_drop_tlen": 0,
        "n_alignments_drop_anchor": 0,
        "n_alignments_flanking_only_kept": 0,
        "n_alignments_orphan_no_bed_mate": 0,
        "n_read_pairs": 0,
        "n_read_pairs_drop_disagree": 0,
    }

    # Wide fetch: ``bed ± max_tlen`` so the mate of a bed-spanning read is
    # visible even if its alignment lies fully outside the polyG bed. With
    # paired-end Illumina at insert sizes up to ``max_tlen``, a fragment's
    # far mate sits at most ``max_tlen`` bp from any base of its pair, so
    # this window captures every relevant mate. Bed-spanning vs
    # flanking-only is decided per-alignment by reference-interval
    # overlap; the anchor check runs only on bed-spanning mates, since a
    # flanking-only mate cannot contain both anchors by construction.
    fetch_start = max(0, bed_start - max_tlen)
    fetch_end = bed_end + max_tlen

    # First pass: collect per-alignment data keyed by query_name for
    # pair-collapse. Each entry is a 6-tuple:
    #   (kind, mi_prefix, strand, ia_seq, up_dict, dn_dict)
    # where kind ∈ {"bed", "flank"}; for "flank" mates, mi_prefix /
    # strand / ia_seq are ``None`` (the bed-spanning mate establishes
    # them in the pair-collapse pass).
    by_qn: dict[
        str,
        list[tuple[str, str | None, str | None, str | None, dict[int, str], dict[int, str]]],
    ] = defaultdict(list)

    for r in bam.fetch(chrom, fetch_start, fetch_end):
        if r.is_secondary or r.is_supplementary or r.is_unmapped:
            continue
        if r.reference_end is None:
            continue
        bed_overlap = r.reference_start < bed_end and r.reference_end > bed_start

        # Alignment-level filters apply to both kinds.
        if r.mapping_quality < min_mapq:
            if bed_overlap:
                qc["n_alignments_overlap_locus"] += 1
                qc["n_alignments_drop_mapq"] += 1
            continue
        if drop_chimeric and r.has_tag("SA"):
            if bed_overlap:
                qc["n_alignments_overlap_locus"] += 1
                qc["n_alignments_drop_sa"] += 1
            continue
        tlen = abs(r.template_length) if r.template_length else 0
        if tlen > max_tlen:
            if bed_overlap:
                qc["n_alignments_overlap_locus"] += 1
                qc["n_alignments_drop_tlen"] += 1
            continue
        if r.query_sequence is None:
            continue

        if bed_overlap:
            qc["n_alignments_overlap_locus"] += 1
            if not r.has_tag("MI"):
                continue

            seq = r.query_sequence.upper()
            if ref_orientation == "-":
                seq = revcomp(seq)

            h_up, off_up = hamming_search(seq, anchor_up)
            h_dn, off_dn = hamming_search(seq, anchor_dn)
            up_ok = h_up <= anchor_hamming_max
            dn_ok = h_dn <= anchor_hamming_max
            if not (up_ok and dn_ok):
                qc["n_alignments_drop_anchor"] += 1
                continue

            ia_start = off_up + anchor_k
            ia_end = off_dn
            if ia_end < ia_start:
                qc["n_alignments_drop_anchor"] += 1
                continue
            ia_seq = seq[ia_start:ia_end]

            try:
                mi_tag = r.get_tag("MI")
            except KeyError:
                continue
            mi_prefix, _, strand = str(mi_tag).rpartition("/")
            if strand not in ("A", "B") or not mi_prefix:
                continue

            up_flank, dn_flank = _alignment_flanking(
                r,
                bed_start=bed_start, bed_end=bed_end,
                g_walk_up=g_walk_up, g_walk_dn=g_walk_dn,
                anchor_k=anchor_k,
                ref_orientation=ref_orientation,
                min_base_q=min_base_q,
            )
            by_qn[r.query_name].append(
                ("bed", mi_prefix, strand, ia_seq, up_flank, dn_flank)
            )
        else:
            up_flank, dn_flank = _alignment_flanking(
                r,
                bed_start=bed_start, bed_end=bed_end,
                g_walk_up=g_walk_up, g_walk_dn=g_walk_dn,
                anchor_k=anchor_k,
                ref_orientation=ref_orientation,
                min_base_q=min_base_q,
            )
            if not up_flank and not dn_flank:
                continue
            by_qn[r.query_name].append(
                ("flank", None, None, None, up_flank, dn_flank)
            )

    # Pair-collapse: group by query_name. Drop >2-mate groups. Drop qns
    # with no bed-spanning mate (the fragment never reached the polyG
    # and can't be assigned an MI/strand/length). Pair-disagreement
    # check only applies when both mates are bed-spanning. Flanking
    # dicts are pair-merged with per-position consensus (drop on
    # disagreement); a bed-spanning + flanking-only pair has no
    # disagreement possible at any single position the flanking-only
    # mate covers — both feed into the merge as today.
    accepted: list[tuple[str, str, str, dict[int, str], dict[int, str]]] = []
    for _qn, mates in by_qn.items():
        if len(mates) > 2:
            continue
        bed_mates = [m for m in mates if m[0] == "bed"]
        flank_mates = [m for m in mates if m[0] == "flank"]
        if not bed_mates:
            qc["n_alignments_orphan_no_bed_mate"] += len(flank_mates)
            continue
        prefs = {m[1] for m in bed_mates}
        strands = {m[2] for m in bed_mates}
        if len(prefs) != 1 or len(strands) != 1:
            continue
        qc["n_read_pairs"] += 1
        if len(bed_mates) == 2 and bed_mates[0][3] != bed_mates[1][3]:
            qc["n_read_pairs_drop_disagree"] += 1
            continue
        flanking_dicts = [(m[4], m[5]) for m in bed_mates + flank_mates]
        up_merged, dn_merged = _merge_pair_flanking(flanking_dicts)
        qc["n_alignments_flanking_only_kept"] += len(flank_mates)
        accepted.append((bed_mates[0][1], bed_mates[0][2], bed_mates[0][3], up_merged, dn_merged))

    # Determine per-locus flanking widths: max distance seen across all
    # accepted reads, on each side.
    flanking_up_width = 0
    flanking_dn_width = 0
    for _mi, _strand, _ia, up, dn in accepted:
        if up:
            flanking_up_width = max(flanking_up_width, max(up.keys()) + 1)
        if dn:
            flanking_dn_width = max(flanking_dn_width, max(dn.keys()) + 1)

    # Build per-read flanking strings, dedup, assemble Counter.
    flanking_seqs: list[str] = []
    flanking_seq_to_id: dict[str, int] = {}
    read_counts: Counter = Counter()
    for mi_, strand_, ia_, up, dn in accepted:
        up_chars = [up.get(flanking_up_width - 1 - i, ".") for i in range(flanking_up_width)]
        dn_chars = [dn.get(i, ".") for i in range(flanking_dn_width)]
        fl = "".join(up_chars) + "|" + "".join(dn_chars)
        fid = flanking_seq_to_id.get(fl)
        if fid is None:
            fid = len(flanking_seqs)
            flanking_seqs.append(fl)
            flanking_seq_to_id[fl] = fid
        read_counts[(mi_, strand_, ia_, fid)] += 1

    return read_counts, flanking_seqs, flanking_up_width, flanking_dn_width, qc


# ---------------------------------------------------------------------------
# Panel orchestration
# ---------------------------------------------------------------------------


def _prolyg_version() -> str:
    try:
        return _pkg_version("prolyG")
    except PackageNotFoundError:
        return "unknown"


def extract_panel(
    bam_path: str | Path,
    bed_path: str | Path,
    *,
    min_mapq: int = 40,
    drop_chimeric: bool = True,
    max_tlen: int = 1000,
    anchor_hamming_max: int = 1,
    pair_disagreement_policy: str = "drop",
    n_alleles_margin: int = 1,
    min_base_q: int = DEFAULT_MIN_FLANKING_BASE_Q,
    verbose: bool = False,
) -> ExtractedPanel:
    """Walk a BAM against a BED and produce an in-memory `ExtractedPanel`.

    Two passes per locus:

    1. Reconstruct the reference window from MD tags, determine
       polyG-canonical orientation, and try to build upstream +
       downstream anchors.
    2. If anchorable, fetch alignments overlapping the BED interval,
       apply parser-side QC filters, pair-collapse R1+R2 by
       `query_name`, and emit per-(MI, strand, seq) read counts.

    `n_alleles` is set to `max_observed_run_length + n_alleles_margin`,
    where the max is taken across all parsed reads in all loci. Reads
    are not pre-filtered against `n_alleles` at extraction; the panel
    is faithful and the inference adapter (`to_loci`) is what gates
    on the model's state space.

    Parameters
    ----------
    bam_path : path
        Sorted, indexed BAM. Must carry MD tags and MI tags.
    bed_path : path
        4-column BED: `chrom\\tstart\\tend\\tname`.
    min_mapq : int, default 40
        Drop alignments with `mapping_quality < min_mapq`.
    drop_chimeric : bool, default True
        Drop alignments with an SA tag.
    max_tlen : int, default 1000
        Drop alignments with `|TLEN| > max_tlen`.
    anchor_hamming_max : int, default 1
        Cap on per-anchor Hamming distance for accepted placement.
    pair_disagreement_policy : str, default "drop"
        Currently only ``"drop"`` is supported. R1+R2 pairs with
        different inter-anchor `seq` are dropped and counted in
        `n_read_pairs_drop_disagree`. Future policies may add
        ``"r1"`` / ``"keep_both"`` on demand.
    n_alleles_margin : int, default 1
        Headroom over `max_observed_run_length` when setting panel-
        wide `n_alleles`. `1` matches the current SD14 convention.
    verbose : bool, default False
        When True, show a tqdm progress bar over the BED iteration
        (one tick per locus). Silent by default to keep tests quiet
        and notebook output uncluttered.

    Returns
    -------
    ExtractedPanel
    """
    if pair_disagreement_policy != "drop":
        raise NotImplementedError(
            f"pair_disagreement_policy={pair_disagreement_policy!r} not supported; "
            "only 'drop' is implemented."
        )

    bam_path = Path(bam_path)
    bed_path = Path(bed_path)

    bed_rows = _load_bed(bed_path)

    bam = pysam.AlignmentFile(str(bam_path), "rb")

    # Per-locus read counts + anchor info + QC, accumulated.
    per_locus_counts: dict[str, Counter] = {}
    per_locus_anchor: dict[str, AnchorResult] = {}
    per_locus_qc: dict[str, dict] = {}
    per_locus_bed: dict[str, tuple[str, int, int]] = {}
    per_locus_flanking_seqs: dict[str, list[str]] = {}
    per_locus_flanking_up_width: dict[str, int] = {}
    per_locus_flanking_dn_width: dict[str, int] = {}

    max_observed_run_length = 0

    bed_iter = (
        tqdm(bed_rows, desc="extract_panel", unit="locus") if verbose else bed_rows
    )

    try:
        for name, chrom, bed_start, bed_end in bed_iter:
            per_locus_bed[name] = (chrom, bed_start, bed_end)

            anchor = _build_anchors_for_locus(
                bam, chrom, bed_start, bed_end,
                pad=REF_WINDOW_PAD, k=FLANKING_SEQ_LEN, min_mapq=min_mapq,
            )
            per_locus_anchor[name] = anchor

            if anchor.status != "anchorable":
                per_locus_counts[name] = Counter()
                per_locus_qc[name] = {
                    "n_alignments_overlap_locus": 0,
                    "n_alignments_drop_mapq": 0,
                    "n_alignments_drop_sa": 0,
                    "n_alignments_drop_tlen": 0,
                    "n_alignments_drop_anchor": 0,
                    "n_read_pairs": 0,
                    "n_read_pairs_drop_disagree": 0,
                }
                per_locus_flanking_seqs[name] = []
                per_locus_flanking_up_width[name] = 0
                per_locus_flanking_dn_width[name] = 0
                continue

            read_counts, flanking_seqs, up_width, dn_width, qc = _extract_locus(
                bam, chrom, bed_start, bed_end,
                ref_orientation=anchor.ref_orientation,
                anchor_up=anchor.upstream_anchor,
                anchor_dn=anchor.downstream_anchor,
                anchor_k=FLANKING_SEQ_LEN,
                g_walk_up=anchor.g_walk_up,
                g_walk_dn=anchor.g_walk_dn,
                min_mapq=min_mapq,
                drop_chimeric=drop_chimeric,
                max_tlen=max_tlen,
                anchor_hamming_max=anchor_hamming_max,
                min_base_q=min_base_q,
            )
            per_locus_counts[name] = read_counts
            per_locus_qc[name] = qc
            per_locus_flanking_seqs[name] = flanking_seqs
            per_locus_flanking_up_width[name] = up_width
            per_locus_flanking_dn_width[name] = dn_width

            # Update panel-wide max observed run length using the
            # uniform parse on each accepted seq.
            for (_mi, _strand, seq, _fid), _c in read_counts.items():
                runs = parse_run_lengths(seq)
                if runs:
                    m = max(runs)
                    if m > max_observed_run_length:
                        max_observed_run_length = m
    finally:
        bam.close()

    n_alleles = max_observed_run_length + n_alleles_margin

    # Build ExtractedLocus per BED row.
    loci: dict[str, ExtractedLocus] = {}
    for name, chrom, bed_start, bed_end in bed_rows:
        anchor = per_locus_anchor[name]
        counts = per_locus_counts[name]
        qc = per_locus_qc[name]

        if anchor.ref_inter_anchor_seq:
            ref_run_lengths = parse_run_lengths(anchor.ref_inter_anchor_seq)
        else:
            ref_run_lengths = ()

        if counts:
            mi_arr = np.asarray([k[0] for k in counts], dtype=object)
            strand_arr = np.asarray([k[1] for k in counts], dtype="U1")
            seq_arr = np.asarray([k[2] for k in counts], dtype=object)
            flanking_id_arr = np.asarray([k[3] for k in counts], dtype=np.int32)
            count_arr = np.asarray(list(counts.values()), dtype=np.int64)
        else:
            mi_arr = np.empty(0, dtype=object)
            strand_arr = np.empty(0, dtype="U1")
            seq_arr = np.empty(0, dtype=object)
            flanking_id_arr = np.empty(0, dtype=np.int32)
            count_arr = np.empty(0, dtype=np.int64)

        flanking_seqs = per_locus_flanking_seqs[name]
        flanking_seq_arr = np.asarray(flanking_seqs, dtype=object)
        up_width = per_locus_flanking_up_width[name]
        dn_width = per_locus_flanking_dn_width[name]

        if anchor.status == "anchorable":
            if anchor.ref_orientation == "+":
                flanking_up_ref_pos_start = (
                    bed_start - anchor.g_walk_up - FLANKING_SEQ_LEN - up_width
                )
                flanking_dn_ref_pos_start = (
                    bed_end + anchor.g_walk_dn + FLANKING_SEQ_LEN
                )
            else:
                flanking_up_ref_pos_start = (
                    bed_end + anchor.g_walk_up + FLANKING_SEQ_LEN + up_width - 1
                )
                flanking_dn_ref_pos_start = (
                    bed_start - anchor.g_walk_dn - FLANKING_SEQ_LEN - 1
                )
        else:
            flanking_up_ref_pos_start = 0
            flanking_dn_ref_pos_start = 0

        # ``n_runs`` is set from the data: max parsed tuple length across all
        # accepted reads at this locus. The reference-derived
        # ``reference_run_lengths`` is kept as diagnostic context but is no
        # longer authoritative for $S_i$. The inference adapter
        # (``ExtractedPanel.to_loci``) recomputes the locus-level max run
        # count from the observed patterns; extraction stays faithful (no
        # pattern filter here).
        per_locus_max_n_runs = 0
        per_locus_max = 0
        for s in seq_arr:
            runs = parse_run_lengths(str(s))
            if runs:
                if len(runs) > per_locus_max_n_runs:
                    per_locus_max_n_runs = len(runs)
                m = max(runs)
                if m > per_locus_max:
                    per_locus_max = m
        n_runs = per_locus_max_n_runs if per_locus_max_n_runs > 0 else 0

        loci[name] = ExtractedLocus(
            mi=mi_arr,
            strand=strand_arr,
            seq=seq_arr,
            count=count_arr,
            flanking_id=flanking_id_arr,
            flanking_seq=flanking_seq_arr,
            n_runs=n_runs,
            ref_inter_anchor_seq=anchor.ref_inter_anchor_seq,
            reference_run_lengths=ref_run_lengths,
            chrom=chrom,
            bed_start=bed_start,
            bed_end=bed_end,
            bed_name=name,
            ref_orientation=anchor.ref_orientation,
            upstream_anchor=anchor.upstream_anchor,
            downstream_anchor=anchor.downstream_anchor,
            anchorability_status=anchor.status,
            max_observed_run_length=per_locus_max,
            n_alignments_overlap_locus=qc["n_alignments_overlap_locus"],
            n_alignments_drop_mapq=qc["n_alignments_drop_mapq"],
            n_alignments_drop_sa=qc["n_alignments_drop_sa"],
            n_alignments_drop_tlen=qc["n_alignments_drop_tlen"],
            n_alignments_drop_anchor=qc["n_alignments_drop_anchor"],
            n_read_pairs=qc["n_read_pairs"],
            n_read_pairs_drop_disagree=qc["n_read_pairs_drop_disagree"],
            n_reads=int(count_arr.sum()),
            g_walk_up=anchor.g_walk_up,
            g_walk_dn=anchor.g_walk_dn,
            flanking_up_width=up_width,
            flanking_dn_width=dn_width,
            flanking_up_ref_pos_start=flanking_up_ref_pos_start,
            flanking_dn_ref_pos_start=flanking_dn_ref_pos_start,
        )

    provenance = ExtractionProvenance(
        bam_path=str(bam_path),
        bed_path=str(bed_path),
        extraction_version=_prolyg_version(),
        extraction_date=datetime.date.today().isoformat(),
        min_mapq=min_mapq,
        drop_chimeric=drop_chimeric,
        max_tlen=max_tlen,
        anchor_hamming_max=anchor_hamming_max,
        pair_disagreement_policy=pair_disagreement_policy,
        n_alleles_margin=n_alleles_margin,
        min_base_q=min_base_q,
        extraction_schema_version=EXTRACTION_SCHEMA_VERSION,
    )

    return ExtractedPanel(loci=loci, n_alleles=n_alleles, provenance=provenance)
