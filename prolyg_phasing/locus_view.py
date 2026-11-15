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
from collections import Counter, defaultdict
from collections.abc import Sequence
from importlib import resources
from pathlib import Path

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
        One entry per distinct (tract, upstream flanking string, downstream
        flanking string): ``tract``, ``up_flanking``, ``dn_flanking``,
        ``reads`` (summed read count) and ``molecules`` (distinct MI).
        Sorted by reads, most first, then by tract.
    """
    counters = {f.name: int(getattr(locus, f.name))
                for f in dataclasses.fields(locus)
                if f.name.startswith(_COUNTER_PREFIXES)}
    counters["n_reads"] = int(locus.n_reads)

    reads: Counter = Counter()
    families: dict[tuple[str, str, str], set] = defaultdict(set)
    flanking = [str(s).split("|") for s in locus.flanking_seq]
    for mi, seq, fid, count in zip(locus.mi, locus.seq, locus.flanking_id, locus.count,
                                   strict=True):
        up, dn = flanking[int(fid)]
        key = (str(seq), up, dn)
        reads[key] += int(count)
        families[key].add(str(mi))
    molecules = [{"tract": t, "up_flanking": u, "dn_flanking": d, "reads": n,
                  "molecules": len(families[(t, u, d)])}
                 for (t, u, d), n in reads.items()]
    molecules.sort(key=lambda m: (-m["reads"], m["tract"], m["up_flanking"], m["dn_flanking"]))

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
    }


def page_data(panel_paths: Sequence[str | Path], loci: Sequence[str]) -> dict:
    """The page's data: per locus, one :func:`locus_view_data` dict per panel.

    Each panel reads only the requested loci. Raises KeyError naming the
    locus and the panel when a panel does not hold a locus.
    """
    samples, by_locus = [], {locus: [] for locus in loci}
    for path in panel_paths:
        panel = ExtractedPanel.load_db(path)
        sample = panel_sample(panel)
        samples.append({"sample": sample, "panel": str(path),
                        "flank_table_sample": panel.provenance.flank_table_sample})
        for locus in loci:
            try:
                extracted = panel.loci[locus]
            except KeyError:
                raise KeyError(f"{path}: the panel holds no locus {locus}") from None
            by_locus[locus].append(locus_view_data(locus, extracted, sample))
    return {"samples": samples, "loci": [{"locus": locus, "blocks": by_locus[locus]}
                                         for locus in loci],
            "rows_shown": ROWS_SHOWN}


def write_locus_page(panel_paths: Sequence[str | Path], loci: Sequence[str],
                     out: str | Path) -> Path:
    """Write the self-contained HTML page of `loci` over the panels; return its path."""
    data = page_data(panel_paths, loci)
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    title = html.escape(", ".join(loci[:4]) + (" …" if len(loci) > 4 else ""))
    page = resources.files("prolyg_phasing").joinpath("locus_view.html").read_text("utf-8")
    out = Path(out)
    out.write_text(page.replace("__TITLE__", title).replace("__DATA__", payload),
                   encoding="utf-8")
    return out
