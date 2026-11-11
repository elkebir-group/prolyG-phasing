"""Synthetic ``ExtractedPanel``/``ExtractedLocus`` builders for this repo's
own test suite.

A copy of prolyG's ``tests/_panel_fixtures.py`` (which prolyG keeps too —
its own ``test_panel_to_loci.py`` and ``test_cn_family_counts.py`` still need
the identical builders, now importing ``ExtractedLocus``/
``ExtractionProvenance`` from this package instead of a local module). Kept
in sync by hand since the two repos do not share a test-helper package.
"""
from __future__ import annotations

import numpy as np

from prolyg_phasing.io._flank_placement import FlankRecord
from prolyg_phasing.io.panel import (
    _FLANK_DROP_COLUMNS,
    EXTRACTION_SCHEMA_VERSION,
    ExtractedLocus,
    ExtractionProvenance,
    parse_run_lengths,
)

# Flanks of a synthetic "+" locus whose repeat lies on [100, 107).
UPSTREAM_FLANK = FlankRecord("ACGTACGTAC", tuple(range(90, 100)), 9)
DOWNSTREAM_FLANK = FlankRecord("TGCATGCATG", tuple(range(107, 117)), 0)

# Every flank-placement drop counter at zero.
ZERO_FLANK_DROPS = dict.fromkeys(_FLANK_DROP_COLUMNS, 0)

# The flank-table fields of a synthetic provenance.
FLANK_TABLE_PROVENANCE = {
    "flank_table_path": "/synthetic/normal_flank_sides.tsv",
    "flank_table_sha256": "0" * 64,
    "flank_table_sample": "NORMAL",
}


def _make_provenance() -> ExtractionProvenance:
    return ExtractionProvenance(
        bam_path="/synthetic/test.bam",
        bed_path="/synthetic/test.bed",
        extraction_version="0.0.0-test",
        extraction_date="2026-05-11",
        min_mapq=40,
        drop_chimeric=True,
        max_tlen=1000,
        **FLANK_TABLE_PROVENANCE,
        pair_disagreement_policy="merge_by_quality",
        extraction_schema_version=EXTRACTION_SCHEMA_VERSION,
        n_alleles_margin=1,
    )


def _make_locus_single_run(
    mi_rows: list[tuple[str, str, str, int]],
    *,
    n_runs: int = 1,
    reference_tract: str = "GGGGGGG",
    reference_run_lengths: tuple[int, ...] = (7,),
    bed_name: str = "Locus1",
    chrom: str = "chr1",
    bed_start: int = 100,
    bed_end: int = 107,
    ref_orientation: str = "+",
    upstream_flank: FlankRecord = UPSTREAM_FLANK,
    downstream_flank: FlankRecord = DOWNSTREAM_FLANK,
    reference_window_status: str = "recovered",
) -> ExtractedLocus:
    """Build an ExtractedLocus from a list of (mi, strand, seq, count) tuples."""
    if mi_rows:
        mi = np.asarray([r[0] for r in mi_rows], dtype=object)
        strand = np.asarray([r[1] for r in mi_rows], dtype="U1")
        seq = np.asarray([r[2] for r in mi_rows], dtype=object)
        count = np.asarray([r[3] for r in mi_rows], dtype=np.int64)
    else:
        mi = np.empty(0, dtype=object)
        strand = np.empty(0, dtype="U1")
        seq = np.empty(0, dtype=object)
        count = np.empty(0, dtype=np.int64)

    max_obs = 0
    for s in seq:
        runs = parse_run_lengths(str(s))
        if runs:
            max_obs = max(max_obs, max(runs))

    return ExtractedLocus(
        mi=mi, strand=strand, seq=seq, count=count,
        n_runs=n_runs,
        reference_tract=reference_tract,
        reference_run_lengths=reference_run_lengths,
        chrom=chrom, bed_start=bed_start, bed_end=bed_end, bed_name=bed_name,
        ref_orientation=ref_orientation,
        upstream_flank=upstream_flank, downstream_flank=downstream_flank,
        reference_window_status=reference_window_status,
        max_observed_run_length=max_obs,
        n_alignments_overlap_locus=int(count.sum()) * 2,  # rough placeholder
        n_alignments_drop_mapq=0,
        n_alignments_drop_sa=0,
        n_alignments_drop_tlen=0,
        **ZERO_FLANK_DROPS,
        n_read_pairs=int(count.sum()),
        n_read_pairs_drop_disagree=0, n_read_pairs_merged=0,
        n_reads=int(count.sum()),
    )
