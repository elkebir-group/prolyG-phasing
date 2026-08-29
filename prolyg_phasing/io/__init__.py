"""IO subpackage — BAM panel extraction and serialization.

Public API:

- :func:`extract_panel` — walk a BAM against a BED, return an
  :class:`ExtractedPanel`.
- :class:`ExtractedPanel`, :class:`ExtractedLocus`,
  :class:`ExtractionProvenance` — the in-memory artifact and its
  ``panel.db`` serialization round trip.
- :func:`parse_run_lengths` — the uniform tuple parse on an
  inter-anchor sequence (re-exported for testing and downstream
  inspection).

The `bam` module is pysam-coupled; the `panel` module is pysam-free,
so artifacts written by `extract_panel` can be loaded on machines
without pysam installed.
"""

from prolyg_phasing.io.bam import extract_panel
from prolyg_phasing.io.panel import (
    ExtractedLocus,
    ExtractedPanel,
    ExtractionProvenance,
    ToLociResult,
    interrupter_pattern,
    observed_pattern_frequencies,
    parse_run_lengths,
    select_patterns_above_freq,
)

__all__ = [
    "extract_panel",
    "ExtractedPanel",
    "ExtractedLocus",
    "ExtractionProvenance",
    "ToLociResult",
    "parse_run_lengths",
    "interrupter_pattern",
    "observed_pattern_frequencies",
    "select_patterns_above_freq",
]
