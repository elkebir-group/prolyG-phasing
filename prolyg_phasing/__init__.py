"""prolyg_phasing — read/family haplotype labeling at polyG loci.

Given an aligned BAM and a BED panel, labels each read family with its
interrupter pattern and, where a flanking SNV resolves it, the haplotype
(``hap_major``/``hap_minor``) it carries. Every label this package
produces is a direct readout of a read's own sequence.

Two stages:

- :func:`prolyg_phasing.io.bam.extract_panel` — BAM + BED →
  :class:`~prolyg_phasing.io.panel.ExtractedPanel`.
- :class:`prolyg_phasing.phasing.PhasingPanel` — the flanking-SNV phasing
  pipeline over an :class:`~prolyg_phasing.io.panel.ExtractedPanel`.

See the README for the CLI (``prolyg-phasing extract`` / ``prolyg-phasing
phase``) and the bundled Snakemake workflow.
"""
