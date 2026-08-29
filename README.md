# prolyG-phasing

Labels each read family at a polyG locus with its interrupter pattern and,
where a flanking single-nucleotide variant (SNV) resolves it, the haplotype
(`hap_major` / `hap_minor`) it carries. Given an aligned BAM and a BED panel,
it produces per-locus, per-family, and per-SNV tables. Every label is a
direct readout of a read's own sequence.

Two stages:

1. **Extract** — walk a BAM against a BED panel, collapse read pairs into a
   deduplicated, per-locus panel (`panel.db`).
2. **Phase** — find flanking positions that are informative (heterozygous)
   in the panel, phase them into two haplotype profiles per locus, and label
   every read family that carries phase evidence.

## Install

```bash
uv sync                       # creates .venv from uv.lock
uv run pre-commit install     # one-time, post-clone
```

`pip install -e ".[dev,bam]"` also works if you manage the environment
yourself. The `bam` extra pulls in `pysam`, needed only for `extract` — a
consumer who already has a `panel.db` and only runs `phase` does not need
it.

## Quickstart — direct execution

A small synthetic BAM + BED ship with the repo (`workflow/example/`, built
by `workflow/example/make_example_data.py`) so this runs immediately: two
loci, realistic PCR-duplicate depth throughout (every row's `count` is
2-5, never a single read). Locus1 has no interrupter allele and no phase
signal — everything phases `unk`. Locus2 has two admitted interrupter
patterns and 30 families split across two linked flanking SNVs, so
`phase`'s output shows real `hap_major`/`hap_minor` calls, a genuine
(< 1.0) interrupter/phase concordance number, and one family whose two
strands carry different vote evidence.

```bash
prolyg-phasing extract \
    --bam workflow/example/example.bam --bed workflow/example/panel.bed \
    --out-db panel.db

prolyg-phasing phase \
    --panel-db panel.db --out-dir phasing/
```

`phase` always writes `phasing.pkl` plus three tables into `--out-dir`:
`haplotype_calls.tsv`, `locus_summary.tsv`, `snv_calls.tsv` (below). Run
`prolyg-phasing extract --help` / `prolyg-phasing phase --help` for the full
flag list.

## Quickstart — Snakemake

The same two steps, over many samples at once. Runs out of the box against
the committed example (`workflow/config/samples.tsv`/`config.yaml` already
point at it):

```bash
snakemake -s workflow/Snakefile --configfile workflow/config/config.yaml --cores N
```

This drives the identical `prolyg-phasing extract`/`prolyg-phasing phase`
CLI per sample — Snakemake is a convenience for running many samples, not a
separate code path. To point it at your own data, edit
`workflow/config/samples.tsv` (`sample_id`, `bam` — the `bam` path is
relative to the repo root) and `workflow/config/config.yaml`'s `panel.bed`
key in place, or override without editing anything:

```bash
snakemake -s workflow/Snakefile --cores N \
    --config samples=/path/to/your/samples.tsv panel='{"bed": "/path/to/your.bed"}'
```

## Output tables

- **`haplotype_calls.tsv`** — one row per `(locus, molecule, strand)`: a
  deduplicated molecule×strand record, not a raw sequencing read. Not
  collapsed to one row per molecule — a molecule can carry both strands,
  and they can disagree. Columns:
  - `locus_id`, `mi` (molecule ID), `strand` (`A` / `B`) — identity.
  - `count` — raw read pairs that collapsed into this row.
  - `interrupter_pattern` — this molecule's majority interrupter pattern,
    as text: its non-`G` tokens joined by `_` (e.g. `A`, or `AT_C` for two
    interrupters), or `-` for a plain, uninterrupted run.
  - `phase_label` — `hap_major` / `hap_minor` / `unk`.
  - `n_major` / `n_minor` / `n_other` — the vote evidence behind
    `phase_label`: how many of this locus's phased SNVs this row's base
    matched the major-haplotype allele, the minor-haplotype allele, or
    neither (a real third base — not the same thing as `unk`; see below).

- **`locus_summary.tsv`** — one row per locus. Columns:
  - `n_rfs` — distinct molecules/families.
  - `n_rows` — this locus's row count in `haplotype_calls.tsv` (`>= n_rfs`,
    since a molecule can appear on both strands).
  - `admitted_patterns` — a `;`-separated `pattern:frequency` list,
    frequency-descending, of every interrupter pattern whose share of
    `n_rfs` clears `--min-pattern-freq` (default 0.1) — e.g. `A:0.8;T:0.2`
    means 80% of this locus's families carry interrupter `A`, 20% carry
    `T`. A pattern below the threshold is left out, so the shares shown do
    not have to sum to 1.0.
  - `n_admitted_patterns` — length of that list.
  - `hap_major_profile` / `hap_minor_profile` — one base per phased SNV
    (same order as `snv_calls.tsv`'s rows for this locus), concatenated —
    e.g. `CT` at a 2-SNV locus is the major haplotype's base at the first
    phased SNV, then the second.
  - `n_snvs_phased` / `n_snvs_dropped_low_linkage` — SNVs kept in the two
    profiles above vs. dropped for insufficient double-coverage with the
    locus's anchor SNV (`--min-linkage-families`).
  - `max_discordance` — worst per-SNV phasing disagreement: for each
    non-anchor phased SNV, the fraction of molecules covering both it and
    the anchor whose consensus bases disagree with the chosen phase; this
    column is the max over those SNVs (`0.0` if the locus has 0 or 1
    phased SNV, since there is no non-anchor SNV to disagree).
  - `n_anchor_families` — the anchor SNV's own family depth (ACGT
    molecules covering it).
  - `reads_total` / `reads_major` / `reads_minor` / `reads_unk` — raw read
    counts (summed `haplotype_calls.tsv` `count`), split by `phase_label`.
  - `phase_interrupter_concordance_major` / `_minor` — among
    `haplotype_calls.tsv` rows labeled `hap_major` (`hap_minor`), the
    modal `interrupter_pattern`'s share. Checks whether the two
    independent evidence sources (flanking SNV, interrupter pattern)
    agree on the same two-way split; `NaN` if the locus has no rows with
    that label.

- **`snv_calls.tsv`** — one row per `(locus, SNV)`, both phased and
  dropped-for-low-linkage (`status`). Columns:
  - `locus_id`, `ref_pos` (genomic position), `segment` (`up` / `dn`,
    which side of the polyG tract), `string_index` (offset within that
    side's flanking string) — identity/position.
  - `allele_major` / `allele_minor` — the top-2 bases by family count at
    this position; `vaf_minor` — the minor base's family-fraction;
    `coverage` — ACGT family depth. These three describe the position on
    its own, independent of phasing.
  - `status` — `phased` or `dropped_low_linkage`.
  - `hap_major_base` / `hap_minor_base` — which base each haplotype
    profile carries here (blank for a dropped SNV, since it was never
    assigned a phase). Can differ from `allele_major`/`allele_minor` when
    this SNV was out-of-phase with the locus's anchor SNV and got flipped
    during linkage resolution — `pair_discordance` below is the evidence
    for that flip.
  - `is_anchor` — whether this is the locus's anchor SNV (the phased SNV
    with the highest `coverage`; ties broken by input order). Blank for a
    dropped SNV.
  - `pair_discordance` — fraction of molecules covering both this SNV and
    the anchor whose consensus bases disagree with the chosen phase (`0.0`
    at the anchor itself). Blank for a dropped SNV.
  - `pair_double_coverage` — molecules covering both this SNV and the
    anchor with a called base at both — the number `--min-linkage-families`
    is checked against. Blank for a dropped SNV (a dropped SNV is exactly
    one whose `pair_double_coverage` fell below that flag).

## Scope

This package produces read-level and locus-level labels. A companion
library, [prolyG](https://github.com/elkebir-group/prolyG), consumes these
panels for PCR-chemistry inference and copy-number calling.

`prolyg_phasing.anchor_phasing` additionally lets a caller vote one panel's
reads against a *different*, already-phased panel's haplotype profiles
(`assign_flanking_haplotypes_anchored`) — e.g. prolyG's bulk pipeline votes
a tumor sample's reads against its matched normal. Library-only: no CLI
subcommand or Snakemake rule wraps it, since prolyG is currently its only
caller.
