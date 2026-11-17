# prolyG-phasing

Labels each read family at a polyG locus with its interrupter pattern and,
where a flanking single-nucleotide variant (SNV) resolves it, the haplotype
(`hap_major` / `hap_minor`) it carries. Given an aligned BAM and a BED panel,
it produces per-locus, per-family, and per-SNV tables. Every label is a
direct readout of a read's own sequence.

Five commands:

1. **Build flanks** (`build-flanks`) — from a donor's matched normal, choose
   two 10-base flanks for each locus. Every sample of the donor is read with
   these flanks.
2. **Extract** (`extract`) — walk a BAM against a BED panel and write a
   deduplicated, per-locus panel (`panel.db`). In each read, we place the
   two flanks at the highest-value position, and the tract is the read
   between them. We merge the two mates of a pair base by base by quality
   before we take the tract. `--workers N` extracts N loci at a time.
3. **Phase** (`phase`) — find flanking positions that are informative
   (heterozygous) in the panel, phase them into two haplotype profiles per
   locus, and label every read family that carries phase evidence.
4. **Ref-phase** (`ref-phase`) — vote a second panel's reads against the
   haplotype profiles of an already-phased reference panel, for example a
   tumor sample against its matched normal.
5. **View** (`view-loci`) — serve a local web page that shows each locus of
   one or more panels: the flanks, the tracts, the pile-up and the counters.

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

Two small synthetic BAMs + a shared BED ship with the repo
(`workflow/example/`, built by `workflow/example/make_example_data.py`) so
this runs immediately: two loci, realistic PCR-duplicate depth throughout
(every row's `count` is 2-5, never a single read). `example.bam` (for
`extract`/`phase`) has Locus1 with no interrupter allele and no phase
signal — everything phases `unk`. Locus2 has two admitted interrupter
patterns and 30 families split across two linked flanking SNVs, so
`phase`'s output shows real `hap_major`/`hap_minor` calls, a genuine
(< 1.0) interrupter/phase concordance number, and one family whose two
strands carry different vote evidence. `example_target.bam` (for
`ref-phase`, below) is a second, independently composed sample at the same
two loci.

`extract` reads every sample of a donor with the flanks that `build-flanks`
builds from the donor's matched normal; here `example.bam` plays the normal.
The sample name in the flank table is the BAM's read-group `SM`.

```bash
prolyg-phasing build-flanks \
    --bam workflow/example/example.bam --bed workflow/example/panel.bed \
    --out-prefix example

prolyg-phasing extract \
    --bam workflow/example/example.bam --bed workflow/example/panel.bed \
    --flanks example_flank_sides.tsv --out-db panel.db

prolyg-phasing phase \
    --panel-db panel.db --out-dir phasing/
```

`phase` always writes `phasing.pkl` plus three tables into `--out-dir`:
`haplotype_calls.tsv`, `locus_summary.tsv`, `snv_calls.tsv` (below).

`ref-phase` votes a *different* panel's reads against an already-phased
reference panel, instead of phasing its own — the reference's SNVs and
haplotype profiles come entirely from a prior `phase` run. A second
synthetic BAM, `example_target.bam`, ships for this: a genuinely different
sample at the same two loci (same genomic alleles, a different — skewed
12/6 rather than balanced 15/15 — family split at Locus2, no interrupter
alternation or strand-mismatch flourish):

```bash
prolyg-phasing extract \
    --bam workflow/example/example_target.bam --bed workflow/example/panel.bed \
    --flanks example_flank_sides.tsv --out-db target_panel.db

prolyg-phasing ref-phase \
    --target-panel-db target_panel.db --reference-snv-calls phasing/snv_calls.tsv \
    --target-sample-id TARGET --reference-sample-id REFERENCE \
    --out-dir ref_phasing/
```

`ref-phase`'s output shows the target's 18 Locus2 families all getting a
confident `hap_major`/`hap_minor` label (no `unk`) from the reference's
profile, split 84/38 reads — close to the family-count ratio (12/6 = 2:1),
the gap coming from per-family depth variation — with `hap_major` landing
on the same allele (`CT`) `phasing/locus_summary.tsv` already reported for
the reference, despite the target never discovering that allele itself.

`--reference-phasing-pkl phasing/phasing.pkl` also works in place of
`--reference-snv-calls`; the two are equivalent, but the table is smaller —
`phasing.pkl` also carries the O(reads) per-family phase-label assignments
this command never needs. `ref-phase` always writes `ref_phasing.pkl`,
`ref_phasing_summary.tsv`, and `haplotype_calls.tsv`, `snv_calls.tsv` and
`locus_summary.tsv` in the formats `phase` writes, into `--out-dir`
(below). So `view-loci --panel <target>.panel.db <ref-phase out-dir>` shows
the target's labels.

Run `prolyg-phasing <command> --help` for any command's full flag list.

## Quickstart — Snakemake

The same three steps, over many samples at once. Runs out of the box against
the committed example (`workflow/config/samples.tsv`/`config.yaml` already
point at it):

```bash
snakemake -s workflow/Snakefile --configfile workflow/config/config.yaml --cores N
```

This drives the identical `prolyg-phasing build-flanks`/`prolyg-phasing
extract`/`prolyg-phasing phase` CLI per sample — Snakemake is a convenience
for running many samples, not a separate code path. To point it at your own
data, edit `workflow/config/samples.tsv` (`sample_id`, `bam` — the `bam`
path is relative to the repo root — and `normal`, the `sample_id` of the
donor's matched normal; a normal names itself) and `workflow/config/config.yaml`'s `panel.bed`
key in place, or override without editing anything:

```bash
snakemake -s workflow/Snakefile --cores N \
    --config samples=/path/to/your/samples.tsv panel='{"bed": "/path/to/your.bed"}'
```

## Locus viewer

`view-loci` starts an HTTP server on 127.0.0.1 that reads the panels on
request and writes no files. Give each panel of a donor with `--panel`, and
after each panel the output directory of its `phase` or `ref-phase` run:

```bash
prolyg-phasing view-loci \
    --panel panel.db phasing/ \
    --panel target_panel.db ref_phasing/ \
    --port 8765
```

Each panel is one sample tab. With a phasing directory, the pile-up splits
its rows by phase label and shows one column per SNV. The viewer stops at
start if a phasing directory belongs to another panel. On a remote machine,
open an ssh tunnel (`ssh -N -L 8765:localhost:8765 <host>`) and then
`http://localhost:8765/`.

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

- **`ref_phasing_summary.tsv`** (`ref-phase` only) — one row per target
  locus. Unlike `locus_summary.tsv`, the target never phases its own
  haplotypes — the profile is borrowed from the reference — so there is no
  `hap_major_profile`/interrupter-concordance pair here, only the vote
  outcome and the SNV coverage that produced it. Columns:
  - `locus_id`.
  - `n_usable_snvs` — reference SNVs the target locus's own flanking window
    covers, after re-locating each by genomic position; `0` means this
    locus phased all-`unk`.
  - `n_rows` — this locus's row count; `n_families`/`n_families_labeled` —
    distinct molecules, and the subset with a non-`unk` label.
  - `reads_total`/`reads_major`/`reads_minor`/`reads_unk` — raw read
    counts, split by vote outcome.

- **`haplotype_calls.tsv`, `snv_calls.tsv`, `locus_summary.tsv`**
  (`ref-phase`) — the same columns as from `phase`, for the target panel:
  - `haplotype_calls.tsv` — one row per target panel row, in panel order,
    with the ref-phase label.
  - `snv_calls.tsv` — one row per reference SNV the target covers, all
    `phased`. `segment` and `string_index` are the target's own position of
    the SNV (re-located by `ref_pos`). `allele_major`/`allele_minor`,
    `hap_major_base`/`hap_minor_base` and `is_anchor` are the reference's.
    **`vaf_minor` and `coverage` are the target's own family counts at that
    position**, counted as `find_informative_snvs` counts: `coverage` is
    the target's ACGT family depth, and `vaf_minor` the target's families
    with the reference's `allele_minor` over that depth (blank with no
    depth). `pair_discordance` and `pair_double_coverage` are blank. A
    locus whose target does not cover the reference anchor has no
    `is_anchor` row.
  - `locus_summary.tsv` — the profiles over those SNVs, and the target's
    reads per label. `n_snvs_dropped_low_linkage` is `0`;
    `max_discordance` and `n_anchor_families` are blank (not measured on
    the target).

## Scope

This package produces read-level and locus-level labels. A companion
library, [prolyG](https://github.com/elkebir-group/prolyG), consumes these
panels for PCR-chemistry inference and copy-number calling.

`ref-phase` additionally lets a caller vote one panel's reads against a
*different*, already-phased panel's haplotype profiles — e.g. prolyG's
bulk pipeline votes a tumor sample's reads against its matched normal. It
is not wired into the Snakemake workflow: its two-panel (target +
reference) input shape does not fit the workflow's one-row-per-sample
table.
