"""Command-line entry point: ``prolyg-phasing extract`` / ``prolyg-phasing phase``.

Thin argparse wrappers, one per stage. Run directly against one BAM (or one
already-extracted ``panel.db``), or drive the same two steps over many
samples through ``workflow/Snakefile``, which shells out to these same
subcommands.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from prolyg_phasing.io.bam import extract_panel
from prolyg_phasing.io.panel import ExtractedPanel
from prolyg_phasing.io.phasing_tables import (
    build_haplotype_calls,
    build_locus_summary,
    build_snv_calls,
)
from prolyg_phasing.phasing import PhasingPanel

_EXTRACT_DESCRIPTION = """\
Walk a sorted, indexed BAM against a BED panel and write the extracted
panel to a panel.db (SQLite) file.

Two passes per locus: (1) reconstruct the reference window from MD tags,
determine polyG-canonical orientation, and try to build upstream +
downstream anchors; (2) if anchorable, fetch alignments overlapping the
BED interval, apply the QC filters below, pair-collapse R1+R2 by read
name, and emit per-(molecule, strand, sequence) read counts.

The BAM must carry MD tags (reference reconstruction) and MI tags (one
per molecule/UMI family — required for every family-level count and
label this package produces)."""


def _add_extract_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bam", required=True, type=Path,
                    help="Sorted, indexed BAM. Must carry MD and MI tags.")
    p.add_argument("--bed", required=True, type=Path,
                    help="4-column BED panel: chrom, start, end, name — "
                         "one row per polyG locus to extract.")
    p.add_argument("--out-db", required=True, type=Path,
                    help="Path to write the panel.db (SQLite) runtime "
                         "artifact. Parent directory is not created.")
    p.add_argument("--min-mapq", type=int, default=40,
                    help="Drop alignments with mapping quality below this.")
    p.add_argument("--no-drop-chimeric", dest="drop_chimeric",
                    action="store_false", default=True,
                    help="Keep chimeric (SA-tagged) alignments instead of "
                         "dropping them.")
    p.add_argument("--max-tlen", type=int, default=1000,
                    help="Drop alignments with |TLEN| greater than this.")
    p.add_argument("--anchor-hamming-max", type=int, default=1,
                    help="Maximum Hamming distance allowed when placing "
                         "the upstream/downstream anchor sequence in a "
                         "read; a placement past this is rejected.")
    p.add_argument("--pair-disagreement-policy", default="drop",
                    choices=["drop"],
                    help="What to do when a read pair's R1 and R2 report "
                         "different inter-anchor sequence. Only 'drop' "
                         "(discard the pair, count it in "
                         "n_read_pairs_drop_disagree) is implemented.")
    p.add_argument("--n-alleles-margin", type=int, default=1,
                    help="Headroom added over the panel-wide max observed "
                         "run length when sizing the allele-count axis "
                         "n_alleles (n_alleles = max_run_length + margin).")
    p.add_argument("--min-flanking-base-q", type=int, default=20,
                    help="Minimum Phred base quality for a flanking-"
                         "sequence position to be included; lower-quality "
                         "positions are masked out, not the whole read.")
    p.add_argument("--verbose", action="store_true",
                    help="Show a progress bar over the BED (one tick per "
                         "locus). Silent by default.")


def _run_extract(args: argparse.Namespace) -> None:
    panel = extract_panel(
        bam_path=args.bam,
        bed_path=args.bed,
        min_mapq=args.min_mapq,
        drop_chimeric=args.drop_chimeric,
        max_tlen=args.max_tlen,
        anchor_hamming_max=args.anchor_hamming_max,
        pair_disagreement_policy=args.pair_disagreement_policy,
        n_alleles_margin=args.n_alleles_margin,
        min_base_q=args.min_flanking_base_q,
        verbose=args.verbose,
    )
    panel.save_db(args.out_db)
    print(f"wrote {args.out_db} ({len(panel.loci)} loci)")


_PHASE_DESCRIPTION = """\
Find informative flanking SNVs in a panel.db, phase them into two
haplotype profiles per locus, and label every read family that carries
phase evidence. Always writes phasing.pkl plus three tables into
--out-dir: haplotype_calls.tsv (per read family), locus_summary.tsv
(per locus), snv_calls.tsv (per SNV) — see the README for their columns.

Three stages, tuned by the flags below: (1) a flanking position is an
informative SNV candidate if it clears --min-coverage and --vaf-min; (2)
each locus's SNVs are phased against an anchor SNV, dropping any partner
whose double-coverage with the anchor falls below
--min-linkage-families; (3) every read family is labeled hap_major /
hap_minor / unk by a coverage-weighted vote over its phased SNVs, gated
by --min-informative-positions and --vote-margin."""


def _add_phase_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--panel-db", required=True, type=Path,
                    help="panel.db written by `prolyg-phasing extract`.")
    p.add_argument("--out-dir", required=True, type=Path,
                    help="Directory for phasing.pkl + the three output "
                         "tables (haplotype_calls.tsv, locus_summary.tsv, "
                         "snv_calls.tsv). Created if missing.")
    p.add_argument("--min-coverage", type=int, default=25,
                    help="Minimum ACGT read-family depth at a flanking "
                         "position for it to be an informative SNV "
                         "candidate. Family units (one vote per molecule), "
                         "not raw reads.")
    p.add_argument("--vaf-min", type=float, default=0.2,
                    help="Minimum minor-allele family fraction (share of "
                         "ACGT-covering families calling the minor base) "
                         "for a position to be an informative SNV "
                         "candidate.")
    p.add_argument("--min-linkage-families", type=int, default=10,
                    help="Minimum molecules covering both the locus's "
                         "anchor SNV and a partner SNV for that partner's "
                         "in-phase/out-of-phase call to be made; below "
                         "this the partner SNV is dropped as "
                         "low-linkage, and the locus still phases with "
                         "its remaining SNVs.")
    p.add_argument("--min-informative-positions", type=int, default=1,
                    help="Minimum number of phased SNVs a read family "
                         "must have votes at (n_major + n_minor) to get a "
                         "non-'unk' hap_major/hap_minor label.")
    p.add_argument("--vote-margin", type=int, default=0,
                    help="Minimum |n_major - n_minor| vote margin a read "
                         "family needs to get a non-'unk' label; a tie or "
                         "near-tie within this margin labels 'unk'.")
    p.add_argument(
        "--min-pattern-freq", type=float, default=0.1,
        help="Display threshold for locus_summary.tsv's admitted_patterns "
             "column: an interrupter pattern needs at least this frequency "
             "share of a locus's read families to be listed.",
    )


def _run_phase(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)

    panel = ExtractedPanel.load_db(args.panel_db)
    phasing = PhasingPanel.from_panel(
        panel,
        min_coverage=args.min_coverage,
        vaf_min=args.vaf_min,
        min_linkage_families=args.min_linkage_families,
        min_informative_positions=args.min_informative_positions,
        vote_margin=args.vote_margin,
    )
    phasing.save_pickle(args.out_dir / "phasing.pkl")

    build_haplotype_calls(panel, phasing).to_csv(
        args.out_dir / "haplotype_calls.tsv", sep="\t", index=False,
    )
    build_locus_summary(panel, phasing, min_pattern_freq=args.min_pattern_freq).to_csv(
        args.out_dir / "locus_summary.tsv", sep="\t", index=False,
    )
    build_snv_calls(phasing).to_csv(
        args.out_dir / "snv_calls.tsv", sep="\t", index=False,
    )

    n_loci = len(phasing.assignments)
    n_with_snvs = sum(1 for h in phasing.haplotypes.values() if len(h.snvs) > 0)
    n_labeled_rows = sum(
        int((a.label >= 0).sum()) for a in phasing.assignments.values()
    )
    n_total_rows = sum(a.n_rows for a in phasing.assignments.values())
    print(
        f"phased {n_loci} loci ({n_with_snvs} with informative SNVs); "
        f"{n_labeled_rows}/{n_total_rows} rows labeled major/minor; "
        f"wrote {args.out_dir}"
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="prolyg-phasing",
        description="Read/family interrupter-pattern and flanking-SNV "
                     "haplotype labeling at polyG loci. Run "
                     "`prolyg-phasing <command> --help` for a command's "
                     "full flag list.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    extract_p = sub.add_parser(
        "extract",
        help="Extract a polyG panel from one BAM into panel.db.",
        description=_EXTRACT_DESCRIPTION,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_extract_args(extract_p)
    extract_p.set_defaults(func=_run_extract)

    phase_p = sub.add_parser(
        "phase",
        help="Phase flanking SNVs of a panel.db and write phasing.pkl "
             "plus the three output tables.",
        description=_PHASE_DESCRIPTION,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_phase_args(phase_p)
    phase_p.set_defaults(func=_run_phase)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
