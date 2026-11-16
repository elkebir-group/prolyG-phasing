"""The relations between a locus's extraction counters that the viewer's counter help states.

The "counters" group of `prolyg_phasing.locus_view.HELP_TEXTS` states each
of these in the text of the counter it explains. The extraction guarantees
them by construction (see `prolyg_phasing.io.bam`); a test calls
`assert_counter_relations` on every locus it extracts.
"""

from __future__ import annotations

from prolyg_phasing.locus_view import locus_view_data


def neither_flank_matched(c) -> int:
    """Alignments where neither flank matches: they count in both side reasons."""
    return (c["n_alignments_drop_upstream_flank_not_placed"]
            + c["n_alignments_drop_downstream_flank_not_placed"]
            + c["n_alignments_drop_no_placement"] + c["n_alignments_drop_no_bed_base"]
            + c["n_alignments_drop_negative_value"] - c["n_alignments_drop_flank"])


def counter_relation_failures(c) -> list[str]:
    """The relations that the counters `c` (a dict by counter name) break."""
    kept = c["n_alignments_placed"] - c["n_alignments_drop_negative_value"]
    neither = neither_flank_matched(c)
    checks = {
        "overlap = MAPQ + SA + TLEN + no MI + placed + flank drop - negative value":
            c["n_alignments_overlap_locus"] == (
                c["n_alignments_drop_mapq"] + c["n_alignments_drop_sa"]
                + c["n_alignments_drop_tlen"] + c["n_alignments_drop_no_mi"]
                + c["n_alignments_placed"] + c["n_alignments_drop_flank"]
                - c["n_alignments_drop_negative_value"]),
        "negative value <= placed": kept >= 0,
        "0 <= neither flank matched <= each side's not-placed count":
            0 <= neither <= min(c["n_alignments_drop_upstream_flank_not_placed"],
                                c["n_alignments_drop_downstream_flank_not_placed"]),
        "read pairs <= placed - negative value": c["n_read_pairs"] <= kept,
        "merged + disagree <= read pairs":
            c["n_read_pairs_merged"] + c["n_read_pairs_drop_disagree"] <= c["n_read_pairs"],
        "reads = read pairs - disagree":
            c["n_reads"] == c["n_read_pairs"] - c["n_read_pairs_drop_disagree"],
    }
    return [name for name, ok in checks.items() if not ok]


def counters_of(locus) -> dict[str, int]:
    """The counters of `locus` (an ExtractedLocus), as the viewer shows them."""
    return locus_view_data(locus.bed_name, locus, "S")["counters"]


def assert_counter_relations(locus) -> None:
    """Every stated counter relation holds at `locus` (an ExtractedLocus)."""
    assert counter_relation_failures(counters_of(locus)) == [], locus.bed_name
