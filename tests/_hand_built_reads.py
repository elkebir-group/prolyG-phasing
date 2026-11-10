"""Hand-built reads with a known true run tuple, and every alignment of each.

Each case is one read at one real hg38 locus, written in polyG orientation,
with the tuple the molecule carries. The locus sequence is the public hg38
reference: 30 bases on each side of the reference tract (from the UCSC
REST API). No base here comes from a sequenced read.

For each read, `placements` lists the alignments an aligner can choose for
the same read string: every shift of a gap along the bases it can slide
over, the gapless form with mismatches, and soft-clipped forms. `to_record`
writes one placement as a `pysam` record with a correct MD tag, on a
synthetic chromosome that holds the locus in "+" or "-" orientation.

Each case sets the flank of each side: the run-side base sits `k` bases
further out than the edge base (the first non-G base next to the repeat),
and a masked offset marks a flank position where the donor has a G.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator

import pysam

from prolyg_phasing.io._anchors import revcomp

FLANK_LEN = 10

# CIGAR operation codes, as pysam reports them.
M, INS, DEL, SOFT = 0, 1, 2, 4


@dataclasses.dataclass(frozen=True)
class LocusSequence:
    """One hg38 locus in polyG orientation: 30 bases, the reference tract, 30 bases."""

    name: str
    hg38_orientation: str
    upstream: str
    tract: str
    downstream: str

    @property
    def reference(self) -> str:
        return self.upstream + self.tract + self.downstream


# hg38, polyG orientation. The tract is the reference sequence between the
# two edge bases.
LOCI: dict[str, LocusSequence] = {
    loc.name: loc
    for loc in (
        LocusSequence("PolyG151", "+", "ACGCATAGGTTTCAGAAGCCATATTTAGAT", "G" * 12,
                      "TGGTTTCAAGATCCTGGAAGAGTGGGTTAT"),
        LocusSequence("PolyG454", "-", "GGAGAAGGGGTCAGAAGAGGCAAGAGAGCC", "G" * 10,
                      "AAACCACTTAAGAAGCTGTGGAGATTTAGA"),
        LocusSequence("PolyG2733", "-", "GCTGCTCTCCATGTGATCCTGGGGTGGGTT", "G" * 11,
                      "CGCGGAGTGGGACGGCAAGGCTGAGTGATC"),
        LocusSequence("PolyG2831", "+", "CTCACCCACCTTATGTATACATATGTGTGT", "G" * 10,
                      "ATGGTCACACTACACTAAAACGAAAGTAGT"),
        LocusSequence("PolyG3550", "+", "CCAGAAATGCCCAGGCTCCCAGAAATGCCC", "G" * 13,
                      "TGTTCCAGTCACAGAAGGGTCCTGGGTGGG"),
        LocusSequence("PolyG4068", "-", "AGGATATCAACGGGGTGGGGCCCGGCGGGT", "G" * 10,
                      "TCCCAACAAGGAAACTCTGTGTGATTGGTT"),
        LocusSequence("PolyG4732", "+", "GCTGAGAGCAGGGTCTCAAGGGTAGACAGA", "G" * 11,
                      "TCAGAGAGGGGCATTCGTGGGGTATGGACA"),
        LocusSequence("PolyG5177", "+", "GCAGGTGAGATGGAGTACAGCTCACTGGGT", "G" * 11,
                      "ATAGAGATGCCCACACATCTGCTGGCATCT"),
        LocusSequence("PolyG7357", "-", "GGACCATTTAGAGGCTACCTGGACCTGGTT", "G" * 10,
                      "TCTGGTGAGCCCCTCGGCCTGGCACGTGCC"),
        LocusSequence("PolyG8740", "+", "GATCCACTATAAAATGGTGGTGGTGGTGGT", "G" * 12,
                      "TGGTTTTGAAAAAAGCCATGGACTGATGCA"),
        LocusSequence("PolyG8882", "+", "TGAGCTATAAGTGACACAATATACCTTTGC", "G" * 10,
                      "TACAGGAATCTCTTATGTTAAAGATAATCT"),
    )
}


@dataclasses.dataclass(frozen=True)
class FlankSpec:
    """One side's flank, in polyG-orientation coordinates.

    `start` indexes `LocusSequence.reference`. `run_side_offset` is 9 for
    the upstream flank and 0 for the downstream flank.
    """

    bases: str
    start: int
    run_side_offset: int
    masked_offsets: tuple[int, ...] = ()

    @property
    def run_side(self) -> int:
        return self.start + self.run_side_offset


def upstream_flank(locus: LocusSequence, k: int, masked: tuple[int, ...] = ()) -> FlankSpec:
    """Upstream flank whose run-side base is `k` bases out from the edge base."""
    end = len(locus.upstream) - k
    start = end - FLANK_LEN
    return FlankSpec(locus.reference[start:end], start, FLANK_LEN - 1, masked)


def downstream_flank(locus: LocusSequence, k: int, masked: tuple[int, ...] = ()) -> FlankSpec:
    """Downstream flank whose run-side base is `k` bases out from the edge base."""
    start = len(locus.upstream) + len(locus.tract) + k
    return FlankSpec(locus.reference[start:start + FLANK_LEN], start, 0, masked)


_NON_G = re.compile(r"[^G]+")


def run_tuple(tract: str) -> tuple[int, ...]:
    """Run tuple of a tract, by the extractor's parse (non-G stretches split runs)."""
    if tract == "":
        return (0,)
    return tuple(len(r) for r in _NON_G.split(tract))


@dataclasses.dataclass(frozen=True)
class ReadCase:
    """One hand-built read with its truth.

    `read` and `ref_start` are in polyG orientation: the read's first base
    sits at `ref_start` of `locus.reference` in the intended alignment.
    `true_tract` is the molecule's tract; `true_tuple` follows from it.
    `drop_reason` is set when the read must be dropped instead.
    `ambiguous` marks a read whose string equals the read of another
    molecule with a different tuple; `true_tuple` then names the molecule
    the read was built from, and the test pins the tuple the convention
    assigns.
    """

    case_id: str
    description: str
    locus: LocusSequence
    up: FlankSpec
    dn: FlankSpec
    read: str
    ref_start: int
    true_tract: str | None
    drop_reason: str | None = None
    ambiguous: bool = False
    ref_end: int = 0  # reference index one past the base the read's last base stands for
    head_clip: int = 0  # read bases before `ref_start` that are not reference bases

    @property
    def true_tuple(self) -> tuple[int, ...] | None:
        return None if self.true_tract is None else run_tuple(self.true_tract)


def _sub(s: str, i: int, base: str) -> str:
    return s[:i] + base + s[i + 1:]


def _ins(s: str, i: int, bases: str) -> str:
    return s[:i] + bases + s[i:]


def _del(s: str, i: int, n: int = 1) -> str:
    return s[:i] + s[i + n:]


def _case(case_id, description, name, *, k_up=0, k_dn=0, up_masked=(), up_ctx=None,
          tract, dn_ctx=None, start=5, dn_len=25, drop_reason=None, ambiguous=False,
          true_tract="tract") -> ReadCase:
    """Assemble a read as upstream context + tract + downstream context.

    `up_ctx` and `dn_ctx` default to the reference, cut at the run-side
    bases of the two flanks; pass an edited copy to put an error in them.
    The read starts `start` bases into the upstream 30 and keeps `dn_len`
    bases past the downstream run-side base.
    """
    locus = LOCI[name]
    up = upstream_flank(locus, k_up, up_masked)
    dn = downstream_flank(locus, k_dn)
    ref = locus.reference
    if up_ctx is None:
        up_ctx = ref[start:up.run_side + 1]
    if dn_ctx is None:
        dn_ctx = ref[dn.run_side:dn.run_side + dn_len]
    read = up_ctx + tract + dn_ctx
    return ReadCase(
        case_id, description, locus, up, dn, read, start,
        tract if true_tract == "tract" else true_tract,
        drop_reason, ambiguous, ref_end=dn.run_side + dn_len,
    )


def _up(name: str, k: int = 0, start: int = 5) -> str:
    """Reference upstream context from `start` to the upstream run-side base."""
    locus = LOCI[name]
    return locus.upstream[start:len(locus.upstream) - k]


def _dn(name: str, k: int = 0, n: int = 25) -> str:
    """Reference downstream context from the downstream run-side base, `n` bases."""
    return LOCI[name].downstream[k:k + n]


G = "G"


def build_cases() -> list[ReadCase]:
    c: list[ReadCase] = []

    # Edge variant, homozygous: the donor has a G at both edge bases, so both flanks move.
    c.append(_case("edge_variant_hom", "PolyG5177: reads CTCACT G13 TAGAG", "PolyG5177",
                   k_up=4, k_dn=1, tract=G * 13))
    # Edge variant, heterozygous: the upstream flank moves one base; one read per allele.
    c.append(_case("edge_variant_het_ref", "PolyG7357 reference allele: T G10", "PolyG7357",
                   k_up=1, tract="T" + G * 10))
    c.append(_case("edge_variant_het_g", "PolyG7357 G allele: G12", "PolyG7357",
                   k_up=1, tract=G * 12))
    # Deleted edge base: reference CC G10, reads C G6; flank at the outer C.
    c.append(_case("deleted_edge_base", "PolyG454: reads C G6", "PolyG454",
                   k_up=1, tract=G * 6))
    # A read error shows a G at the edge base of an invariant side:
    # the read is dropped, not lengthened.
    c.append(_case("false_g_invariant", "PolyG151: G at the edge T", "PolyG151",
                   up_ctx=_up("PolyG151")[:-1] + "G",
                   tract=G * 12, drop_reason="g_at_run_side", true_tract=None))
    # Edge interrupter: TTTGC A G9.
    c.append(_case("edge_interrupter", "PolyG8882: TTTGC A G9", "PolyG8882",
                   tract="A" + G * 9))
    # (TGG)n flank with one copy more and one copy less.
    up8740 = _up("PolyG8740")  # starts at upstream[5]; the (TGG)n starts at upstream[14]
    c.append(_case("tgg_flank_extra_copy", "PolyG8740: one more TGG", "PolyG8740",
                   up_ctx=_ins(up8740, 14 - 5, "TGG"), tract=G * 12))
    c.append(_case("tgg_flank_missing_copy", "PolyG8740: one TGG less", "PolyG8740",
                   up_ctx=_del(up8740, 14 - 5, 3), tract=G * 12))
    # Tandem-copy flank: the read covers both copies of the upstream anchor.
    c.append(_case("tandem_copy_flank", "PolyG3550: read covers both anchor copies",
                   "PolyG3550", start=0, tract=G * 13))
    # Post-run indel errors in the downstream flank at PolyG4068. The donor
    # carries a G at the upstream edge base, so that flank moves out 4.
    up4068 = _up("PolyG4068", 4)
    c.append(_case("post_run_deletion", "PolyG4068: one C of CCC deleted", "PolyG4068",
                   k_up=4, up_ctx=up4068, tract="GGG" + G + G * 10,
                   dn_ctx=_del(_dn("PolyG4068"), 2)))
    c.append(_case("post_run_insertion", "PolyG4068: one C inserted in CCC", "PolyG4068",
                   k_up=4, up_ctx=up4068, tract="GGG" + G + G * 10,
                   dn_ctx=_ins(_dn("PolyG4068"), 2, "C")))
    c.append(_case("post_run_deletion_aa", "PolyG4068: one A of AA deleted", "PolyG4068",
                   k_up=4, up_ctx=up4068, tract="GGG" + G + G * 10,
                   dn_ctx=_del(_dn("PolyG4068"), 4)))
    # A one-base deletion in the TTT of the upstream flank: the CIGAR path on
    # the upstream side.
    up151 = _up("PolyG151")
    c.append(_case("pre_run_deletion", "PolyG151: one T of TTT deleted", "PolyG151",
                   up_ctx=_del(up151, len(up151) - 6), tract=G * 12))
    # Heterozygous SNV inside the upstream flank, 6 bases from the run side.
    c.append(_case("snv_in_flank", "PolyG151: T>C 6 bases out", "PolyG151",
                   up_ctx=_sub(up151, len(up151) - 7, "C"), tract=G * 12))
    # Non-G substitution at the run-side base: PolyG2733 downstream C>T.
    c.append(_case("non_g_run_side", "PolyG2733: downstream run-side C>T", "PolyG2733",
                   tract=G * 11, dn_ctx="T" + _dn("PolyG2733")[1:]))
    # G at the run-side base of a moved flank (PolyG7357 outer T).
    up7357 = _up("PolyG7357", 1)
    c.append(_case("false_g_moved", "PolyG7357: G at the outer T", "PolyG7357",
                   k_up=1, up_ctx=up7357[:-1] + "G", tract="T" + G * 10,
                   drop_reason="g_at_run_side", true_tract=None))
    # An invariant side; two G candidates in a row; a flank that moves 4 bases.
    c.append(_case("invariant_side", "PolyG151: reference read", "PolyG151", tract=G * 12))
    c.append(_case("two_candidates_moved", "PolyG4732: C G G G G10", "PolyG4732",
                   k_up=3, tract=G * 13))
    c.append(_case("flank_moves_four", "PolyG2831: G10 ATGG", "PolyG2831",
                   k_dn=4, tract=G * 10 + "ATGG"))
    # Masked flank position: the donor has a G 7 out; the read has that G and
    # one more mismatch. The mask is hypothetical at this locus.
    c.append(_case("masked_plus_mismatch", "PolyG151: G at masked offset 2 and one mismatch",
                   "PolyG151", up_masked=(2,),
                   up_ctx=_sub(_sub(up151, len(up151) - 8, "G"), len(up151) - 4, "C"),
                   tract=G * 12))
    # Two edits in the flank: dropped.
    c.append(_case("two_edits", "PolyG151: two mismatches in the upstream flank", "PolyG151",
                   up_ctx=_sub(_sub(up151, len(up151) - 7, "C"), len(up151) - 4, "C"),
                   tract=G * 12, drop_reason="flank_not_placed", true_tract=None))
    # Near a read end, the read's tract is longer than the reference's: the
    # read ends 2 bases past the downstream flank, or starts 2 bases before
    # the upstream flank.
    c.append(_case("near_end_downstream", "PolyG151: G15, read ends after the flank",
                   "PolyG151", tract=G * 15, dn_len=12))
    c.append(_case("near_end_upstream", "PolyG151: G15, read starts before the flank",
                   "PolyG151", start=18, tract=G * 15))
    # Reads that are ambiguous by construction: the read string also comes
    # from a molecule with another tuple.
    c.append(_case("boundary_t_deletion", "PolyG7357 reference allele, one T of TT deleted",
                   "PolyG7357", k_up=1, up_ctx=up7357, tract=G * 10,
                   ambiguous=True, true_tract="T" + G * 10))
    c.append(_case("boundary_t_insertion", "PolyG4068: one T inserted after the run",
                   "PolyG4068", k_up=4, up_ctx=up4068, tract="GGG" + G + G * 10,
                   dn_ctx="T" + _dn("PolyG4068"), ambiguous=True))
    c.append(_case("tgg_flank_substitution", "PolyG8740: G>A next to the edge T",
                   "PolyG8740", up_ctx=up8740[:-2] + "A" + up8740[-1], tract=G * 12))
    c.append(_case("tandem_copy_error_in_near_copy",
                   "PolyG3550: both copies, one mismatch in the near copy",
                   "PolyG3550", start=0,
                   up_ctx=_sub(_up("PolyG3550", 0, 0), 23, "C"), tract=G * 13))
    # The read ends inside the tract, 4 bases before the downstream flank, so
    # the projected flank is not inside the read.
    ends_in_tract = _case("read_ends_in_tract", "PolyG151: read ends after 8 of 12 Gs",
                          "PolyG151", tract=G * 8, dn_ctx="",
                          drop_reason="flank_not_covered", true_tract=None)
    c.append(dataclasses.replace(ends_in_tract,
                                 ref_end=ends_in_tract.up.run_side + 1 + 8))
    # The molecule lost the upstream edge T and all 12 Gs. The upstream flank
    # and the downstream flank then place their run-side bases on one T.
    c.append(_case("tract_and_edge_deleted", "PolyG151: edge T and G12 deleted",
                   "PolyG151", up_ctx=up151[:-1], tract="",
                   drop_reason="flanks_crossed", true_tract=None))
    # A 2-base deletion (AG of TTTAGAT) inside the upstream flank: two edits.
    c.append(_case("two_base_deletion_in_flank", "PolyG151: AG deleted from TTTAGAT",
                   "PolyG151", up_ctx=_del(up151, len(up151) - 4, 2), tract=G * 12,
                   drop_reason="flank_not_placed", true_tract=None))
    return c


def soft_clipped_flank_case() -> ReadCase:
    """A read whose first bases are not reference, so the aligner clips into the flank.

    Six non-reference bases precede the upstream flank's outer three bases.
    One form clips the six non-reference bases; the other clips three flank bases too.
    """
    locus = LOCI["PolyG151"]
    up = upstream_flank(locus, 0)
    dn = downstream_flank(locus, 0)
    read = "CTCTCT" + locus.reference[up.start:up.run_side + 1] + G * 12 + _dn("PolyG151")
    return ReadCase("soft_clip_in_flank", "PolyG151: clip over the outer flank bases",
                    locus, up, dn, read, up.start, G * 12,
                    ref_end=dn.run_side + 25, head_clip=6)


# ---------------------------------------------------------------------------
# Equivalent placements
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Placement:
    """One alignment of a read, polyG orientation: reference start and CIGAR."""

    form: str
    ref_start: int
    cigar: tuple[tuple[int, int], ...]


def _mismatches(read: str, ref: str, ref_start: int, cigar) -> int:
    q, p, n = 0, ref_start, 0
    for op, length in cigar:
        if op == M:
            n += sum(a != b for a, b in zip(read[q:q + length], ref[p:p + length], strict=True))
            q += length
            p += length
        elif op in (INS, SOFT):
            q += length
        elif op == DEL:
            p += length
    return n


def _ref_span(cigar) -> int:
    return sum(length for op, length in cigar if op in (M, DEL))


def _compact(cigar) -> tuple[tuple[int, int], ...]:
    out: list[list[int]] = []
    for op, length in cigar:
        if length == 0:
            continue
        if out and out[-1][0] == op:
            out[-1][1] += length
        else:
            out.append([op, length])
    return tuple((op, length) for op, length in out)


def placements(case: ReadCase, *, min_aligned: int = 20) -> list[Placement]:
    """Every alignment of `case.read` that an aligner can choose.

    - **Gap shifts.** The read differs from the reference span by `delta`
      bases. Every one-gap alignment of length `|delta|` with the fewest
      mismatches is listed. These are the shifts of the gap along the
      homopolymer or tandem unit it sits in, plus any equal-score
      rearrangement that trades a gap shift for a mismatch.
    - **Gapless.** Anchored at the read end away from the length change, so
      the mismatches fall between the change and the near read end. An
      aligner extends from the long exact segment; it does not shift it.
    - **Soft-clipped.** The gapless form with the mismatched segment clipped:
      from its first mismatch to the read end, or from the read start to
      its last mismatch, on the side where the mismatches lie.
    """
    ref = case.locus.reference
    read = case.read
    clip = case.head_clip
    body = read[clip:]
    out: list[Placement] = []

    ref_end = case.ref_end
    delta = len(body) - (ref_end - case.ref_start)
    head = ((SOFT, clip),) if clip else ()

    if delta == 0:
        gapless = head + ((M, len(body)),)
        out.append(Placement("gapless", case.ref_start, _compact(gapless)))
    else:
        length = abs(delta)
        best: list[tuple[int, tuple]] = []
        for q in range(1, len(body)):
            if delta > 0:
                if q + length >= len(body):
                    break
                cig = ((M, q), (INS, length), (M, len(body) - q - length))
            else:
                cig = ((M, q), (DEL, length), (M, len(body) - q))
            cig = head + cig
            if case.ref_start + _ref_span(cig) > len(ref):
                continue
            best.append((_mismatches(read, ref, case.ref_start, cig), _compact(cig)))
        fewest = min(m for m, _ in best)
        seen = set()
        for m, cig in best:
            if m == fewest and cig not in seen:
                seen.add(cig)
                out.append(Placement("gap_shift", case.ref_start, cig))
        # Gapless, anchored at the read end away from the length change. The
        # aligner extends from the long exact segment, so only the short
        # segment between the change and the near read end shows mismatches.
        gap_q = sum(p.cigar[0][1] for p in out) / len(out)
        if gap_q >= len(body) / 2:
            form, start = "gapless_start_anchored", case.ref_start
        else:
            form, start = "gapless_end_anchored", ref_end - len(body)
        if start >= 0 and start + len(body) <= len(ref):
            out.append(Placement(form, start, _compact(head + ((M, len(body)),))))

    # A clip that runs from the non-reference head into the flank's outer bases.
    if clip:
        into = 3
        cig = ((SOFT, clip + into), (M, len(body) - into))
        out.append(Placement("clipped_into_flank", case.ref_start + into, cig))

    # Soft-clipped form of the gapless placement: the short segment past the
    # first (or before the last) mismatch is clipped, on the same side as
    # the mismatches.
    for p in [p for p in out if p.form.startswith("gapless")]:
        mism = [i for i in range(clip, len(read))
                if read[i] != ref[p.ref_start + i - clip]]
        if not mism:
            continue
        first, last = mism[0], mism[-1]
        tail_side = (first + last) / 2 >= len(read) / 2
        if tail_side and first - clip >= min_aligned:
            cig = head + ((M, first - clip), (SOFT, len(read) - first))
            out.append(Placement(p.form + "_tail_clipped", p.ref_start, _compact(cig)))
        if not tail_side and len(read) - last - 1 >= min_aligned and clip == 0:
            cig = ((SOFT, last + 1), (M, len(read) - last - 1))
            out.append(Placement(p.form + "_head_clipped", p.ref_start + last + 1,
                                 _compact(cig)))
    return out


# ---------------------------------------------------------------------------
# pysam records
# ---------------------------------------------------------------------------

CHROM = "synth"
CHROM_PAD = 20  # neutral bases on each side of the locus on the synthetic chromosome


def chromosome(locus: LocusSequence, orientation: str) -> str:
    """Synthetic chromosome: padding + locus (polyG orientation, or its reverse complement)."""
    body = locus.reference if orientation == "+" else revcomp(locus.reference)
    return "A" * CHROM_PAD + body + "A" * CHROM_PAD


def md_tag(read: str, ref: str, ref_start: int, cigar) -> str:
    """MD tag of a gapped alignment. Insertions and soft clips do not appear in MD."""
    parts: list[str] = []
    run = 0
    q, p = 0, ref_start
    for op, length in cigar:
        if op == M:
            for i in range(length):
                if read[q + i] == ref[p + i]:
                    run += 1
                else:
                    parts.append(f"{run}{ref[p + i]}")
                    run = 0
            q += length
            p += length
        elif op in (INS, SOFT):
            q += length
        elif op == DEL:
            parts.append(f"{run}^{ref[p:p + length]}")
            run = 0
            p += length
    parts.append(str(run))
    return "".join(parts)


def to_reference_orientation(case: ReadCase, placement: Placement, orientation: str
                             ) -> tuple[str, int, tuple[tuple[int, int], ...]]:
    """The read, its start and CIGAR on the synthetic chromosome."""
    if orientation == "+":
        return case.read, CHROM_PAD + placement.ref_start, placement.cigar
    span = _ref_span(placement.cigar)
    start = CHROM_PAD + len(case.locus.reference) - (placement.ref_start + span)
    return revcomp(case.read), start, tuple(reversed(placement.cigar))


def to_record(header: pysam.AlignmentHeader, case: ReadCase, placement: Placement,
              orientation: str, qname: str) -> pysam.AlignedSegment:
    """One placement as a `pysam` record with MD and NM tags."""
    chrom = chromosome(case.locus, orientation)
    seq, start, cigar = to_reference_orientation(case, placement, orientation)
    seg = pysam.AlignedSegment(header)
    seg.query_name = qname
    seg.query_sequence = seq
    seg.query_qualities = pysam.qualitystring_to_array("I" * len(seq))
    seg.flag = 0
    seg.reference_id = 0
    seg.reference_start = start
    seg.mapping_quality = 60
    seg.cigartuples = list(cigar)
    md = md_tag(seq, chrom, start, cigar)
    nm = _mismatches(seq, chrom, start, cigar) + sum(
        length for op, length in cigar if op in (INS, DEL))
    seg.set_tags([("MD", md), ("NM", nm), ("MI", f"{qname}/A", "Z")])
    return seg


def header_for(locus: LocusSequence, orientation: str) -> pysam.AlignmentHeader:
    return pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": CHROM, "LN": len(chromosome(locus, orientation))}],
    })


def all_records(
    case: ReadCase, orientation: str
) -> Iterator[tuple[Placement, pysam.AlignedSegment]]:
    header = header_for(case.locus, orientation)
    for i, p in enumerate(placements(case)):
        yield p, to_record(header, case, p, orientation, f"{case.case_id}_{i}")


# ---------------------------------------------------------------------------
# Plain dynamic-programming reference for the flank match
# ---------------------------------------------------------------------------


def flank_edit_distance(flank: str, read: str, run_side_index: int, *, upstream: bool,
                        masked: tuple[int, ...] = ()) -> int:
    """Fewest edits that place `flank` with its run-side base on `read[run_side_index]`.

    The run-side base is matched against that read base and costs nothing:
    the caller checks it is a non-G base. The other nine flank bases align
    to the read bases on the far side of it, by unit-cost edit distance,
    with the outer end free in the read. A masked flank offset matches any
    read base. Plain quadratic DP; the reference for the kernel's band-1
    search.
    """
    if upstream:
        pattern = flank[:-1][::-1]
        text = read[:run_side_index][::-1]
        mask = {len(flank) - 2 - m for m in masked if m != len(flank) - 1}
    else:
        pattern = flank[1:]
        text = read[run_side_index + 1:]
        mask = {m - 1 for m in masked if m != 0}
    # Row i: pattern prefix of length i. Column j: text prefix of length j.
    # Pattern starts at the run side, so it is anchored at text position 0;
    # its outer end may stop at any text position.
    n, m = len(pattern), len(text)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            sub = 0 if (i - 1 in mask or pattern[i - 1] == text[j - 1]) else 1
            cur[j] = min(prev[j - 1] + sub, prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return min(prev)
