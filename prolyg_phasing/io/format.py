"""The one rendering of an interrupter pattern / allele / diploid genotype.

There were four, and they disagreed. ``io.panel._format_pattern`` joined
interrupters with "_", ``plotting._view.format_pattern`` with "·", and
``inference.infer.fmt_geno`` (the published column) with **nothing at all**
-- which is not merely inconsistent but lossy: an interrupter is a non-G
*string*, not a base, so ("CC","T","T") and ("C","CT","T") both rendered
"CCTT" and could not be told apart. That reached the published
``gexp_calls.tsv``, where 305 of 248,080 rendered alleles were ambiguous.
The run count bounds how many interrupters there are (|l| - 1) but says
nothing about where the boundaries fall.

One function, one parameter. ``max_token`` is the only thing that
legitimately differs between a file column and a plot legend: a
14-character interrupter overflows a tile title, and must NOT be
abbreviated in an artifact.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type-only: Allele/LocusGenotype back the diploid genotype call, which
    # this package never makes, so this stays outside a hard install-time
    # dependency (`from __future__ import annotations` above already
    # stringifies the two signatures that name them).
    from prolyG.inference.genotype import Allele, LocusGenotype

#: Between interrupter strings. Must not be a base symbol and must not be empty,
#: or a multi-base interrupter is indistinguishable from several single-base ones.
PATTERN_SEP = "_"
#: Between run lengths.
LENGTH_SEP = "."
#: Between the two alleles of a diploid genotype.
ALLELE_SEP = "/"
#: A pattern with no interrupters (a single-run, all-G allele).
NO_PATTERN = "-"


def format_pattern(haplotype, *, max_token: int | None = None) -> str:
    """Interrupter pattern :math:`h` as text; ``"-"`` when there are none.

    Parameters
    ----------
    haplotype : tuple of str
        The non-G interrupter strings, in order.
    max_token : int, optional
        Abbreviate any token longer than this to ``first2…last2``. ``None``
        (the default) renders faithfully and is what every **artifact** must
        use; a plot passes a limit so a long token cannot overflow a tile
        title or legend. Abbreviated output is display-only and is not
        round-trippable -- that is the whole difference between the two.
    """
    h = tuple(haplotype)
    if not h:
        return NO_PATTERN
    if max_token is not None:
        h = tuple(t if len(t) <= max_token else f"{t[:2]}…{t[-2:]}" for t in h)
    return PATTERN_SEP.join(h)


def format_allele(allele: Allele, *, max_token: int | None = None) -> str:
    """One allele as ``pattern:len.len``.

    Takes an :class:`~prolyG.inference.genotype.Allele`, which is the one
    representation every caller now carries. It previously also accepted a
    bare ``(haplotype, lengths)`` pair; that branch let a caller hand over
    an allele whose length vector did not match its pattern, which is how
    an impossible allele reached a test fixture. ``Allele`` rejects that
    pair at construction instead.
    """
    lens = LENGTH_SEP.join(str(int(x)) for x in allele.lengths)
    return f"{format_pattern(allele.haplotype, max_token=max_token)}:{lens}"


def format_genotype(genotype: LocusGenotype, *, max_token: int | None = None) -> str:
    """A diploid genotype as ``allele/allele``, in :math:`\\gamma` order.

    Takes a :class:`~prolyG.inference.genotype.LocusGenotype`. This is what
    ``gexp_calls.tsv``'s ``g_norm`` and ``g_exp_marginal`` columns hold; it
    is called with the default ``max_token`` there, so the text is
    faithful.
    """
    return ALLELE_SEP.join(
        format_allele(a, max_token=max_token)
        for a in (genotype.gamma1, genotype.gamma2)
    )
