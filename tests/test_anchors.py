"""Tests for prolyg_phasing.io._anchors.hamming_search.

The function scores a whole batch of reads against one anchor, settling
the reads that carry the anchor verbatim with a substring search and
scoring the rest with one set of numpy calls per block, because
extraction profiles as CPU-bound on it. These tests pin its contract
against a naive per-read reference implementation so a future rewrite
cannot silently change behavior. They mix read lengths within a batch and
mix exact against inexact reads — those are where a batched
implementation can go wrong without a single-read test noticing.
"""

from __future__ import annotations

import random

import pytest

from prolyg_phasing.io._anchors import _HAMMING_BLOCK, hamming_search


def naive_hamming_search(read_seq: str, anchor: str) -> tuple[int, int]:
    """Reference implementation: brute-force sliding window, leftmost tie-break."""
    k = len(anchor)
    if len(read_seq) < k:
        return (k, -1)
    best = k + 1
    best_off = -1
    for i in range(len(read_seq) - k + 1):
        d = sum(c1 != c2 for c1, c2 in zip(read_seq[i : i + k], anchor, strict=True))
        if d < best:
            best = d
            best_off = i
    return (best, best_off)


def search_one(read_seq: str, anchor: str) -> tuple[int, int]:
    """`hamming_search` on a one-read batch, as a plain `(distance, offset)`."""
    distances, offsets = hamming_search([read_seq], anchor)
    assert distances.shape == (1,)
    assert offsets.shape == (1,)
    return (int(distances[0]), int(offsets[0]))


def test_exact_match_at_start():
    assert search_one("ACGTACGT", "ACGT") == (0, 0)


def test_exact_match_mid_read():
    assert search_one("TTTTACGTTTTT", "ACGT") == (0, 4)


def test_no_exact_match_returns_best_offset():
    # anchor "AAAA" vs "ACGT...": best alignment has 3 mismatches somewhere.
    assert search_one("ACGTACGT", "AAAA") == naive_hamming_search("ACGTACGT", "AAAA")


def test_read_shorter_than_anchor():
    assert search_one("AC", "ACGTACGT") == (8, -1)


def test_read_equal_length_to_anchor():
    assert search_one("ACGA", "ACGT") == (1, 0)


def test_tie_returns_leftmost_offset():
    # "TTTT" is equidistant (distance 3) from every 4-mer window of
    # "ACGTACGT" that starts on a non-T base; leftmost must win.
    read, anchor = "ACGTACGT", "TTTT"
    assert search_one(read, anchor) == naive_hamming_search(read, anchor)


def test_empty_batch():
    distances, offsets = hamming_search([], "ACGT")
    assert distances.shape == (0,)
    assert offsets.shape == (0,)


def test_batch_of_mixed_lengths_matches_per_read_reference():
    # A read longer than the others must not lend its trailing offsets to
    # a shorter neighbour, and a read shorter than the anchor must not
    # borrow a placement from the batch. The batch also mixes reads that
    # carry the anchor verbatim with reads that do not, so the two paths
    # have to write their results back to the right rows.
    anchor = "ACGTA"
    reads = ["ACGTA", "TT", "TTTTTACGTATTTTTTTTT", "", "GGGGGGGGGG", "ACGTA" * 4]
    distances, offsets = hamming_search(reads, anchor)
    for i, read in enumerate(reads):
        assert (int(distances[i]), int(offsets[i])) == naive_hamming_search(read, anchor)


def test_batch_with_no_exact_match_anywhere():
    # The substring shortcut fires for nobody, so every read goes through
    # the windowed scan and the write-back path for it.
    anchor = "ACGTACGTAC"
    reads = ["TTTTTTTTTTTTTTT", "GGGGGGGGGGGG", "ACGTACGTAT", "AC"]
    distances, offsets = hamming_search(reads, anchor)
    for i, read in enumerate(reads):
        assert (int(distances[i]), int(offsets[i])) == naive_hamming_search(read, anchor)


def test_batch_spanning_multiple_blocks():
    # More reads than one internal block, so the per-block packing and
    # the write-back into the whole-batch output are exercised.
    rng = random.Random(1234)
    anchor = "ACGTACGTAC"
    reads = [
        "".join(rng.choice("ACGTN") for _ in range(rng.randint(0, 40)))
        for _ in range(2 * _HAMMING_BLOCK + 7)
    ]
    distances, offsets = hamming_search(reads, anchor)
    for i, read in enumerate(reads):
        assert (int(distances[i]), int(offsets[i])) == naive_hamming_search(read, anchor)


@pytest.mark.parametrize("seed", range(20))
def test_matches_naive_reference_on_random_inputs(seed):
    rng = random.Random(seed)
    bases = "ACGTN"
    anchor_len = rng.randint(1, 12)
    anchor = "".join(rng.choice(bases) for _ in range(anchor_len))
    reads = [
        "".join(rng.choice(bases) for _ in range(rng.randint(0, 30)))
        for _ in range(200)
    ]
    distances, offsets = hamming_search(reads, anchor)
    for i, read_seq in enumerate(reads):
        assert (int(distances[i]), int(offsets[i])) == naive_hamming_search(read_seq, anchor)
        # A one-read batch must agree with the same read inside a batch.
        assert search_one(read_seq, anchor) == naive_hamming_search(read_seq, anchor)
