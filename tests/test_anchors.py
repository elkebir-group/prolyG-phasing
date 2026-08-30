"""Tests for prolyg_phasing.io._anchors.hamming_search.

The function is numpy-vectorized for performance (extraction profiles as
CPU-bound on it — see the PR that added this file). These tests pin its
contract against a naive reference implementation so a future rewrite
cannot silently change behavior.
"""

from __future__ import annotations

import random

import pytest

from prolyg_phasing.io._anchors import hamming_search


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


def test_exact_match_at_start():
    assert hamming_search("ACGTACGT", "ACGT") == (0, 0)


def test_exact_match_mid_read():
    assert hamming_search("TTTTACGTTTTT", "ACGT") == (0, 4)


def test_no_exact_match_returns_best_offset():
    # anchor "AAAA" vs "ACGT...": best alignment has 3 mismatches somewhere.
    dist, off = hamming_search("ACGTACGT", "AAAA")
    assert dist == naive_hamming_search("ACGTACGT", "AAAA")[0]
    assert off == naive_hamming_search("ACGTACGT", "AAAA")[1]


def test_read_shorter_than_anchor():
    assert hamming_search("AC", "ACGTACGT") == (8, -1)


def test_read_equal_length_to_anchor():
    assert hamming_search("ACGA", "ACGT") == (1, 0)


def test_tie_returns_leftmost_offset():
    # "TTTT" is equidistant (distance 3) from every 4-mer window of
    # "ACGTACGT" that starts on a non-T base; leftmost must win.
    read, anchor = "ACGTACGT", "TTTT"
    assert hamming_search(read, anchor) == naive_hamming_search(read, anchor)


@pytest.mark.parametrize("seed", range(20))
def test_matches_naive_reference_on_random_inputs(seed):
    rng = random.Random(seed)
    bases = "ACGTN"
    for _ in range(200):
        read_len = rng.randint(0, 30)
        anchor_len = rng.randint(1, 12)
        read_seq = "".join(rng.choice(bases) for _ in range(read_len))
        anchor = "".join(rng.choice(bases) for _ in range(anchor_len))
        assert hamming_search(read_seq, anchor) == naive_hamming_search(read_seq, anchor)
