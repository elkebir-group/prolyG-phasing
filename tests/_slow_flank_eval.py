"""A plain, slow reference for the flank match, the placement and the objective.

It shares no code with the library. It compares substrings directly, one
read and one position at a time. All values are in half units: a tract G
scores +2, any other base -4, a substitution costs 2, and a gap of g bases
costs 3 + g. A match allows one substitution, or one gap of 1 or 2 bases
between two inner flank bases, never both. A placement needs a marked read
base strictly between its two edges; `marked` is the set of marked read
positions in polyG orientation.
"""

from __future__ import annotations

from collections.abc import Collection

MAX_GAP = 2


def every_base(read: str) -> frozenset[int]:
    """Every position of the read: the marks of a read that lies inside the BED interval."""
    return frozenset(range(len(read)))


def marks(reads: list[str], marked: list[Collection[int]] | None = None
          ) -> list[tuple[tuple[int, int], ...]]:
    """The library's form of the marks: per read, the sorted runs of marked
    positions as ``(start, end)``. With `marked` None, every base is marked."""
    out = []
    for row, read in enumerate(reads):
        positions = sorted(every_base(read) if marked is None else marked[row])
        runs: list[list[int]] = []
        for i in positions:
            if runs and runs[-1][1] == i:
                runs[-1][1] = i + 1
            else:
                runs.append([i, i + 1])
        out.append(tuple((a, b) for a, b in runs))
    return out


def _holds_mark(marked: Collection[int], a: int, b: int) -> bool:
    return any(a < i < b for i in marked)


def _mismatches_up_to_one(s: str, x: str) -> int | None:
    """The mismatches of two equal-length strings if at most one, else None."""
    mm = 0
    for a, b in zip(s, x, strict=True):
        if a != b:
            mm += 1
            if mm > 1:
                return None
    return mm


def substring_cost(s: str, x: str) -> int | None:
    """Lowest cost of the 10-base flank `x` against the whole substring `s`, or None."""
    k = len(s)
    if k == len(x):
        mm = _mismatches_up_to_one(s, x)
        return None if mm is None else 2 * mm
    best = None
    if len(x) < k <= len(x) + MAX_GAP:
        g = k - len(x)
        for i in range(1, len(x)):
            if s[:i] + s[i + g:] == x:                 # read bases s[i:i+g] are extra
                best = 3 + g
    elif len(x) - MAX_GAP <= k < len(x):
        g = len(x) - k
        for i in range(1, len(x) - g):
            if s == x[:i] + x[i + g:]:                 # x[i:i+g] are missing in the read
                best = 3 + g
    return best


def u_positions(read: str, u: str) -> dict[int, int]:
    """{a: lowest cost} over tract edges a (read[a] != G, read[a + 1] == G)."""
    out = {}
    for a in range(len(read) - 1):
        if read[a] == "G" or read[a + 1] != "G":
            continue
        for k in range(len(u) - MAX_GAP, len(u) + MAX_GAP + 1):
            if a - k + 1 < 0:
                continue
            c = substring_cost(read[a - k + 1:a + 1], u)
            if c is not None and (a not in out or c < out[a]):
                out[a] = c
    return out


def v_positions(read: str, v: str) -> dict[int, int]:
    """{b: lowest cost} over tract edges b (read[b] != G, read[b - 1] == G)."""
    out = {}
    for b in range(1, len(read)):
        if read[b] == "G" or read[b - 1] != "G":
            continue
        for k in range(len(v) - MAX_GAP, len(v) + MAX_GAP + 1):
            if b + k > len(read):
                continue
            c = substring_cost(read[b:b + k], v)
            if c is not None and (b not in out or c < out[b]):
                out[b] = c
    return out


def score(tract: str) -> int:
    return sum(2 if ch == "G" else -4 for ch in tract)


def best_placement(read: str, u: str, v: str, marked: Collection[int]
                   ) -> tuple[int, int, int] | None:
    """(value, a, b) of the read's best placement, or None.

    Of placements with the top value, the largest a wins, then the
    smallest b: the shortest tract.
    """
    best = None
    ups, downs = u_positions(read, u), v_positions(read, v)
    for a in sorted(ups, reverse=True):
        for b in sorted(downs):
            if b <= a or not _holds_mark(marked, a, b):
                continue
            val = score(read[a + 1:b]) - ups[a] - downs[b]
            if best is None or val > best[0]:
                best = (val, a, b)
    return best


def objective(reads: list[tuple[str, Collection[int], int]], u: str, v: str
              ) -> tuple[int, int]:
    """F in half units, and the reads with a placement. `reads` is (string, marked, count)."""
    total, placed = 0, 0
    for read, marked, count in reads:
        p = best_placement(read, u, v, marked)
        if p is None:
            continue
        total += count * max(0, p[0])
        placed += count
    return total, placed


def candidates(reads: list[tuple[str, Collection[int], int]]) -> tuple[set[str], set[str]]:
    """Every 10-base A/C/G/T string at a tract edge of a placement, by the constraints.

    A ``u`` ends at an upstream edge ``a`` and does not end with three G and
    then its last base; a ``v`` starts at a downstream edge ``b`` and does
    not start with its first base and then three G. The edge must face an
    edge on the other side, with room for a match there (8 bases) and a
    marked base in between.
    """
    us, vs = set(), set()
    for read, marked, _ in reads:
        n = len(read)
        ups = [a for a in range(n - 1) if read[a] != "G" and read[a + 1] == "G"]
        downs = [b for b in range(1, n) if read[b] != "G" and read[b - 1] == "G"]
        for a in ups:
            u = read[a - 9:a + 1]
            if (a >= 9 and set(u) <= set("ACGT") and u[-4:-1] != "GGG"
                    and any(b <= n - 8 and _holds_mark(marked, a, b) for b in downs)):
                us.add(u)
        for b in downs:
            v = read[b:b + 10]
            if (b <= n - 10 and set(v) <= set("ACGT") and v[1:4] != "GGG"
                    and any(a >= 7 and _holds_mark(marked, a, b) for a in ups)):
                vs.add(v)
    return us, vs
