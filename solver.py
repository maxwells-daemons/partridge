"""Brute-force solver for the Partridge Puzzle.

A solution for parameter N is a tiling of an S x S square (S = N(N+1)/2) by
k of each k x k tile for k = 1..N.

State representation: a "skyline" array of length S, where `skyline[c]` is
the row of the topmost empty cell in column c. Because we always place the
next tile at the top-left-most empty cell, the empty region stays staircase-
shaped and the skyline captures the full state exactly.

From the skyline:
  - the next cell to fill is `(skyline[c*], c*)` where c* is the leftmost
    column achieving the minimum skyline height;
  - a k x k tile fits at (r, c*) iff skyline[c*..c*+k-1] are all equal to r,
    r + k <= S, and c* + k <= S;
  - placing advances those k entries to r + k; unplacing reverts them.

Crucially, the maximum k that can possibly fit at (r, c*) is the length of
the run of columns starting at c* whose skyline equals r, bounded by the
board edges. We compute this once per step and only try sizes up to that
bound — this prunes the majority of dead-end tile-size attempts that the
previous bitmask version made with `fits` calls.

Solutions are emitted as a JSON array of [size, row, col] triples, one per
tile, in scan-line placement order.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path


def side_length(n: int) -> int:
    return n * (n + 1) // 2


def initial_counts(n: int) -> list[int]:
    # counts[k] = number of k x k tiles remaining (counts[0] unused).
    return [0] + list(range(1, n + 1))


def _next_cell(skyline: list[int], side: int) -> tuple[int, int, int]:
    """Return (row, col, max_k) for the next placement slot.

    `row` is min(skyline); `col` is the leftmost column achieving it; `max_k`
    is the longest run of `skyline == row` starting at `col`, capped by
    remaining board width and height. Returns (-1, -1, 0) if the board is
    full (min height == side).
    """
    r = min(skyline)
    if r == side:
        return -1, -1, 0
    c = skyline.index(r)
    run = 0
    limit = min(side - r, side - c)
    for i in range(c, c + limit):
        if skyline[i] == r:
            run += 1
        else:
            break
    return r, c, run


# ----- search -----


def _search(
    skyline: list[int],
    counts: list[int],
    placements: list[tuple[int, int, int]],
    side: int,
    n: int,
    on_solution,
    stop: list[bool],
    rng: random.Random | None = None,
) -> None:
    if stop[0]:
        return
    r, c, max_k = _next_cell(skyline, side)
    if r < 0:
        on_solution(placements)
        return
    # Iteration order: shuffle for diversity if rng is given, else larger-first
    # (which prunes the tree fastest).
    order = list(range(min(n, max_k), 0, -1))
    if rng is not None:
        rng.shuffle(order)
    for k in order:
        if counts[k] == 0:
            continue
        # place
        for i in range(c, c + k):
            skyline[i] = r + k
        counts[k] -= 1
        placements.append((k, r, c))
        _search(skyline, counts, placements, side, n, on_solution, stop, rng)
        placements.pop()
        counts[k] += 1
        # unplace
        for i in range(c, c + k):
            skyline[i] = r
        if stop[0]:
            return


def _enumerate_prefixes(
    n: int, depth: int
) -> list[list[tuple[int, int, int]]]:
    """All valid scan-line placement prefixes of the given length."""
    side = side_length(n)
    skyline = [0] * side
    counts = initial_counts(n)
    out: list[list[tuple[int, int, int]]] = []
    placements: list[tuple[int, int, int]] = []

    def rec() -> None:
        if len(placements) == depth:
            out.append(list(placements))
            return
        r, c, max_k = _next_cell(skyline, side)
        if r < 0:
            return
        for k in range(min(n, max_k), 0, -1):
            if counts[k] == 0:
                continue
            for i in range(c, c + k):
                skyline[i] = r + k
            counts[k] -= 1
            placements.append((k, r, c))
            rec()
            placements.pop()
            counts[k] += 1
            for i in range(c, c + k):
                skyline[i] = r

    rec()
    return out


def _worker(args) -> None:
    prefix, n, max_per_worker, queue, seed = args
    rng = random.Random(seed) if seed is not None else None
    side = side_length(n)
    skyline = [0] * side
    counts = initial_counts(n)
    placements: list[tuple[int, int, int]] = []
    for k, r, c in prefix:
        # replay prefix (it's known-valid from enumeration, but stay safe)
        if counts[k] == 0:
            return
        for i in range(c, c + k):
            skyline[i] = r + k
        counts[k] -= 1
        placements.append((k, r, c))

    found = [0]
    stop = [False]

    def on_solution(pl: list[tuple[int, int, int]]) -> None:
        queue.put([list(t) for t in pl])
        found[0] += 1
        if max_per_worker is not None and found[0] >= max_per_worker:
            stop[0] = True

    _search(skyline, counts, placements, side, n, on_solution, stop, rng)
    queue.put(None)  # sentinel signalling this worker is done


def solve(
    n: int = 8,
    output: Path | str = "solutions.jsonl",
    max_solutions: int | None = None,
    workers: int | None = None,
    prefix_depth: int = 3,
    seed: int | None = None,
    randomize: bool = True,
) -> int:
    """Run the solver and stream solutions to `output`. Returns count written.

    When `randomize` is True, each worker uses a distinct RNG seed so they
    explore the search tree in different orders — this makes streamed
    solutions visually diverse rather than sharing long shared prefixes.
    """
    workers = workers or mp.cpu_count()
    prefixes = _enumerate_prefixes(n, prefix_depth)
    if not prefixes:
        prefixes = [[]]

    master_rng: random.Random | None = None
    if randomize:
        master_rng = random.Random(seed)
        master_rng.shuffle(prefixes)

    print(
        f"N={n}: {len(prefixes)} prefixes of depth {prefix_depth}, {workers} workers",
        file=sys.stderr,
    )

    out_path = Path(output)
    count = 0
    start = time.time()

    with mp.Manager() as manager:
        queue = manager.Queue()
        pool = mp.Pool(workers)
        try:
            # When randomizing, cap each worker at 1 solution so m results come
            # from m different prefixes (→ visibly different layouts). Without
            # randomization we keep the old behavior of letting each worker
            # enumerate up to max_solutions.
            per_worker_cap = 1 if master_rng is not None else max_solutions
            tasks = [
                (
                    pfx,
                    n,
                    per_worker_cap,
                    queue,
                    master_rng.randrange(2**63) if master_rng else None,
                )
                for pfx in prefixes
            ]
            pool.map_async(_worker, tasks)
            pool.close()

            remaining_workers = len(prefixes)
            with out_path.open("w") as f:
                while remaining_workers > 0:
                    item = queue.get()
                    if item is None:
                        remaining_workers -= 1
                        continue
                    f.write(json.dumps(item) + "\n")
                    f.flush()
                    count += 1
                    if max_solutions is not None and count >= max_solutions:
                        break
        finally:
            pool.terminate()
            pool.join()

    elapsed = time.time() - start
    print(f"Wrote {count} solutions to {out_path} in {elapsed:.1f}s", file=sys.stderr)
    return count


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-n", "--size", type=int, default=8, help="Puzzle parameter N")
    p.add_argument("-o", "--output", default="solutions.jsonl")
    p.add_argument(
        "-m",
        "--max-solutions",
        type=int,
        default=None,
        help="Stop after this many solutions (default: find all)",
    )
    p.add_argument("-w", "--workers", type=int, default=None)
    p.add_argument(
        "-d",
        "--prefix-depth",
        type=int,
        default=3,
        help="Number of placements to enumerate before dispatching to workers",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for randomized search order (default: nondeterministic)",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Disable randomization (solutions will share long prefixes)",
    )
    args = p.parse_args()
    solve(
        n=args.size,
        output=args.output,
        max_solutions=args.max_solutions,
        workers=args.workers,
        prefix_depth=args.prefix_depth,
        seed=args.seed,
        randomize=not args.deterministic,
    )


if __name__ == "__main__":
    main()
