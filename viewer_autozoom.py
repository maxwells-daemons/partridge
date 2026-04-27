"""Auto-zooming variant of viewer.py.

Same rendering as viewer.py, but instead of waiting for mouse clicks, a random
tile in the current tiling is picked and zoomed into as soon as the previous
zoom animation completes. The next zoom's target and subdivision are chosen
*as soon as the previous zoom finishes* and rendered during the dwell period,
so the sub-tiling is already visible when the zoom starts — no "pop."

Controls:
  space  pause/resume
  q/esc  quit
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pygame

from solver import side_length
from viewer_interactive import (
    Anim,
    WINDOW_SIZE,
    load_solutions,
    render,
)


# Total zoom factor per cycle is `side` (we go from the whole square down to
# the 1x1 tile). At 1.2s the zoom is slower and easier to follow.
ZOOM_DURATION = 5.0


def _rot90(pl: list[list[int]], side: int) -> list[list[int]]:
    # 90° CW: a tile at (r, c) of size k moves to (c, side - r - k).
    return [[k, c, side - r - k] for k, r, c in pl]


def _flip_h(pl: list[list[int]], side: int) -> list[list[int]]:
    return [[k, r, side - c - k] for k, r, c in pl]


def expand_symmetries(sols: list[list[list[int]]], n: int) -> list[list[list[int]]]:
    """Return sols plus all rotations and horizontal flips, deduped. Gives up
    to 8x more tilings to cycle through without re-running the solver."""
    side = side_length(n)
    out: list[list[list[int]]] = []
    seen: set[tuple[tuple[int, int, int], ...]] = set()
    for pl in sols:
        v = [t[:] for t in pl]
        variants = [v]
        for _ in range(3):
            variants.append(_rot90(variants[-1], side))
        variants += [_flip_h(x, side) for x in list(variants)]
        for variant in variants:
            key = tuple(sorted((k, r, c) for k, r, c in variant))
            if key in seen:
                continue
            seen.add(key)
            out.append(variant)
    return out


# Tiny pause between zooms so successive cycles don't strictly blur together.
# The overlay is drawn during dwell, so pop-in is avoided either way.
DWELL_SECONDS = 0.05


def run(
    solutions: list[list[list[int]]],
    n: int,
    seed: int | None,
    animated_tiles: bool,
) -> None:
    pygame.init()
    pygame.display.set_caption(f"Partridge auto N={n}")
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    clock = pygame.time.Clock()

    side = side_length(n)
    rng = random.Random(seed)

    current_sol_idx = 0
    current_pl: list[list[int]] = solutions[0]

    def queue_next(now: float) -> Anim:
        """Pick the 1x1 tile as the next target and schedule the zoom to
        start after DWELL_SECONDS. Always zooming into the size-1 tile means
        we dive maximally each step — the subdivision we'll land on takes up
        1/side of the current frame, so the zoom feels substantial."""
        target_idx = next(i for i, (k, _, _) in enumerate(current_pl) if k == 1)
        next_sol_idx = rng.randrange(len(solutions))
        return Anim(
            target_idx=target_idx,
            next_pl=solutions[next_sol_idx],
            next_sol_idx=next_sol_idx,
            start_t=now + DWELL_SECONDS,
        )

    anim: Anim = queue_next(pygame.time.get_ticks() / 1000.0)

    grid_overlay = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
    for i in range(1, side):
        x = round(i * WINDOW_SIZE / side)
        pygame.draw.line(grid_overlay, (255, 255, 255, 55), (x, 0), (x, WINDOW_SIZE))
        pygame.draw.line(grid_overlay, (255, 255, 255, 55), (0, x), (WINDOW_SIZE, x))

    paused = False
    pause_began: float | None = None

    running = True
    while running:
        now = pygame.time.get_ticks() / 1000.0
        # Freeze time while paused by treating `now` as the instant pause began.
        effective_now = pause_began if paused else now
        assert effective_now is not None

        # t in [0, 1] is the zoom progress. Negative means we're still in the
        # dwell period — the overlay renders but the camera hasn't moved yet.
        t = (effective_now - anim.start_t) / ZOOM_DURATION

        if t >= 1.0:
            current_pl = anim.next_pl
            current_sol_idx = anim.next_sol_idx
            anim = queue_next(effective_now)
            viewport = (0.0, 0.0, float(side))
        else:
            t_clamped = max(0.0, t)
            k, r, c = current_pl[anim.target_idx]
            # Exponential on vs → constant perceived zoom rate (no speed-up
            # at the end). Crucially, pan is driven by the zoom's progress
            # u = (side - vs)/(side - k), not by t directly. This keeps the
            # viewport pinned to the canvas (vx + vs stays ≤ side the whole
            # way), so the target can't swing off-screen mid-animation.
            vs = side * (k / side) ** t_clamped
            u = (side - vs) / (side - k)
            viewport = (c * u, r * u, vs)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    if not paused:
                        paused = True
                        pause_began = now
                    else:
                        assert pause_began is not None
                        anim.start_t += now - pause_began
                        paused = False
                        pause_began = None

        render(
            screen,
            current_pl,
            anim,
            viewport,
            n,
            side,
            hover_idx=None,
            progress=t_clamped,
            animated_tiles=animated_tiles,
        )
        screen.blit(grid_overlay, (0, 0))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-n", "--size", type=int, default=8)
    p.add_argument("-s", "--solutions", default="solutions.jsonl")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--no-animated-tiles",
        dest="animated_tiles",
        action="store_false",
        default=True,
        help="Disable progressive tile appearance during zooms.",
    )
    args = p.parse_args()

    path = Path(args.solutions)
    if not path.exists():
        print(f"No solution file at {path}. Run solver.py first.", file=sys.stderr)
        sys.exit(1)

    sols = load_solutions(path)
    if not sols:
        print("No solutions loaded.", file=sys.stderr)
        sys.exit(1)
    sols = expand_symmetries(sols, args.size)
    print(f"Loaded {len(sols)} tilings (including symmetries)", file=sys.stderr)

    run(sols, args.size, args.seed, args.animated_tiles)


if __name__ == "__main__":
    main()
