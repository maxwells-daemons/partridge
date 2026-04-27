"""Interactive viewer for fractal Partridge Puzzle solutions.

Loads all solutions from a jsonl file. The square is tiled using one solution;
hovering highlights a tile; clicking a tile subdivides it with the *next*
solution and smoothly zooms the view so the clicked tile fills the window.

Instead of building an ever-deepening tree of Fractions, we keep only two
levels of state: the `current` tiling on screen, and — during a zoom — the
`next` tiling that will replace it. When the zoom completes, the clicked
tile's subdivision becomes the new current tiling in integer coordinates,
so no values ever shrink toward zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pygame
from distinctipy import distinctipy

from solver import side_length


TOL_MUTED = (
    (255, 173, 173),
    (255, 214, 165),
    (253, 255, 182),
    (202, 255, 191),
    (155, 246, 255),
    (160, 196, 255),
    (189, 178, 255),
    (255, 198, 255),
)


WINDOW_SIZE = 800
BG_COLOR = (20, 20, 24)
ZOOM_DURATION = 4  # seconds


# ---------- colors ----------


@lru_cache(maxsize=None)
def _palette(n: int) -> tuple[tuple[int, int, int], ...]:
    if n <= len(TOL_MUTED):
        return TOL_MUTED[:n]
    colors = distinctipy.get_colors(
        n,
        exclude_colors=[(0, 0, 0), (1, 1, 1), (20 / 255, 20 / 255, 24 / 255)],
        pastel_factor=0.2,
        rng=0,
    )
    return tuple((int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors)


def tile_color(unit_size: int, n: int) -> tuple[int, int, int]:
    return _palette(n)[unit_size - 1]


# ---------- zoom state ----------


@dataclass
class Anim:
    target_idx: int       # index into current_pl of the clicked tile
    next_pl: list[list[int]]  # solution that will replace current after zoom
    next_sol_idx: int
    start_t: float


def _ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def _pick_tile(pl: list[list[int]], wx: float, wy: float) -> int | None:
    for i, (k, r, c) in enumerate(pl):
        if c <= wx < c + k and r <= wy < r + k:
            return i
    return None


def _spiral_tile_order(pl: list[list[int]], side: int) -> list[list[int]]:
    """Return tiles in an outward-to-center spiral order.

    The spiral path walks the side x side grid from the outer boundary toward
    the center. A tile is added the first time the path encounters any of its
    cells, so the final tile in the returned list is the one covering the
    center of the board.
    """
    grid = [[-1] * side for _ in range(side)]
    for idx, (k, r, c) in enumerate(pl):
        for dr in range(k):
            for dc in range(k):
                grid[r + dr][c + dc] = idx

    order: list[int] = []
    seen: set[int] = set()
    top = 0
    bottom = side - 1
    left = 0
    right = side - 1

    while left <= right and top <= bottom:
        for x in range(left, right + 1):
            idx = grid[top][x]
            if idx not in seen:
                seen.add(idx)
                order.append(idx)
        top += 1

        for y in range(top, bottom + 1):
            idx = grid[y][right]
            if idx not in seen:
                seen.add(idx)
                order.append(idx)
        right -= 1

        if top <= bottom:
            for x in range(right, left - 1, -1):
                idx = grid[bottom][x]
                if idx not in seen:
                    seen.add(idx)
                    order.append(idx)
            bottom -= 1

        if left <= right:
            for y in range(bottom, top - 1, -1):
                idx = grid[y][left]
                if idx not in seen:
                    seen.add(idx)
                    order.append(idx)
            left += 1

    return [pl[idx] for idx in order]


# ---------- rendering ----------


def _draw_tile(
    screen: pygame.Surface,
    wx: float, wy: float, ws: float,
    vx: float, vy: float, vs: float,
    screen_scale: float,
    color: tuple[int, int, int],
    highlight: bool,
) -> None:
    if wx + ws <= vx or wx >= vx + vs or wy + ws <= vy or wy >= vy + vs:
        return
    x1 = round((wx - vx) * screen_scale)
    y1 = round((wy - vy) * screen_scale)
    x2 = round((wx + ws - vx) * screen_scale)
    y2 = round((wy + ws - vy) * screen_scale)
    rect = pygame.Rect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))
    pygame.draw.rect(screen, color, rect)
    pygame.draw.rect(screen, (0, 0, 0), rect, 1)
    if highlight:
        pygame.draw.rect(screen, (255, 255, 255), rect, 3)


def render(
    screen: pygame.Surface,
    current_pl: list[list[int]],
    anim: Anim | None,
    viewport: tuple[float, float, float],
    n: int,
    side: int,
    hover_idx: int | None,
    progress: float = 1.0,
    animated_tiles: bool = True,
    spiral_tiles: bool = False,
) -> None:
    vx, vy, vs = viewport
    screen_scale = WINDOW_SIZE / vs
    screen.fill(BG_COLOR)

    for idx, (k, r, c) in enumerate(current_pl):
        _draw_tile(
            screen, c, r, k, vx, vy, vs, screen_scale,
            tile_color(k, n),
            highlight=(anim is None and hover_idx == idx),
        )
        # During a zoom, overlay the next tiling inside the target tile. It
        # tiles the (c, r, k) cell exactly, so it fully covers the parent.
        if anim is not None and idx == anim.target_idx:
            sub_scale = k / side
            if animated_tiles:
                if spiral_tiles:
                    sorted_sub = _spiral_tile_order(anim.next_pl, side)
                else:
                    sorted_sub = sorted(anim.next_pl, key=lambda x: x[0], reverse=True)
                num_to_draw = int(progress * len(sorted_sub))
                for k2, r2, c2 in sorted_sub[:num_to_draw]:
                    _draw_tile(
                        screen,
                        c + c2 * sub_scale, r + r2 * sub_scale, k2 * sub_scale,
                        vx, vy, vs, screen_scale,
                        tile_color(k2, n),
                        highlight=False,
                    )
            else:
                for k2, r2, c2 in anim.next_pl:
                    _draw_tile(
                        screen,
                        c + c2 * sub_scale, r + r2 * sub_scale, k2 * sub_scale,
                        vx, vy, vs, screen_scale,
                        tile_color(k2, n),
                        highlight=False,
                    )


# ---------- main loop ----------


def run(
    solutions: list[list[list[int]]],
    n: int,
    animated_tiles: bool,
    spiral_tiles: bool,
) -> None:
    pygame.init()
    pygame.display.set_caption(f"Partridge N={n}")
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    clock = pygame.time.Clock()

    side = side_length(n)

    current_sol_idx = 0
    current_pl: list[list[int]] = solutions[0]
    anim: Anim | None = None
    hover_idx: int | None = None

    # Pre-rendered grid overlay: divides the window into side x side cells,
    # so the lines align with the current tiling's units at full zoom.
    grid_overlay = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
    for i in range(1, side):
        x = round(i * WINDOW_SIZE / side)
        pygame.draw.line(grid_overlay, (255, 255, 255, 55), (x, 0), (x, WINDOW_SIZE))
        pygame.draw.line(grid_overlay, (255, 255, 255, 55), (0, x), (WINDOW_SIZE, x))

    def start_zoom(target_idx: int) -> None:
        """Begin a zoom into current_pl[target_idx]. Cycles to the next
        solution for the subdivision."""
        nonlocal anim
        next_sol_idx = (current_sol_idx + 1) % len(solutions)
        anim = Anim(
            target_idx=target_idx,
            next_pl=solutions[next_sol_idx],
            next_sol_idx=next_sol_idx,
            start_t=pygame.time.get_ticks() / 1000.0,
        )

    def finish_zoom() -> None:
        """Renormalize: the clicked tile becomes the whole square again,
        its subdivision (anim.next_pl) becomes the new current tiling."""
        nonlocal current_pl, current_sol_idx, anim, hover_idx
        assert anim is not None
        current_pl = anim.next_pl
        current_sol_idx = anim.next_sol_idx
        anim = None
        hover_idx = None  # positions have changed; wait for next mousemove

    running = True
    while running:
        now = pygame.time.get_ticks() / 1000.0

        # Advance or complete the zoom, and compute the viewport for this frame.
        if anim is not None:
            t = (now - anim.start_t) / ZOOM_DURATION
            if t >= 1.0:
                finish_zoom()
                viewport = (0.0, 0.0, float(side))
                progress = 1.0
            else:
                e = _ease(t)
                k, r, c = current_pl[anim.target_idx]
                viewport = (c * e, r * e, side + (k - side) * e)
                progress = t
        else:
            viewport = (0.0, 0.0, float(side))
            progress = 1.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            elif event.type == pygame.MOUSEMOTION:
                if anim is None:
                    mx, my = event.pos
                    wx = viewport[0] + (mx / WINDOW_SIZE) * viewport[2]
                    wy = viewport[1] + (my / WINDOW_SIZE) * viewport[2]
                    hover_idx = _pick_tile(current_pl, wx, wy)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Clicks during a zoom: snap the in-flight animation to its end
                # (renormalize), then start a new zoom from the fresh state.
                if anim is not None:
                    finish_zoom()
                mx, my = event.pos
                wx = (mx / WINDOW_SIZE) * side
                wy = (my / WINDOW_SIZE) * side
                idx = _pick_tile(current_pl, wx, wy)
                if idx is not None:
                    start_zoom(idx)

        render(
            screen,
            current_pl,
            anim,
            viewport,
            n,
            side,
            hover_idx,
            progress,
            animated_tiles,
            spiral_tiles,
        )
        screen.blit(grid_overlay, (0, 0))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def load_solutions(path: Path) -> list[list[list[int]]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-n", "--size", type=int, default=8)
    p.add_argument("-s", "--solutions", default=None)
    p.add_argument(
        "--no-animated-tiles",
        dest="animated_tiles",
        action="store_false",
        default=True,
        help="Disable progressive tile appearance during zooms.",
    )
    p.add_argument(
        "--spiral-tiles",
        dest="spiral_tiles",
        action="store_true",
        default=False,
        help="Animate the next tiling's tiles in a spiral order toward the center.",
    )
    args = p.parse_args()
    if args.solutions is None:
        args.solutions = f"solutions_n{args.size}.jsonl"

    path = Path(args.solutions)
    if not path.exists():
        print(f"No solution file at {path}. Run solver.py first.", file=sys.stderr)
        sys.exit(1)

    sols = load_solutions(path)
    if len(sols) < 2:
        print(
            f"Need at least 2 solutions for 'next tiling' subdivision; have {len(sols)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    run(sols, args.size, args.animated_tiles, args.spiral_tiles)


if __name__ == "__main__":
    main()
