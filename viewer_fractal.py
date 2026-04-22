"""Multi-level fractal viewer for Partridge Puzzle solutions.

Unlike viewer.py (which shows one tiling at a time and zooms on click), this
viewer subdivides *every* tile recursively to a fixed depth, using a different
solution at each level. The whole fractal is on screen at once.

Controls:
  +/=   increase depth
  -     decrease depth
  r     reshuffle which solution is used at each level
  q/esc quit
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import pygame
from distinctipy import distinctipy

from solver import side_length


WINDOW_SIZE = 800
BG_COLOR = (20, 20, 24)


# ---------- colors ----------
#
# Each leaf tile is colored by (depth, unit_size). The viewer shows all levels
# at once, so a tile's "unit_size" alone isn't enough to tell levels apart —
# a level-2 size-3 tile looks nothing like a level-0 size-3 tile visually.
# We generate one distinct color per (depth, size) pair.


@lru_cache(maxsize=None)
def _palette(total: int) -> tuple[tuple[int, int, int], ...]:
    colors = distinctipy.get_colors(
        total,
        exclude_colors=[(0, 0, 0), (1, 1, 1), (20 / 255, 20 / 255, 24 / 255)],
        pastel_factor=0.3,
        rng=0,
    )
    return tuple((int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors)


def tile_color(depth: int, unit_size: int, n: int, max_depth: int) -> tuple[int, int, int]:
    pal = _palette(max_depth * n)
    return pal[depth * n + (unit_size - 1)]


# ---------- fractal construction ----------


@dataclass
class Tile:
    x: Fraction
    y: Fraction
    size: Fraction
    unit_size: int
    depth: int
    children: list["Tile"] | None = None


def build_fractal(
    root_side: int,
    solutions: list[list[list[int]]],
    max_depth: int,
    n: int,
    seed: int,
) -> Tile:
    """Build a tree of depth `max_depth`. Every tile picks an independent
    random solution for its subdivision, so no two sibling cells share a
    tiling (unless the RNG happens to collide)."""
    side = side_length(n)
    root = Tile(
        x=Fraction(0), y=Fraction(0), size=Fraction(root_side),
        unit_size=side, depth=0,
    )
    rng = random.Random(seed)

    def recur(tile: Tile) -> None:
        if tile.depth >= max_depth:
            return
        placements = solutions[rng.randrange(len(solutions))]
        scale = tile.size / side
        tile.children = [
            Tile(
                x=tile.x + c * scale,
                y=tile.y + r * scale,
                size=k * scale,
                unit_size=k,
                depth=tile.depth + 1,
            )
            for k, r, c in placements
        ]
        for child in tile.children:
            recur(child)

    recur(root)
    return root


# ---------- rendering ----------


def draw_tree(
    screen: pygame.Surface,
    tile: Tile,
    scale: float,
    n: int,
    max_depth: int,
) -> None:
    if tile.children is None:
        x1 = round(float(tile.x) * scale)
        y1 = round(float(tile.y) * scale)
        x2 = round(float(tile.x + tile.size) * scale)
        y2 = round(float(tile.y + tile.size) * scale)
        rect = pygame.Rect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))
        color = tile_color(tile.depth - 1, tile.unit_size, n, max_depth)
        pygame.draw.rect(screen, color, rect)
        # Only draw edges if the tile is big enough that the border won't
        # swamp the fill — at depth 3+ tiles can be <4px wide.
        if rect.width >= 4 and rect.height >= 4:
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)
        return
    for c in tile.children:
        draw_tree(screen, c, scale, n, max_depth)


# ---------- main loop ----------


def run(solutions: list[list[list[int]]], n: int, initial_depth: int) -> None:
    pygame.init()
    pygame.display.set_caption(f"Partridge fractal N={n}")
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    clock = pygame.time.Clock()

    side = side_length(n)
    scale = WINDOW_SIZE / side

    max_depth = max(1, initial_depth)
    level_budget = 8
    seed = 0
    root = build_fractal(side, solutions, max_depth, n, seed)

    def rebuild() -> Tile:
        _palette(max_depth * n)
        return build_fractal(side, solutions, max_depth, n, seed)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    if max_depth < level_budget:
                        max_depth += 1
                        root = rebuild()
                elif event.key == pygame.K_MINUS:
                    if max_depth > 1:
                        max_depth -= 1
                        root = rebuild()
                elif event.key == pygame.K_r:
                    seed += 1
                    root = rebuild()

        screen.fill(BG_COLOR)
        draw_tree(screen, root, scale, n, max_depth)
        # hud = font.render(
        #     f"depth={max_depth}   (+/- change depth, r reshuffle, q quit)",
        #     True, (240, 240, 240), (0, 0, 0),
        # )
        # screen.blit(hud, (8, 8))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def load_solutions(path: Path) -> list[list[list[int]]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-n", "--size", type=int, default=8)
    p.add_argument("-s", "--solutions", default="solutions.jsonl")
    p.add_argument("-d", "--depth", type=int, default=2)
    args = p.parse_args()

    path = Path(args.solutions)
    if not path.exists():
        print(f"No solution file at {path}. Run solver.py first.", file=sys.stderr)
        sys.exit(1)

    sols = load_solutions(path)
    if not sols:
        print("No solutions loaded.", file=sys.stderr)
        sys.exit(1)

    run(sols, args.size, args.depth)


if __name__ == "__main__":
    main()
