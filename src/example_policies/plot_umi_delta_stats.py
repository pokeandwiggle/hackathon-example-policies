"""Plot UMI-delta stepwise percentile statistics to PNG (stdlib-only).

Creates one horizontal row per action dimension, with overlaid curves for:
- p_low   (blue)
- p_high  (green)
- spread  (red)

Usage:
  python src/example_policies/plot_umi_delta_stats.py --stats /path/to/stepwise_percentile_stats.json
  python src/example_policies/plot_umi_delta_stats.py --stats /path/to/dataset_root --out umi_stats.png
"""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path

STEPWISE_STATS_FILENAME = "stepwise_percentile_stats.json"


def _resolve_stats_path(path: Path) -> Path:
    return path / STEPWISE_STATS_FILENAME if path.is_dir() else path


def _load_stats(path: Path) -> tuple[list[list[float]], list[list[float]]]:
    data = json.loads(path.read_text())
    if "p_low" not in data or "p_high" not in data:
        raise ValueError(f"Expected keys p_low/p_high in {path}")
    return data["p_low"], data["p_high"]


def _set_px(buf: bytearray, w: int, h: int, x: int, y: int, rgb: tuple[int, int, int]) -> None:
    if x < 0 or y < 0 or x >= w or y >= h:
        return
    i = (y * w + x) * 3
    buf[i : i + 3] = bytes(rgb)


def _draw_line(
    buf: bytearray,
    w: int,
    h: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    rgb: tuple[int, int, int],
) -> None:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        _set_px(buf, w, h, x0, y0, rgb)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _draw_rect(buf: bytearray, w: int, h: int, x: int, y: int, rw: int, rh: int, rgb: tuple[int, int, int]) -> None:
    _draw_line(buf, w, h, x, y, x + rw, y, rgb)
    _draw_line(buf, w, h, x, y + rh, x + rw, y + rh, rgb)
    _draw_line(buf, w, h, x, y, x, y + rh, rgb)
    _draw_line(buf, w, h, x + rw, y, x + rw, y + rh, rgb)


def _plot_series(
    buf: bytearray,
    w: int,
    h: int,
    x: int,
    y: int,
    pw: int,
    ph: int,
    values: list[float],
    vmin: float,
    vmax: float,
    rgb: tuple[int, int, int],
) -> None:
    n = len(values)
    if n < 2:
        return
    rng = max(vmax - vmin, 1e-9)

    prev_x = x
    prev_y = y + ph - int((values[0] - vmin) / rng * ph)
    for i in range(1, n):
        xi = x + int(i * pw / (n - 1))
        yi = y + ph - int((values[i] - vmin) / rng * ph)
        _draw_line(buf, w, h, prev_x, prev_y, xi, yi, rgb)
        prev_x, prev_y = xi, yi


def _save_png(path: Path, w: int, h: int, rgb: bytearray) -> None:
    raw = bytearray()
    stride = w * 3
    for y in range(h):
        raw.append(0)
        row = rgb[y * stride : (y + 1) * stride]
        raw.extend(row)

    compressed = zlib.compress(bytes(raw), level=9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data))
            + tag
            + data
            + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    ihdr = struct.pack("!IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png.extend(chunk(b"IHDR", ihdr))
    png.extend(chunk(b"IDAT", compressed))
    png.extend(chunk(b"IEND", b""))
    path.write_bytes(png)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot UMI-delta stepwise percentile stats to PNG")
    parser.add_argument("--stats", type=Path, required=True, help="Stats JSON path or dataset root")
    parser.add_argument("--out", type=Path, default=Path("umi_delta_stepwise_stats.png"), help="Output PNG path")
    args = parser.parse_args()

    stats_path = _resolve_stats_path(args.stats)
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    p_low, p_high = _load_stats(stats_path)
    if not p_low:
        raise ValueError("Empty p_low in stats file")

    action_dim = len(p_low[0])
    spread = [[hi - lo for lo, hi in zip(lo_row, hi_row)] for lo_row, hi_row in zip(p_low, p_high)]

    width = 1400
    margin = 16
    row_h = 58
    height = margin * 2 + action_dim * row_h
    plot_x = margin + 10
    plot_w = width - margin * 2 - 20

    img = bytearray([255] * (width * height * 3))

    for d in range(action_dim):
        y = margin + d * row_h
        ph = row_h - 14
        _draw_rect(img, width, height, plot_x, y + 4, plot_w, ph, (230, 230, 230))

        low = [row[d] for row in p_low]
        high = [row[d] for row in p_high]
        spr = [row[d] for row in spread]
        vals = low + high + spr
        vmin, vmax = min(vals), max(vals)

        _plot_series(img, width, height, plot_x + 1, y + 5, plot_w - 2, ph - 2, low, vmin, vmax, (31, 119, 180))
        _plot_series(img, width, height, plot_x + 1, y + 5, plot_w - 2, ph - 2, high, vmin, vmax, (44, 160, 44))
        _plot_series(img, width, height, plot_x + 1, y + 5, plot_w - 2, ph - 2, spr, vmin, vmax, (214, 39, 40))

    _save_png(args.out, width, height, img)
    print(f"Saved plot: {args.out}")


if __name__ == "__main__":
    main()
