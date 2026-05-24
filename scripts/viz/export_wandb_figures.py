#!/usr/bin/env python3
"""Export presentation-ready W&B figures from a local run directory."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
from matplotlib.ticker import FuncFormatter


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
KV_RE = re.compile(
    r"(?P<key>[A-Za-z0-9_./-]+)="
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
HORIZON_RE = re.compile(r"^evaluation/rollout/mse_horizon_(?P<horizon>\d+)$")
TABLE_STEP_RE = re.compile(r"_(?P<step>\d+)_")


@dataclass(frozen=True)
class Series:
    steps: np.ndarray
    values: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a local W&B run directory and export black-and-white figures "
            "for training MSE loss, teacher-forced MSE, and rollout MSE by horizon."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Local W&B run directory, e.g. wandb/run-20260518_112336-rjqub0vv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for figures. Defaults to assets/wandb_figures/<run_id>.",
    )
    parser.add_argument(
        "--loss-key",
        default=None,
        help="Metric to use for the training MSE plot. Defaults to l2_loss if present, otherwise raw_l2_loss.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=251,
        help="Moving-average window for the training MSE curve. Use 1 to disable smoothing.",
    )
    parser.add_argument(
        "--loss-ymax",
        type=float,
        default=0.7,
        help="Upper y-axis limit for the training MSE plot. Use 0 or a negative value to disable.",
    )
    parser.add_argument(
        "--loss-ymin",
        type=float,
        default=0.2,
        help="Lower y-axis limit for the training MSE plot.",
    )
    parser.add_argument(
        "--loss-scale",
        choices=["log", "linear"],
        default="linear",
        help="Y-axis scale for the training MSE plot.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
        help="File formats to write.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=320,
        help="DPI for raster outputs.",
    )
    return parser.parse_args()


def run_id_from_dir(run_dir: Path) -> str:
    name = run_dir.name.rstrip("/")
    if "-" in name:
        return name.split("-")[-1]
    return name


def files_dir_for(run_dir: Path) -> Path:
    return run_dir / "files" if (run_dir / "files").is_dir() else run_dir


def parse_output_log(log_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            clean = ANSI_RE.sub("", line)
            metrics = {match.group("key"): float(match.group("value")) for match in KV_RE.finditer(clean)}
            if "step" not in metrics:
                continue
            metrics["step"] = int(metrics["step"])
            rows.append(metrics)
    return rows


def series_for(rows: Iterable[dict[str, float]], key: str) -> Series:
    points = [(row["step"], row[key]) for row in rows if key in row]
    if not points:
        raise ValueError(f"Metric not found in output.log: {key}")
    steps, values = zip(*points)
    return Series(np.asarray(steps, dtype=float), np.asarray(values, dtype=float))


def choose_loss_key(rows: list[dict[str, float]], requested_key: str | None) -> str:
    if requested_key:
        return requested_key

    available = {key for row in rows for key in row}
    for key in ("l2_loss", "raw_l2_loss", "train/l2_loss", "loss/l2"):
        if key in available:
            return key

    candidates = sorted(key for key in available if key.endswith("l2_loss") and "/" not in key)
    if candidates:
        return candidates[0]
    raise ValueError("Could not find a training MSE/L2 metric. Pass --loss-key explicitly.")


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < 3:
        return values
    window = min(window, values.size)
    if window % 2 == 0:
        window -= 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def latest_rollout_from_log(rows: list[dict[str, float]]) -> tuple[int, list[tuple[int, float]]] | None:
    latest_step = None
    latest_points: list[tuple[int, float]] = []
    for row in rows:
        points: list[tuple[int, float]] = []
        for key, value in row.items():
            match = HORIZON_RE.match(key)
            if match:
                points.append((int(match.group("horizon")), value))
        if points:
            latest_step = int(row["step"])
            latest_points = sorted(points)
    if latest_step is None:
        return None
    return latest_step, latest_points


def table_step(path: Path) -> int:
    match = TABLE_STEP_RE.search(path.stem)
    return int(match.group("step")) if match else -1


def latest_rollout_from_table(files_dir: Path) -> tuple[int, list[tuple[int, float]]] | None:
    table_paths = sorted(
        files_dir.glob("media/table/evaluation/plots/rollout_horizon_mse_table_*.table.json"),
        key=table_step,
    )
    if not table_paths:
        return None

    path = table_paths[-1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    columns = payload.get("columns", [])
    try:
        step_idx = columns.index("step")
        value_idx = columns.index("lineVal")
    except ValueError as exc:
        raise ValueError(f"Unexpected W&B table format in {path}") from exc

    points = [(int(row[step_idx]), float(row[value_idx])) for row in payload.get("data", [])]
    return table_step(path), sorted(points)


def style_plots() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#111111",
            "axes.linewidth": 1.0,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "grid.color": "#d7d7d7",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def format_step(value: float, _pos: int) -> str:
    if abs(value) >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def clean_axes(ax: plt.Axes) -> None:
    ax.grid(True, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=4, width=0.9)
    ax.xaxis.set_major_formatter(FuncFormatter(format_step))
    ax.margins(x=0.015)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str], dpi: int) -> list[Path]:
    output_paths = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi if fmt == "png" else None)
        output_paths.append(path)
    plt.close(fig)
    return output_paths


def plot_loss(
    series: Series,
    output_dir: Path,
    smooth_window: int,
    loss_ymin: float,
    loss_ymax: float,
    loss_scale: str,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    smoothed = moving_average(series.values, smooth_window)

    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    if smooth_window > 1:
        ax.plot(series.steps, series.values, color="#bdbdbd", linewidth=0.55, alpha=0.45, label="Raw")
        ax.plot(series.steps, smoothed, color="#111111", linewidth=2.0, label="Smoothed")
        ax.legend(loc="upper right")
    else:
        ax.plot(series.steps, series.values, color="#111111", linewidth=1.5)
    ax.set_title("Training: MSE Loss", loc="left", fontweight="semibold", pad=10)
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE")
    if loss_scale == "log":
        ax.set_yscale("log")
        visible_values = np.concatenate([series.values, smoothed])
        visible_values = visible_values[visible_values > 0]
        if loss_ymax > 0:
            under_limit = visible_values[visible_values <= loss_ymax]
            visible_values = under_limit if under_limit.size else visible_values
        ymin = loss_ymin if loss_ymin > 0 else float(np.nanmin(visible_values)) * 0.9
        ymax = loss_ymax if loss_ymax > 0 else float(np.nanmax(visible_values)) * 1.1
        if ymin >= ymax:
            ymin = ymax / 100.0
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
    elif loss_ymax > 0:
        ax.set_ylim(loss_ymin, loss_ymax)
    clean_axes(ax)
    return save_figure(fig, output_dir, "total_mse_loss", formats, dpi)


def plot_teacher_forced(series: Series, output_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    ax.plot(
        series.steps,
        series.values,
        color="#111111",
        linewidth=1.8,
        marker="o",
        markersize=3.0,
        markerfacecolor="white",
        markeredgewidth=1.0,
    )
    ax.set_title("Evaluation: Teacher-Forced MSE", loc="left", fontweight="semibold", pad=10)
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE")
    clean_axes(ax)
    return save_figure(fig, output_dir, "teacher_forced_final_mse", formats, dpi)


def plot_rollout(
    rollout_step: int,
    rollout_points: list[tuple[int, float]],
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    horizons = np.asarray([point[0] for point in rollout_points], dtype=float)
    values = np.asarray([point[1] for point in rollout_points], dtype=float)

    fig, ax = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)
    ax.plot(
        horizons,
        values,
        color="#111111",
        linewidth=2.0,
        marker="o",
        markersize=4.5,
        markerfacecolor="white",
        markeredgewidth=1.1,
    )
    ax.set_title("Evaluation: Rollout MSE", loc="left", fontweight="semibold", pad=10)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("MSE")
    ax.set_xticks(horizons)
    clean_axes(ax)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(value)}"))
    ax.text(
        0.99,
        0.02,
        f"step {format_step(rollout_step, 0)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#555555",
        fontsize=9,
    )
    return save_figure(fig, output_dir, "rollout_mse_by_horizon", formats, dpi)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    files_dir = files_dir_for(run_dir)
    log_path = files_dir / "output.log"
    if not log_path.is_file():
        raise FileNotFoundError(f"Could not find output.log under: {files_dir}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("assets") / "wandb_figures" / run_id_from_dir(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = parse_output_log(log_path)
    if not rows:
        raise ValueError(f"No metric rows found in: {log_path}")

    style_plots()

    loss_key = choose_loss_key(rows, args.loss_key)
    written: list[Path] = []
    written.extend(
        plot_loss(
            series_for(rows, loss_key),
            output_dir,
            args.smooth,
            args.loss_ymin,
            args.loss_ymax,
            args.loss_scale,
            args.formats,
            args.dpi,
        )
    )
    written.extend(
        plot_teacher_forced(
            series_for(rows, "evaluation/teacher_forced/final_mse"),
            output_dir,
            args.formats,
            args.dpi,
        )
    )

    rollout = latest_rollout_from_log(rows) or latest_rollout_from_table(files_dir)
    if rollout is None:
        raise ValueError("Could not find rollout horizon MSE metrics in output.log or W&B table files.")
    rollout_step, rollout_points = rollout
    written.extend(plot_rollout(rollout_step, rollout_points, output_dir, args.formats, args.dpi))

    print(f"Read {len(rows):,} metric rows from {log_path}")
    print(f"Using loss metric: {loss_key}")
    print("Wrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
