"""Generate the two-panel gamma-sweep figure ``fig:gammaphase``.

Expected project layout (one file for every gamma/seed pair):

artifacts/
  eval_gamma_0_5_lq_0_2_lvec_1_0_stepsxphase_50000/
    seed_1/
      diagnostics/
        gamma_0_5__phase_0__mode_sf_action_optimization__model_sf__task_backward/
          rollout_dynamics_timeseries_def.json

The script reads the run-level aggregates already stored in each JSON; it does
not recompute the diagnostics from the per-step timeseries.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


GAMMAS = np.array([0.50, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99], dtype=float)
SEEDS = (1, 2, 3)

RUN_TEMPLATE = "eval_gamma_{gamma_token}_lq_0_2_lvec_1_0_stepsxphase_50000"
DIAGNOSTIC_TEMPLATE = (
    "gamma_{gamma_token}"
    "__phase_0"
    "__mode_sf_action_optimization"
    "__model_sf"
    "__task_backward"
)
DEFAULT_JSON_NAME = "rollout_dynamics_timeseries_def.json"

# JSON keys produced by diagnose_rollout_dynamics.py.
METRIC_KEYS = {
    "sustained_flip_rate": "sustained_flip_rate",
    "flip_fraction": "flip_fraction_mean",
    "sbi_mean": "spurious_basin_indicator_mean",
    "median_delta_q_post": "q_after_jump_norm_at_flip_median",
}


def gamma_token(gamma: float) -> str:
    """Convert 0.50 -> '0_5', 0.97 -> '0_97', etc."""
    return f"{gamma:g}".replace(".", "_")


def build_json_path(
    artifacts_dir: Path,
    gamma: float,
    seed: int,
    json_name: str,
) -> Path:
    """Build the exact path of one gamma/seed diagnostic JSON."""
    token = gamma_token(gamma)
    return (
        artifacts_dir
        / RUN_TEMPLATE.format(gamma_token=token)
        / f"seed_{seed}"
        / "diagnostics"
        / DIAGNOSTIC_TEMPLATE.format(gamma_token=token)
        / json_name
    )


def optional_float(value: object) -> float:
    """Convert a JSON number to float and None to NaN."""
    if value is None:
        return float("nan")
    return float(value)


def load_run_metrics(json_path: Path) -> dict[str, float]:
    """Extract the four run-level quantities required by fig:gammaphase."""
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    try:
        basin = data["spurious_basin_diagnostics"]
    except KeyError as exc:
        raise KeyError(
            f"Missing 'spurious_basin_diagnostics' in {json_path}"
        ) from exc

    metrics: dict[str, float] = {}
    for output_name, json_key in METRIC_KEYS.items():
        if json_key not in basin:
            raise KeyError(f"Missing key '{json_key}' in {json_path}")
        metrics[output_name] = optional_float(basin[json_key])

    return metrics


def collect_data(
    artifacts_dir: Path,
    json_name: str,
    strict: bool,
) -> list[dict[str, float | int | str]]:
    """Load all available gamma/seed files into one long-form table."""
    rows: list[dict[str, float | int | str]] = []
    missing_paths: list[Path] = []

    for gamma in GAMMAS:
        for seed in SEEDS:
            json_path = build_json_path(
                artifacts_dir=artifacts_dir,
                gamma=float(gamma),
                seed=seed,
                json_name=json_name,
            )

            if not json_path.is_file():
                missing_paths.append(json_path)
                continue

            metrics = load_run_metrics(json_path)
            rows.append(
                {
                    "gamma": float(gamma),
                    "seed": seed,
                    **metrics,
                    "source_file": str(json_path),
                }
            )

    if missing_paths:
        message = "\n".join(f"  - {path}" for path in missing_paths)
        if strict:
            raise FileNotFoundError(
                "Some expected diagnostic files are missing:\n" + message
            )
        print("WARNING: some expected files were not found and will be skipped:")
        print(message)

    if not rows:
        raise FileNotFoundError(
            f"No diagnostic JSON files found under: {artifacts_dir}"
        )

    return rows


def rows_to_matrix(
    rows: Iterable[dict[str, float | int | str]],
    metric: str,
) -> np.ndarray:
    """Return a [num_gamma, num_seed] matrix, using NaN for missing values."""
    matrix = np.full((len(GAMMAS), len(SEEDS)), np.nan, dtype=float)

    gamma_to_index = {float(gamma): index for index, gamma in enumerate(GAMMAS)}
    seed_to_index = {seed: index for index, seed in enumerate(SEEDS)}

    for row in rows:
        gamma_index = gamma_to_index[float(row["gamma"])]
        seed_index = seed_to_index[int(row["seed"])]
        matrix[gamma_index, seed_index] = float(row[metric])

    return matrix


def aggregate_across_seeds(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean, population std, and valid-seed count at every gamma."""
    means = np.full(matrix.shape[0], np.nan, dtype=float)
    stds = np.full(matrix.shape[0], np.nan, dtype=float)
    counts = np.zeros(matrix.shape[0], dtype=int)

    for gamma_index, values in enumerate(matrix):
        valid = values[np.isfinite(values)]
        counts[gamma_index] = len(valid)
        if len(valid) > 0:
            means[gamma_index] = float(np.mean(valid))
            # ddof=0 matches np.std used by the diagnostics/report pipeline.
            stds[gamma_index] = float(np.std(valid, ddof=0))

    return means, stds, counts


def scatter_seed_points(
    ax: plt.Axes,
    matrix: np.ndarray,
    base_x: np.ndarray,
    color: str,
    metric_offset: float,
    seed_markers: tuple[str, ...],
) -> None:
    """Show the three independent run-level points at every gamma."""
    seed_jitter = np.array([-0.055, 0.0, 0.055], dtype=float)

    for seed_index, marker in enumerate(seed_markers):
        y = matrix[:, seed_index]
        valid = np.isfinite(y)
        ax.scatter(
            base_x[valid] + metric_offset + seed_jitter[seed_index],
            y[valid],
            marker=marker,
            s=48,
            color=color,
            alpha=0.65,
            edgecolors="white",
            linewidths=0.6,
            zorder=4,
        )


def plot_mean_and_band(
    ax: plt.Axes,
    base_x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    label: str,
    color: str,
    marker: str,
    linestyle: str,
    offset: float,
    clip: tuple[float, float] | None = None,
) -> None:
    """Plot the across-seed mean curve and its ±1 std band."""
    x = base_x + offset
    lower = mean - std
    upper = mean + std

    if clip is not None:
        lower = np.clip(lower, clip[0], clip[1])
        upper = np.clip(upper, clip[0], clip[1])

    valid = np.isfinite(mean)
    ax.plot(
        x[valid],
        mean[valid],
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=2.2,
        markersize=6.5,
        label=label,
        zorder=5,
    )
    ax.fill_between(
        x,
        lower,
        upper,
        where=np.isfinite(lower) & np.isfinite(upper),
        color=color,
        alpha=0.16,
        linewidth=0,
        zorder=1,
    )


def create_figure(
    rows: list[dict[str, float | int | str]],
    output_dir: Path,
) -> None:
    """Create and save the two-panel headline gamma-sweep figure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sustained = rows_to_matrix(rows, "sustained_flip_rate")
    flip_fraction = rows_to_matrix(rows, "flip_fraction")
    sbi = rows_to_matrix(rows, "sbi_mean")
    delta_q = rows_to_matrix(rows, "median_delta_q_post")

    sustained_mean, sustained_std, _ = aggregate_across_seeds(sustained)
    flip_mean, flip_std, _ = aggregate_across_seeds(flip_fraction)
    sbi_mean, sbi_std, _ = aggregate_across_seeds(sbi)
    delta_mean, delta_std, delta_count = aggregate_across_seeds(delta_q)

    # Avoid implying that a single available delta-Q run has measured variance.
    delta_std_for_band = delta_std.copy()
    delta_std_for_band[delta_count < 2] = np.nan

    x = np.arange(len(GAMMAS), dtype=float)
    seed_markers = ("o", "^", "s")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))

    # ----------------------------
    # Panel (a): physical flipping
    # ----------------------------
    ax = axes[0]
    scatter_seed_points(ax, sustained, x, "tab:blue", -0.025, seed_markers)
    scatter_seed_points(ax, flip_fraction, x, "tab:orange", 0.025, seed_markers)

    plot_mean_and_band(
        ax,
        x,
        sustained_mean,
        sustained_std,
        label="Sustained flip rate",
        color="tab:blue",
        marker="o",
        linestyle="-",
        offset=-0.025,
        clip=(0.0, 1.0),
    )
    plot_mean_and_band(
        ax,
        x,
        flip_mean,
        flip_std,
        label="Flip fraction",
        color="tab:orange",
        marker="s",
        linestyle="--",
        offset=0.025,
        clip=(0.0, 1.0),
    )

    ax.set_title("(a) Postural instability")
    ax.set_xlabel(r"Discount factor $\gamma$")
    ax.set_ylabel("Fraction")
    ax.set_ylim(-0.04, 1.08)
    ax.set_xticks(x, [f"{gamma:.2f}" for gamma in GAMMAS])
    ax.grid(alpha=0.25)

    metric_handles, metric_labels = ax.get_legend_handles_labels()
    seed_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor="0.45",
            markeredgecolor="white",
            markersize=7,
            label=f"Seed {seed}",
        )
        for seed, marker in zip(SEEDS, seed_markers)
    ]
    ax.legend(
        metric_handles + seed_handles,
        metric_labels + [f"Seed {seed}" for seed in SEEDS],
        loc="upper left",
        fontsize=8.5,
        ncol=2,
        frameon=True,
    )

    # -------------------------------------------
    # Panel (b): critic's spurious-basin evidence
    # -------------------------------------------
    ax = axes[1]
    scatter_seed_points(ax, sbi, x, "tab:purple", -0.025, seed_markers)
    scatter_seed_points(ax, delta_q, x, "tab:green", 0.025, seed_markers)

    plot_mean_and_band(
        ax,
        x,
        sbi_mean,
        sbi_std,
        label="SBI mean",
        color="tab:purple",
        marker="o",
        linestyle="-",
        offset=-0.025,
    )
    plot_mean_and_band(
        ax,
        x,
        delta_mean,
        delta_std_for_band,
        label=r"Median $\Delta Q_{\mathrm{post}}$",
        color="tab:green",
        marker="s",
        linestyle="--",
        offset=0.025,
    )

    ax.axhline(0.0, color="0.25", linestyle=":", linewidth=1.1, zorder=0)
    ax.set_title("(b) Spurious-basin value signal")
    ax.set_xlabel(r"Discount factor $\gamma$")
    ax.set_ylabel("Dimensionless diagnostic value")
    ax.set_xticks(x, [f"{gamma:.2f}" for gamma in GAMMAS])
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, frameon=True)

    fig.suptitle(
        "Gamma sweep: instability saturates while the perceived basin value keeps growing",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    pdf_path = output_dir / "fig_gammaphase.pdf"
    png_path = output_dir / "fig_gammaphase.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {pdf_path}")
    print(f"Saved figure: {png_path}")


def save_source_csv(
    rows: list[dict[str, float | int | str]],
    output_dir: Path,
) -> None:
    """Save the exact values used for the figure for reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fig_gammaphase_source_data.csv"

    fieldnames = [
        "gamma",
        "seed",
        "sustained_flip_rate",
        "flip_fraction",
        "sbi_mean",
        "median_delta_q_post",
        "source_file",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved source data: {csv_path}")


def print_aggregated_values(
    rows: list[dict[str, float | int | str]],
) -> None:
    """Print the across-seed values so they can be checked against the report."""
    metric_names = [
        "sustained_flip_rate",
        "flip_fraction",
        "sbi_mean",
        "median_delta_q_post",
    ]

    matrices = {name: rows_to_matrix(rows, name) for name in metric_names}

    print("\nAcross-seed summary (mean ± population std):")
    print(
        "gamma | sustained flip rate | flip fraction | SBI mean | median DeltaQ_post"
    )
    print("-" * 88)

    for gamma_index, gamma in enumerate(GAMMAS):
        cells: list[str] = []
        for name in metric_names:
            values = matrices[name][gamma_index]
            valid = values[np.isfinite(values)]
            if len(valid) == 0:
                cells.append("N/A")
            elif len(valid) == 1:
                cells.append(f"{valid[0]:.3f}")
            else:
                cells.append(f"{np.mean(valid):.3f} ± {np.std(valid):.3f}")

        print(
            f"{gamma:.2f} | "
            + " | ".join(f"{cell:>18}" for cell in cells)
        )


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_artifacts = script_dir / "artifacts"

    parser = argparse.ArgumentParser(
        description="Generate fig:gammaphase from 7 gamma values x 3 seeds."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=default_artifacts,
        help=(
            "Path to the artifacts directory. Default: an 'artifacts' folder next "
            "to this script."
        ),
    )
    parser.add_argument(
        "--json-name",
        default=DEFAULT_JSON_NAME,
        help=f"Diagnostic JSON filename (default: {DEFAULT_JSON_NAME}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <artifacts-dir>/figures.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop immediately if any of the expected 21 JSON files is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else artifacts_dir / "figures"
    )

    rows = collect_data(
        artifacts_dir=artifacts_dir,
        json_name=args.json_name,
        strict=args.strict,
    )

    print(f"Loaded {len(rows)} diagnostic files (expected: {len(GAMMAS) * len(SEEDS)}).")
    print_aggregated_values(rows)
    save_source_csv(rows, output_dir)
    create_figure(rows, output_dir)


if __name__ == "__main__":
    main()
