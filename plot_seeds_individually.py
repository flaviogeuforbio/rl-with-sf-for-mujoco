"""
Plot SF-DDPG / DDPG Walker2d results across all seeds, for both the from-scratch
and transfer-from-Cheetah conditions.

  1. Classifies each run as succeeded/failed using returns from the latter half of training (mean < 100 = failed).
  2. Shows every individual seed's curve, color-coded by outcome
  3. Averages ONLY across the successful seeds for the bold summary line, and
     labels that average explicitly as conditional on success.
  4. Reports the success counts in the console, and saves a separate bar chart of success rates. 


Edit RUN_DIRS below to point at your own folder layout, then run:
    python plot_seeds_honest.py
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


# --- WHERE DATA SIT ---
# This must point at the single folder that directly CONTAINS seed_1, seed_2, ...
# Windows note: backslashes (\) are Python's escape character, so either put an
# "r" right before the opening quote (a "raw string", shown below), or just use
# forward slashes instead -- Windows accepts both, and forward slashes need no
# escaping at all. Either of these two lines works identically:
#
#   BASE_DIR = Path(r"C:\Users\mauro\Documents\RLProject\rl-with-sf-for-mujoco\artifacts\completed_checkpoints")
#   BASE_DIR = Path("C:/Users/mauro/Documents/RLProject/rl-with-sf-for-mujoco/artifacts/completed_checkpoints")
BASE_DIR = Path(r"C:\Users\mauro\Documents\RLProject\rl-with-sf-for-mujoco\artifacts\completed_checkpoints")

# --- WHERE TO SAVE THE PLOTS ---
# Created automatically if it doesn't exist yet.
# Point this at a subfolder of "figures" folder
OUTPUT_DIR = Path(r"C:\Users\mauro\Documents\RLProject\rl-with-sf-for-mujoco\artifacts\figures\walker_seed_comparison")

SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
CONDITIONS = ["scratch", "transfer"]
ALGOS = [
    ("SF-DDPG", "sf_ddpg_returns.json"),
    ("DDPG", "ddpg_returns.json"),
]

# A run is classified as FAILED if its mean return over the second half of
# training is below this. Healthy runs in this project cluster at 600-1700;
# failed runs sit at ~0. 100 sits cleanly in the gap between them
FAIL_THRESHOLD = 100

SMOOTH_WINDOW = 150  # episode-return smoothing window for the plotted curves


def resolve_dir(seed: int, condition: str) -> Path:
    """Map (seed, condition) -> the directory containing its returns json files.

    The layout is the SAME shape for every seed and both conditions:
        BASE_DIR / seed_<N> / seed_<N>_<condition>_completed /
    e.g. BASE_DIR/seed_1/seed_1_scratch_completed/, BASE_DIR/seed_3/seed_3_transfer_completed/

    Because it's uniform across seeds, one line covers every case -- no
    per-seed special-casing needed. If you ever reorganize your folders,
    this is the only function you need to change; everything downstream
    just calls resolve_dir(seed, condition) and doesn't care how it's
    implemented internally.
    """
    return BASE_DIR / f"seed_{seed}" / f"seed_{seed}_{condition}_completed"


# ============================================================================
# LOADING
# ============================================================================

def load_walker_returns(seed: int, condition: str, json_name: str) -> np.ndarray:
    """Load the Walker-phase returns list for one run. Handles both scratch
    (Walker is the only/first phase in the json) and transfer (Walker is the
    second phase, after Cheetah) layouts automatically."""
    d = resolve_dir(seed, condition)
    # DEBUG
    # print(f"d = {d}")
    # END DEBUG 
    with open(d / json_name) as f:
        phases = json.load(f)
    # scratch: only ever has 1 phase (Walker). transfer: has 2 (Cheetah, Walker).
    return np.array(phases[-1])


def smooth(x: np.ndarray, w: int = SMOOTH_WINDOW) -> np.ndarray:
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def classify(returns: np.ndarray) -> tuple[bool, float]:
    """Returns (succeeded, latter_half_mean). Using the latter half rather than
    e.g. the last 500 episodes: individual episode returns here are extremely
    high-variance (std often exceeds the mean), so a short window can land on
    an unlucky/lucky patch of an otherwise-stable oscillation. Half the run is
    long enough to average that out while still reflecting late-training
    behavior rather than the initial ramp-up."""
    latter_half_mean = float(returns[len(returns) // 2:].mean())
    return latter_half_mean >= FAIL_THRESHOLD, latter_half_mean

#################### MAIN #######################
# ============================================================================
# LOAD + CLASSIFY EVERYTHING
# ============================================================================

data = {}          # (algo, condition, seed) -> smoothed return curve
outcomes = {}       # (algo, condition, seed) -> (succeeded: bool, latter_half_mean: float)

for algo_name, json_name in ALGOS:
    for condition in CONDITIONS:
        for seed in SEEDS:
            # DEBUG
            # print(f"Loading {algo_name} / {condition} / seed {seed} ...")
            # print(f"seed = {seed}, condition = {condition}, json_name = {json_name}")
            # END DEBUG
            raw = load_walker_returns(seed, condition, json_name)
            succeeded, lat_half = classify(raw)
            data[(algo_name, condition, seed)] = smooth(raw)
            outcomes[(algo_name, condition, seed)] = (succeeded, lat_half)

# ============================================================================
# PRINT A TEXT SUMMARY FIRST
# ============================================================================

print("=" * 78)
print("SUCCESS / FAILURE SUMMARY  (failed = latter-half mean return < "
      f"{FAIL_THRESHOLD})")
print("=" * 78)
for algo_name, _ in ALGOS:
    for condition in CONDITIONS:
        seeds_here = [(s, outcomes[(algo_name, condition, s)]) for s in SEEDS]
        n_ok = sum(1 for _, (ok, _) in seeds_here if ok)
        print(f"\n{algo_name} / {condition}: {n_ok}/{len(SEEDS)} succeeded")
        for s, (ok, val) in seeds_here:
            print(f"   seed {s}: {'OK      ' if ok else 'FAILED  '} "
                  f"latter-half mean return = {val:8.1f}")

total_ok = sum(1 for v in outcomes.values() if v[0])
print(f"\nOverall: {total_ok}/{len(outcomes)} runs succeeded "
      f"({len(outcomes) - total_ok} failed)")

# ============================================================================
# PLOT: 2x2 grid -- rows = algorithm, columns = condition
# Every seed shown individually (green=succeeded, red=failed, thin lines),
# plus a bold line = mean across SUCCEEDED seeds only, with SEM shading.
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharey=True)

for row, (algo_name, _) in enumerate(ALGOS):
    for col, condition in enumerate(CONDITIONS):
        ax = axes[row, col]
        succeeded_curves = []

        for seed in SEEDS:
            curve = data[(algo_name, condition, seed)]
            ok, lat_half = outcomes[(algo_name, condition, seed)]
            
            # Map the seed to a value between 0.0 and 1.0 for the colormap.
            # (seed - 1) / 9 gives [0.0, 0.11, 0.22, 0.33, 0.44, 0.56, 0.67, 0.78, 0.89, 1.0] for seeds 1-10.
            color_idx = (seed - 1) / max(1, len(SEEDS) - 1)
            
            if ok:
                color = plt.cm.winter(color_idx)  # Cold family (blue to green)
            else:
                color = plt.cm.autumn(color_idx)  # Warm family (red to yellow)
                
            alpha = 0.65 
            
            ax.plot(curve, color=color, alpha=alpha, linewidth=1,
                    label=f"seed {seed} ({'ok' if ok else 'FAILED'})")
            if ok:
                succeeded_curves.append(curve)

        # Mean-of-successes, only if at least 2 succeeded (a single curve isn't a mean)
        if len(succeeded_curves) >= 2:
            min_len = min(len(c) for c in succeeded_curves)
            stacked = np.array([c[:min_len] for c in succeeded_curves])
            mean = stacked.mean(axis=0)
            sem = stacked.std(axis=0) / np.sqrt(len(succeeded_curves))
            x = np.arange(min_len)
            ax.plot(x, mean, color="black", linewidth=2.2,
                    label=f"mean of {len(succeeded_curves)} successful seeds")
            ax.fill_between(x, mean - sem, mean + sem, color="black", alpha=0.15)
        elif len(succeeded_curves) == 1:
            ax.plot(succeeded_curves[0], color="black", linewidth=2.2,
                    label="only successful seed")

        n_ok = len(succeeded_curves)
        ax.set_title(f"{algo_name} - {condition}  ({n_ok}/{len(SEEDS)} seeds succeeded)",
                     fontweight="bold" if n_ok < len(SEEDS) else "normal")
        ax.set_xlabel("Episode (within Walker phase)")
        if col == 0:
            ax.set_ylabel(f"Return (smoothed, w={SMOOTH_WINDOW})")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)
        ax.set_xlim(left=0, right=15000)

fig.suptitle("Walker2d-v5: all 10 seeds, scratch vs. transfer-from-Cheetah, SF-DDPG vs. DDPG\n"
             "Cold colors = succeeded, warm colors = failed (collapsed to near-zero return) -- ",
             fontweight="bold", fontsize=12)
fig.tight_layout()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # creates the folder (and any missing parents) if needed

out_path_1 = OUTPUT_DIR / "all_seeds_comparison.pdf"
fig.savefig(out_path_1, dpi=150)
print(f"\nSaved: {out_path_1}")

# ============================================================================
# SECONDARY PLOT: clean summary bar chart of success rates
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(8, 5.5))
labels, rates = [], []
for algo_name, _ in ALGOS:
    for condition in CONDITIONS:
        n_ok = sum(1 for s in SEEDS if outcomes[(algo_name, condition, s)][0])
        labels.append(f"{algo_name}\n{condition}")
        rates.append(n_ok / len(SEEDS) * 100)

bars = ax2.bar(labels, rates, color=["navy", "blue", "darkorange", "orange"])
ax2.set_ylabel("Success rate (%)")
ax2.set_ylim(0, 100)
ax2.set_title("Success rate by algorithm x condition (out of 10 seeds each)", fontweight="bold")
for bar, rate, algo_cond in zip(bars, rates, labels):
    n_ok = round(rate / 10)
    ax2.text(bar.get_x() + bar.get_width() / 2, rate + 2, f"{n_ok}/10",
              ha="center", fontweight="bold")
ax2.grid(alpha=0.3, axis="y")
fig2.tight_layout()
out_path_2 = OUTPUT_DIR / "success_rate_summary.pdf"
fig2.savefig(out_path_2, dpi=150)
print(f"Saved: {out_path_2}")
