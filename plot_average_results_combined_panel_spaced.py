import os
import json
import numpy as np
import matplotlib.pyplot as plt

def _seed_sort_key(name):
    """Sort seed_2 before seed_10."""
    try:
        return int(name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return name


def load_data(base_path, algo_name, run_kind=None):
    """
    Load Phase 0 and Phase 1 returns across seeds.

    Supported directory layouts
    ---------------------------
    Old/direct layout:
        base_path/
            seed_1/
                sf_ddpg_returns.json
                ddpg_returns.json

    New completed-checkpoint layout:
        base_path/
            seed_1/
                seed_1_transfer_completed/
                    sf_ddpg_returns.json
                    ddpg_returns.json
                seed_1_scratch_completed/
                    sf_ddpg_returns.json
                    ddpg_returns.json

    Parameters
    ----------
    base_path:
        Root directory containing the outer seed_* folders.
    algo_name:
        "sf_ddpg" or "ddpg".
    run_kind:
        None for the old/direct layout, or "transfer"/"scratch" for the
        nested completed-checkpoint layout.

    The function also falls back to the old/direct layout when run_kind is
    supplied, so older experiment folders remain supported.
    """
    base_path = os.path.normpath(base_path)

    if not os.path.isdir(base_path):
        raise FileNotFoundError(
            f"Data root directory not found:\n{base_path}"
        )

    all_seeds_phase_0 = []
    all_seeds_phase_1 = []

    seed_names = sorted(
        (
            name
            for name in os.listdir(base_path)
            if name.startswith("seed_")
            and os.path.isdir(os.path.join(base_path, name))
        ),
        key=_seed_sort_key,
    )

    if not seed_names:
        raise RuntimeError(
            "No outer seed_* directories found in:\n"
            f"{base_path}"
        )

    loaded_paths = []

    for seed_name in seed_names:
        outer_seed_dir = os.path.join(base_path, seed_name)

        candidate_run_dirs = []

        if run_kind is not None:
            if run_kind not in {"transfer", "scratch"}:
                raise ValueError(
                    "run_kind must be None, 'transfer', or 'scratch'."
                )

            candidate_run_dirs.append(
                os.path.join(
                    outer_seed_dir,
                    f"{seed_name}_{run_kind}_completed",
                )
            )

        # Backward-compatible fallback for the old directory structure.
        candidate_run_dirs.append(outer_seed_dir)

        json_path = None

        for candidate_dir in candidate_run_dirs:
            candidate_json = os.path.join(
                candidate_dir,
                f"{algo_name}_returns.json",
            )
            if os.path.isfile(candidate_json):
                json_path = candidate_json
                break

        if json_path is None:
            expected = "\n  - ".join(
                os.path.join(
                    candidate,
                    f"{algo_name}_returns.json",
                )
                for candidate in candidate_run_dirs
            )
            print(
                f"Warning: no {algo_name} returns found for {seed_name}.\n"
                f"Expected one of:\n  - {expected}"
            )
            continue

        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if (
            not isinstance(data, list)
            or len(data) == 0
            or not isinstance(data[0], list)
        ):
            raise ValueError(
                "Unexpected returns JSON structure in:\n"
                f"{json_path}\n"
                "Expected [[phase_0_returns], [phase_1_returns]] "
                "or [[single_phase_returns]]."
            )

        all_seeds_phase_0.append(data[0])

        if len(data) > 1 and isinstance(data[1], list):
            all_seeds_phase_1.append(data[1])

        loaded_paths.append(json_path)

    print(
        f"Loaded {len(loaded_paths)} {algo_name} file(s)"
        + (
            f" for run_kind='{run_kind}'"
            if run_kind is not None
            else ""
        )
        + f" from:\n{base_path}"
    )

    return all_seeds_phase_0, all_seeds_phase_1

def smooth_and_pad(data_list, window=20): # Applies a moving average to each run and pads them to the same length based on the shortest run (even if they should all be the same length, we want to be safe and ensure they are aligned for averaging)
    if not data_list: return np.array([]) # Return an empty array if the input list is empty
    smoothed_data = [np.convolve(run, np.ones(window)/window, mode='valid') for run in data_list] # Apply moving average to each run (convolution with a uniform kernel 1/N where N is the window size = average over the window)
    min_len = min(len(run) for run in smoothed_data) # Find the minimum length among the smoothed runs to ensure we can pad them to the same length
    return np.array([run[:min_len] for run in smoothed_data]) # Pad each smoothed run to the minimum length by slicing (this ensures all runs are the same length for averaging)

def plot_curve(
    ax,
    data,
    label,
    color,
    linestyle="-",
    x_start=0,
):
    """Plot mean learning curve and mean ± SEM across seeds."""

    if data.size == 0:
        print(f"No data available for: {label}")
        return

    n_seeds = data.shape[0]

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # Standard Error of the Mean.
    sem = std / np.sqrt(n_seeds)
    margin = sem

    # x_start allows the x-axis to account for the smoothing window.
    x = np.arange(x_start, x_start + len(mean))

    ax.plot(
        x,
        mean,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=2,
    )

    ax.fill_between(
        x,
        mean - margin,
        mean + margin,
        color=color,
        alpha=0.15,
    )

def generate_transfer_comparison_plot(seq_dir, scratch_dir, gamma, main_title, window_size=20):
    """Generates the comparison plot and returns the Matplotlib figure and axes objects."""
    
    sf_seq_p0, sf_seq_p1 = load_data(seq_dir, "sf_ddpg", run_kind="transfer") # Load Phase 0 and Phase 1 returns for SF-DDPG from the sequential training directory
    base_seq_p0, base_seq_p1 = load_data(seq_dir, "ddpg", run_kind="transfer") # Load Phase 0 and Phase 1 returns for Standard DDPG from the sequential training directory
    
    sf_scratch_p0, _ = load_data(scratch_dir, "sf_ddpg", run_kind="scratch") # Load Phase 0 returns for SF-DDPG from the scratch training directory (we only care about Phase 0 for the scratch runs since they are trained from scratch on Task 2)
    base_scratch_p0, _ = load_data(scratch_dir, "ddpg", run_kind="scratch") # Load Phase 0 returns for Standard DDPG from the scratch training directory (same reason as above)
    
    sf_seq_p0_c = smooth_and_pad(sf_seq_p0, window_size)
    sf_seq_p1_c = smooth_and_pad(sf_seq_p1, window_size)
    base_seq_p0_c = smooth_and_pad(base_seq_p0, window_size)
    base_seq_p1_c = smooth_and_pad(base_seq_p1, window_size)
    
    sf_scratch_c = smooth_and_pad(sf_scratch_p0, window_size)
    base_scratch_c = smooth_and_pad(base_scratch_p0, window_size)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # Add the overall figure title
    fig.suptitle(main_title, fontsize=16, fontweight='bold')
    
    plot_curve(axes[0], sf_seq_p0_c, "SF-DDPG", "blue")
    plot_curve(axes[0], base_seq_p0_c, "Standard DDPG", "orange")
    axes[0].set_title(f"Task 1, $\\gamma$ = {gamma}")
    axes[0].set_xlabel("Episodes")
    axes[0].set_ylabel("Return (Smoothed)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    plot_curve(axes[1], sf_seq_p1_c, "SF-DDPG (After Task 1)", "blue")
    plot_curve(axes[1], base_seq_p1_c, "Standard DDPG (After Task 1)", "orange")
    
    plot_curve(axes[1], sf_scratch_c, "SF-DDPG (From Scratch)", "blue", linestyle='--')
    plot_curve(axes[1], base_scratch_c, "Standard DDPG (From Scratch)", "orange", linestyle='--')
    
    axes[1].set_title(f"Task 2 - Transfer vs Scratch, $\\gamma$ = {gamma}")
    axes[1].set_xlabel("Episodes")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    return fig, axes


def generate_gamma_sweep_plot(run_dir, gamma_label, main_title, window_size=20):
    """Generates the comparison plot for a single gamma sweep run."""
    
    # 1. Load data 
    sf_p0, sf_p1 = load_data(run_dir, "sf_ddpg") 
    base_p0, base_p1 = load_data(run_dir, "ddpg")

    # 2. Smooth data (Removed all scratch_c variables)
    sf_p0_c = smooth_and_pad(sf_p0, window_size)
    sf_p1_c = smooth_and_pad(sf_p1, window_size)
    base_p0_c = smooth_and_pad(base_p0, window_size)
    base_p1_c = smooth_and_pad(base_p1, window_size)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    # Add the overall figure title
    fig.suptitle(main_title, fontsize=16, fontweight='bold')

    # --- Phase 0 (Task 1) ---
    plot_curve(axes[0], sf_p0_c, f"SF-DDPG ($\\gamma$ = {gamma_label})", "blue")
    plot_curve(axes[0], base_p0_c, f"Baseline DDPG ($\\gamma$ = {gamma_label})", "orange")
    axes[0].set_title("Task 1")
    axes[0].set_xlabel("Episodes")
    axes[0].set_ylabel("Return (Smoothed)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # --- Phase 1 (Task 2) ---
    plot_curve(axes[1], sf_p1_c, f"SF-DDPG ($\\gamma$ = {gamma_label})", "blue")
    plot_curve(axes[1], base_p1_c, f"Baseline DDPG ($\\gamma$ = {gamma_label})", "orange")
    axes[1].set_title("Task 2 - Transfer Learning vs Scratch")
    axes[1].set_xlabel("Episodes")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    return fig, axes

def generate_cheetah_forward_plot(
    seq_dir,
    gamma,
    main_title="Phase 0: HalfCheetah Forward Training",
    window_size=20,
):
    """
    Plot Phase 0 HalfCheetah forward returns for SF-DDPG and standard DDPG.

    seq_dir must contain:
        seed_1/
            sf_ddpg_returns.json
            ddpg_returns.json
        seed_2/
            ...
    """

    # data[0] is Phase 0: HalfCheetah forward.
    sf_cheetah_returns, _ = load_data(seq_dir, "sf_ddpg", run_kind="transfer")
    ddpg_cheetah_returns, _ = load_data(seq_dir, "ddpg", run_kind="transfer")

    print(f"SF-DDPG seeds loaded: {len(sf_cheetah_returns)}")
    print(f"Standard DDPG seeds loaded: {len(ddpg_cheetah_returns)}")

    if not sf_cheetah_returns:
        raise RuntimeError(
            f"No SF-DDPG Phase 0 data found in:\n{seq_dir}"
        )

    if not ddpg_cheetah_returns:
        raise RuntimeError(
            f"No standard DDPG Phase 0 data found in:\n{seq_dir}"
        )

    # Smooth each seed independently and align seeds within each architecture.
    sf_cheetah_smoothed = smooth_and_pad(
        sf_cheetah_returns,
        window=window_size,
    )

    ddpg_cheetah_smoothed = smooth_and_pad(
        ddpg_cheetah_returns,
        window=window_size,
    )

    if sf_cheetah_smoothed.size == 0:
        raise RuntimeError("SF-DDPG smoothed data are empty.")

    if ddpg_cheetah_smoothed.size == 0:
        raise RuntimeError("DDPG smoothed data are empty.")

    # Use the same episode interval for both architectures.
    common_length = min(
        sf_cheetah_smoothed.shape[1],
        ddpg_cheetah_smoothed.shape[1],
    )

    sf_cheetah_smoothed = sf_cheetah_smoothed[:, :common_length]
    ddpg_cheetah_smoothed = ddpg_cheetah_smoothed[:, :common_length]

    fig, ax = plt.subplots(figsize=(10, 6))

    # The first valid moving average represents episodes 1...window_size.
    plot_curve(
        ax,
        sf_cheetah_smoothed,
        label="SF-DDPG",
        color="blue",
        x_start=window_size,
    )

    plot_curve(
        ax,
        ddpg_cheetah_smoothed,
        label="Standard DDPG",
        color="orange",
        x_start=window_size,
    )

    ax.set_title(
        f"{main_title}, $\\gamma$ = {gamma}",
        fontsize=14,
    )

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Return (Smoothed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    return fig, ax


def generate_walker_transfer_plot(
    seq_dir,
    scratch_dir,
    gamma,
    main_title="Phase 1: Walker2d Forward — Transfer vs Scratch",
    window_size=20,
):
    """
    Plot Walker2d Phase 1 learning curves after HalfCheetah pretraining and
    compare them with Walker2d training from scratch.

    Sequential/transfer directory:
        data[0] -> HalfCheetah Phase 0
        data[1] -> Walker2d Phase 1

    Walker-from-scratch directory:
        data[0] -> Walker2d, because it is the only executed phase.
    """

    # Transfer runs: Walker2d is the second phase, therefore data[1].
    _, sf_transfer_returns = load_data(seq_dir, "sf_ddpg", run_kind="transfer")
    _, ddpg_transfer_returns = load_data(seq_dir, "ddpg", run_kind="transfer")

    # Scratch runs: Walker2d is the only phase, therefore data[0].
    sf_scratch_returns, _ = load_data(scratch_dir, "sf_ddpg", run_kind="scratch")
    ddpg_scratch_returns, _ = load_data(scratch_dir, "ddpg", run_kind="scratch")

    print(f"SF-DDPG transfer seeds loaded: {len(sf_transfer_returns)}")
    print(f"Standard DDPG transfer seeds loaded: {len(ddpg_transfer_returns)}")
    print(f"SF-DDPG scratch seeds loaded: {len(sf_scratch_returns)}")
    print(f"Standard DDPG scratch seeds loaded: {len(ddpg_scratch_returns)}")

    missing = []
    if not sf_transfer_returns:
        missing.append("SF-DDPG transfer, Phase 1")
    if not ddpg_transfer_returns:
        missing.append("Standard DDPG transfer, Phase 1")
    if not sf_scratch_returns:
        missing.append("SF-DDPG from scratch")
    if not ddpg_scratch_returns:
        missing.append("Standard DDPG from scratch")

    if missing:
        raise RuntimeError(
            "Missing Walker2d data: " + ", ".join(missing) + "\n\n"
            f"Sequential directory: {seq_dir}\n"
            f"Scratch directory: {scratch_dir}"
        )

    sf_transfer_smoothed = smooth_and_pad(
        sf_transfer_returns,
        window=window_size,
    )
    ddpg_transfer_smoothed = smooth_and_pad(
        ddpg_transfer_returns,
        window=window_size,
    )
    sf_scratch_smoothed = smooth_and_pad(
        sf_scratch_returns,
        window=window_size,
    )
    ddpg_scratch_smoothed = smooth_and_pad(
        ddpg_scratch_returns,
        window=window_size,
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # Solid lines: Walker2d after HalfCheetah pretraining.
    plot_curve(
        ax,
        sf_transfer_smoothed,
        label="SF-DDPG (After HalfCheetah)",
        color="blue",
        linestyle="-",
        x_start=window_size,
    )
    plot_curve(
        ax,
        ddpg_transfer_smoothed,
        label="Standard DDPG (After HalfCheetah)",
        color="orange",
        linestyle="-",
        x_start=window_size,
    )

    # Dashed lines: Walker2d trained from scratch.
    plot_curve(
        ax,
        sf_scratch_smoothed,
        label="SF-DDPG (From Scratch)",
        color="blue",
        linestyle="--",
        x_start=window_size,
    )
    plot_curve(
        ax,
        ddpg_scratch_smoothed,
        label="Standard DDPG (From Scratch)",
        color="orange",
        linestyle="--",
        x_start=window_size,
    )

    ax.set_title(
        f"{main_title}, $\\gamma$ = {gamma}",
        fontsize=14,
    )
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Return (Smoothed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    return fig, ax


def generate_cheetah_walker_combined_plot(
    seq_dir,
    scratch_dir,
    gamma,
    main_title="HalfCheetah to Walker2d Transfer",
    window_size=20,
):
    """
    Create one two-panel figure.

    Left panel:
        Phase 0 HalfCheetah forward training for SF-DDPG and standard DDPG.

    Right panel:
        Phase 1 Walker2d training after HalfCheetah pretraining, compared
        with Walker2d training from scratch.

    Sequential/transfer JSON structure:
        data[0] -> HalfCheetah Phase 0
        data[1] -> Walker2d Phase 1

    Walker-from-scratch JSON structure:
        data[0] -> Walker2d, because it is the only executed phase.
    """

    # Sequential run: Phase 0 is HalfCheetah and Phase 1 is Walker2d.
    sf_cheetah_returns, sf_walker_transfer_returns = load_data(
        seq_dir,
        "sf_ddpg",
        run_kind="transfer",
    )
    ddpg_cheetah_returns, ddpg_walker_transfer_returns = load_data(
        seq_dir,
        "ddpg",
        run_kind="transfer",
    )

    # From-scratch run: Walker2d is stored in data[0].
    sf_walker_scratch_returns, _ = load_data(
        scratch_dir,
        "sf_ddpg",
        run_kind="scratch",
    )
    ddpg_walker_scratch_returns, _ = load_data(
        scratch_dir,
        "ddpg",
        run_kind="scratch",
    )

    print("\nCombined-panel data:")
    print(f"  SF-DDPG HalfCheetah seeds: {len(sf_cheetah_returns)}")
    print(f"  DDPG HalfCheetah seeds: {len(ddpg_cheetah_returns)}")
    print(
        "  SF-DDPG Walker transfer seeds: "
        f"{len(sf_walker_transfer_returns)}"
    )
    print(
        "  DDPG Walker transfer seeds: "
        f"{len(ddpg_walker_transfer_returns)}"
    )
    print(
        "  SF-DDPG Walker scratch seeds: "
        f"{len(sf_walker_scratch_returns)}"
    )
    print(
        "  DDPG Walker scratch seeds: "
        f"{len(ddpg_walker_scratch_returns)}"
    )

    missing = []

    if not sf_cheetah_returns:
        missing.append("SF-DDPG HalfCheetah Phase 0")
    if not ddpg_cheetah_returns:
        missing.append("Standard DDPG HalfCheetah Phase 0")
    if not sf_walker_transfer_returns:
        missing.append("SF-DDPG Walker2d transfer Phase 1")
    if not ddpg_walker_transfer_returns:
        missing.append("Standard DDPG Walker2d transfer Phase 1")
    if not sf_walker_scratch_returns:
        missing.append("SF-DDPG Walker2d from scratch")
    if not ddpg_walker_scratch_returns:
        missing.append("Standard DDPG Walker2d from scratch")

    if missing:
        raise RuntimeError(
            "Missing data required for the combined plot:\n  - "
            + "\n  - ".join(missing)
            + "\n\n"
            + f"Sequential directory: {seq_dir}\n"
            + f"Scratch directory: {scratch_dir}"
        )

    # Smooth every seed independently.
    sf_cheetah_smoothed = smooth_and_pad(
        sf_cheetah_returns,
        window=window_size,
    )
    ddpg_cheetah_smoothed = smooth_and_pad(
        ddpg_cheetah_returns,
        window=window_size,
    )

    sf_walker_transfer_smoothed = smooth_and_pad(
        sf_walker_transfer_returns,
        window=window_size,
    )
    ddpg_walker_transfer_smoothed = smooth_and_pad(
        ddpg_walker_transfer_returns,
        window=window_size,
    )
    sf_walker_scratch_smoothed = smooth_and_pad(
        sf_walker_scratch_returns,
        window=window_size,
    )
    ddpg_walker_scratch_smoothed = smooth_and_pad(
        ddpg_walker_scratch_returns,
        window=window_size,
    )

    # Use the same horizontal interval for both Cheetah architectures.
    cheetah_common_length = min(
        sf_cheetah_smoothed.shape[1],
        ddpg_cheetah_smoothed.shape[1],
    )
    sf_cheetah_smoothed = sf_cheetah_smoothed[:, :cheetah_common_length]
    ddpg_cheetah_smoothed = ddpg_cheetah_smoothed[
        :,
        :cheetah_common_length,
    ]

    # Use the same horizontal interval for all Walker2d curves.
    walker_common_length = min(
        sf_walker_transfer_smoothed.shape[1],
        ddpg_walker_transfer_smoothed.shape[1],
        sf_walker_scratch_smoothed.shape[1],
        ddpg_walker_scratch_smoothed.shape[1],
    )
    sf_walker_transfer_smoothed = sf_walker_transfer_smoothed[
        :,
        :walker_common_length,
    ]
    ddpg_walker_transfer_smoothed = ddpg_walker_transfer_smoothed[
        :,
        :walker_common_length,
    ]
    sf_walker_scratch_smoothed = sf_walker_scratch_smoothed[
        :,
        :walker_common_length,
    ]
    ddpg_walker_scratch_smoothed = ddpg_walker_scratch_smoothed[
        :,
        :walker_common_length,
    ]

    # One figure with two side-by-side panels and independent y-axes.
    # This lets HalfCheetah and Walker2d use different return scales.
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(17, 6),
        sharey=False,
    )

    if main_title:
        fig.suptitle(
            main_title,
            fontsize=16,
            fontweight="bold",
        )

    # ---------------------------------------------------------
    # Left panel: HalfCheetah Phase 0
    # ---------------------------------------------------------
    plot_curve(
        axes[0],
        sf_cheetah_smoothed,
        label="SF-DDPG",
        color="blue",
        x_start=window_size,
    )
    plot_curve(
        axes[0],
        ddpg_cheetah_smoothed,
        label="Standard DDPG",
        color="orange",
        x_start=window_size,
    )

    axes[0].set_title(
        f"Phase 0: HalfCheetah Forward, $\\gamma$ = {gamma}",
        fontsize=14,
    )
    axes[0].set_xlabel("Episodes")
    axes[0].set_ylabel("Return (Smoothed)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # Right panel: Walker2d transfer vs scratch
    # ---------------------------------------------------------
    plot_curve(
        axes[1],
        sf_walker_transfer_smoothed,
        label="SF-DDPG (After HalfCheetah)",
        color="blue",
        linestyle="-",
        x_start=window_size,
    )
    plot_curve(
        axes[1],
        ddpg_walker_transfer_smoothed,
        label="Standard DDPG (After HalfCheetah)",
        color="orange",
        linestyle="-",
        x_start=window_size,
    )
    plot_curve(
        axes[1],
        sf_walker_scratch_smoothed,
        label="SF-DDPG (From Scratch)",
        color="blue",
        linestyle="--",
        x_start=window_size,
    )
    plot_curve(
        axes[1],
        ddpg_walker_scratch_smoothed,
        label="Standard DDPG (From Scratch)",
        color="orange",
        linestyle="--",
        x_start=window_size,
    )

    axes[1].set_title(
        f"Phase 1: Walker2d Forward — Transfer vs Scratch, "
        f"$\\gamma$ = {gamma}",
        fontsize=14,
    )
    axes[1].set_xlabel("Episodes")
    axes[1].set_ylabel("Return (Smoothed)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Keep the panels visually close, as in the reference figure.
    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.12,
        top=0.84 if main_title else 0.91,
        # Extra horizontal space prevents the Walker2d y-axis label
        # and tick labels from overlapping the HalfCheetah panel.
        wspace=0.14,
    )

    return fig, axes


def resolve_transfer_and_scratch_roots(
    completed_root=None,
    seq_dir=None,
    scratch_dir=None,
):
    """
    Resolve input roots for transfer and scratch data.

    With the new nested layout, completed_root is sufficient and both roots
    point to the same directory. The run_kind argument used by load_data()
    selects the proper inner folder for each seed.
    """
    if completed_root is not None:
        root = os.path.normpath(completed_root)
        return root, root

    if seq_dir is None or scratch_dir is None:
        raise ValueError(
            "Provide either --completed_root, or both --seq_dir and "
            "--scratch_dir."
        )

    return (
        os.path.normpath(seq_dir),
        os.path.normpath(scratch_dir),
    )

if __name__ == "__main__":
    import argparse

    class CleanArgumentParser(argparse.ArgumentParser):
        def convert_arg_line_to_args(self, arg_line):
            line = arg_line.strip()
            if not line or line.startswith("#"):
                return []
            return [line]

    parser = CleanArgumentParser(description="Generate learning curve plots.", fromfile_prefix_chars='@')
    parser.add_argument("--transfer_dir", type=str, default="transfer_learning", help="Subfolder in artifacts containing the transfer runs")
    parser.add_argument("--steps", type=str, default="50000", help="Number of steps per phase used in the folder names")
    parser.add_argument("--lambda_q", type=str, default="0.2", help="Q-loss weight used in the run")
    parser.add_argument("--lambda_vec", type=str, default="1.0", help="Vec-loss weight used in the run")
    parser.add_argument("--gamma", type=str, default="0.99", help="Gamma discount factor used in the transfer learning run")
    parser.add_argument("--prefix", type=str, default="3DFeatures", help="Prefix of the folder name (e.g., eval or 3DFeatures)")
    parser.add_argument("--plot_transfer", action="store_true", help="Generate the transfer learning comparison plot")
    parser.add_argument("--plot_gamma", action="store_true", help="Generate the gamma sweep plots")
    parser.add_argument("--plot_all", action="store_true", help="Generate all plots")
    parser.add_argument("--scratch_suffix", type=str, default="backward_only", help="Suffix for the scratch training folder")
    parser.add_argument("--main_title", type=str, default="Task 1: Forward | Task 2: Backward", help="Main title for the whole figure")

    parser.add_argument(
    "--plot_cheetah",
    action="store_true",
    help="Plot only Phase 0 HalfCheetah forward training.",
    )

    parser.add_argument(
        "--plot_walker",
        action="store_true",
        help=(
            "Plot Walker2d Phase 1 after HalfCheetah pretraining and compare "
            "it with Walker2d training from scratch."
        ),
    )

    parser.add_argument(
        "--plot_combined",
        action="store_true",
        help=(
            "Generate one two-panel figure with HalfCheetah Phase 0 on the "
            "left and Walker2d transfer-vs-scratch on the right."
        ),
    )

    parser.add_argument(
        "--seq_dir",
        type=str,
        default=None,
        help=(
            "Direct path to the sequential Cheetah-to-Walker run directory "
            "containing the seed_* folders."
        ),
    )

    parser.add_argument(
        "--scratch_dir",
        type=str,
        default=None,
        help=(
            "Direct path to the Walker2d-from-scratch run directory "
            "containing the seed_* folders."
        ),
    )


    parser.add_argument(
        "--completed_root",
        type=str,
        default=None,
        help=(
            "Common root containing outer seed_* folders. Each outer seed "
            "must contain seed_*_transfer_completed and "
            "seed_*_scratch_completed."
        ),
    )

    parser.add_argument(
        "--window_size",
        type=int,
        default=20,
        help="Moving-average window measured in episodes.",
    )

    args = parser.parse_args()

    if args.plot_all:
        args.plot_transfer = True
        args.plot_gamma = True
        args.plot_cheetah = True
        args.plot_walker = True
        args.plot_combined = True

    if not (
        args.plot_transfer
        or args.plot_gamma
        or args.plot_cheetah
        or args.plot_walker
        or args.plot_combined
    ):
        print(
            "Error: specify at least one option among "
            "--plot_transfer, --plot_gamma, --plot_cheetah, "
            "--plot_walker, --plot_combined, or --plot_all"
        )
        raise SystemExit(1)

    ROOT_DIR = "artifacts"
    
    # Format the parameters to match the folder naming convention (replace . with _)
    lq_str = args.lambda_q.replace('.', '_')
    lvec_str = args.lambda_vec.replace('.', '_')
    gamma_str = args.gamma.replace('.', '_')

    # Create the figures directory
    figures_dir = os.path.join(ROOT_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)


    if args.plot_combined:
        SEQ_DIR, SCRATCH_DIR = resolve_transfer_and_scratch_roots(
            completed_root=args.completed_root,
            seq_dir=args.seq_dir,
            scratch_dir=args.scratch_dir,
        )

        if not os.path.isdir(SEQ_DIR):
            raise FileNotFoundError(
                "Sequential/transfer directory not found:\n"
                f"{SEQ_DIR}\n\n"
                "The directory must contain the outer seed_* folders. In the new layout, each outer seed contains the corresponding seed_*_transfer_completed or seed_*_scratch_completed folder."
            )

        if not os.path.isdir(SCRATCH_DIR):
            raise FileNotFoundError(
                "Walker-from-scratch directory not found:\n"
                f"{SCRATCH_DIR}\n\n"
                "The directory must contain the outer seed_* folders. In the new layout, each outer seed contains the corresponding seed_*_transfer_completed or seed_*_scratch_completed folder."
            )

        output_folder = os.path.join(
            figures_dir,
            f"cheetah_walker_combined_{args.steps}_steps",
        )
        os.makedirs(output_folder, exist_ok=True)

        fig, axes = generate_cheetah_walker_combined_plot(
            seq_dir=SEQ_DIR,
            scratch_dir=SCRATCH_DIR,
            gamma=args.gamma,
            main_title=args.main_title,
            window_size=args.window_size,
        )

        pdf_path = os.path.join(
            output_folder,
            (
                f"cheetah_walker_combined_"
                f"gamma_{gamma_str}_"
                f"lq_{lq_str}_"
                f"lvec_{lvec_str}_"
                f"steps_{args.steps}.pdf"
            ),
        )

        png_path = os.path.join(
            output_folder,
            (
                f"cheetah_walker_combined_"
                f"gamma_{gamma_str}_"
                f"lq_{lq_str}_"
                f"lvec_{lvec_str}_"
                f"steps_{args.steps}.png"
            ),
        )

        fig.savefig(
            pdf_path,
            bbox_inches="tight",
        )
        fig.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        print(f"Saved combined PDF: {pdf_path}")
        print(f"Saved combined PNG: {png_path}")

    if args.plot_cheetah:

        if args.completed_root is not None:
            SEQ_DIR = os.path.normpath(args.completed_root)

        elif args.seq_dir is not None:
            SEQ_DIR = os.path.normpath(args.seq_dir)

        else:
            # Fallback: reconstruct the directory using the old naming convention.
            base_name_seq = (
                f"{args.prefix}_transfer_gamma_{gamma_str}"
                f"_lq_{lq_str}"
                f"_lvec_{lvec_str}"
                f"_stepsxphase_{args.steps}"
                f"_transfer_learning"
            )

            if os.path.isabs(args.transfer_dir):
                transfer_root = args.transfer_dir
            else:
                transfer_root = os.path.join(ROOT_DIR, args.transfer_dir)

            SEQ_DIR = os.path.join(
                transfer_root,
                base_name_seq,
            )

        if not os.path.isdir(SEQ_DIR):
            raise FileNotFoundError(
                "Sequential training directory not found:\n"
                f"{SEQ_DIR}\n\n"
                "The directory must contain the outer seed_* folders. In the new layout, each outer seed contains the corresponding seed_*_transfer_completed or seed_*_scratch_completed folder."
            )

        output_folder = os.path.join(
            figures_dir,
            f"cheetah_forward_{args.steps}_steps",
        )
        os.makedirs(output_folder, exist_ok=True)

        fig, ax = generate_cheetah_forward_plot(
            seq_dir=SEQ_DIR,
            gamma=args.gamma,
            main_title="Phase 0: HalfCheetah Forward",
            window_size=args.window_size,
        )

        pdf_path = os.path.join(
            output_folder,
            (
                f"cheetah_forward_"
                f"gamma_{gamma_str}_"
                f"lq_{lq_str}_"
                f"lvec_{lvec_str}_"
                f"steps_{args.steps}.pdf"
            ),
        )

        png_path = os.path.join(
            output_folder,
            (
                f"cheetah_forward_"
                f"gamma_{gamma_str}_"
                f"lq_{lq_str}_"
                f"lvec_{lvec_str}_"
                f"steps_{args.steps}.png"
            ),
        )

        fig.savefig(
            pdf_path,
            bbox_inches="tight",
        )

        fig.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Saved PDF: {pdf_path}")
        print(f"Saved PNG: {png_path}")

    if args.plot_walker:
        SEQ_DIR, SCRATCH_DIR = resolve_transfer_and_scratch_roots(
            completed_root=args.completed_root,
            seq_dir=args.seq_dir,
            scratch_dir=args.scratch_dir,
        )

        if not os.path.isdir(SEQ_DIR):
            raise FileNotFoundError(
                "Sequential/transfer directory not found:\n"
                f"{SEQ_DIR}\n\n"
                "The directory must contain the outer seed_* folders. In the new layout, each outer seed contains the corresponding seed_*_transfer_completed or seed_*_scratch_completed folder."
            )

        if not os.path.isdir(SCRATCH_DIR):
            raise FileNotFoundError(
                "Walker-from-scratch directory not found:\n"
                f"{SCRATCH_DIR}\n\n"
                "The directory must contain the outer seed_* folders. In the new layout, each outer seed contains the corresponding seed_*_transfer_completed or seed_*_scratch_completed folder."
            )

        output_folder = os.path.join(
            figures_dir,
            f"walker_transfer_vs_scratch_{args.steps}_steps",
        )
        os.makedirs(output_folder, exist_ok=True)

        fig, ax = generate_walker_transfer_plot(
            seq_dir=SEQ_DIR,
            scratch_dir=SCRATCH_DIR,
            gamma=args.gamma,
            main_title="Phase 1: Walker2d Forward — Transfer vs Scratch",
            window_size=args.window_size,
        )

        pdf_path = os.path.join(
            output_folder,
            (
                f"walker_transfer_vs_scratch_"
                f"gamma_{gamma_str}_"
                f"lq_{lq_str}_"
                f"lvec_{lvec_str}_"
                f"steps_{args.steps}.pdf"
            ),
        )

        png_path = os.path.join(
            output_folder,
            (
                f"walker_transfer_vs_scratch_"
                f"gamma_{gamma_str}_"
                f"lq_{lq_str}_"
                f"lvec_{lvec_str}_"
                f"steps_{args.steps}.png"
            ),
        )

        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved PDF: {pdf_path}")
        print(f"Saved PNG: {png_path}")

    if args.plot_transfer:
        # Dynamically build the directory strings using the gamma argument
        base_name_seq = f"{args.prefix}_transfer_gamma_{gamma_str}_lq_{lq_str}_lvec_{lvec_str}_stepsxphase_{args.steps}_transfer_learning"
        base_name_scratch = f"{args.prefix}_transfer_gamma_{gamma_str}_lq_{lq_str}_lvec_{lvec_str}_stepsxphase_{args.steps}_{args.scratch_suffix}"
        
        SEQ_DIR = os.path.join(ROOT_DIR, args.transfer_dir, base_name_seq)
        SCRATCH_DIR = os.path.join(ROOT_DIR, args.transfer_dir, base_name_scratch)
        
        # Dynamically name the output folder
        transfer_comparison_folder = os.path.join(figures_dir, f"transfer_comparison_{args.prefix}_gamma_{gamma_str}_{args.steps}_steps")
        os.makedirs(transfer_comparison_folder, exist_ok=True)

        fig, axes = generate_transfer_comparison_plot(SEQ_DIR, SCRATCH_DIR, args.gamma, args.main_title, window_size=args.window_size)
        
        save_path = os.path.join(transfer_comparison_folder, f"transfer_{args.prefix}_gamma_{args.gamma}_lq_{lq_str}_lvec_{lvec_str}_stepsxphase_{args.steps}.pdf")
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Saved: {save_path}")

    if args.plot_gamma:
        # Dynamically name the output folder based on the number of steps
        gamma_folder = os.path.join(figures_dir, f"gamma_sweep_{args.steps}_steps")
        os.makedirs(gamma_folder, exist_ok=True)
        
        gamma_values = ["0.1", "0.2" , "0.3", "0.5", "0.7", "0.8"]

        for gamma in gamma_values:
            g_str = gamma.replace('.', '_')
            folder_name_1 = "gamma_sweep"  # Base folder name for gamma sweep runs
            folder_name_2 = f"{args.prefix}_gamma_{g_str}_lq_{lq_str}_lvec_{lvec_str}_stepsxphase_{args.steps}" 
            run_dir = os.path.join(ROOT_DIR, folder_name_1, folder_name_2)
            
            fig, axes = generate_gamma_sweep_plot(run_dir, gamma, args.main_title)

            pdf_filename = f"gamma_{g_str}_lq_{lq_str}_lvec_{lvec_str}_stepsxphase_{args.steps}.pdf"
            save_path = os.path.join(gamma_folder, pdf_filename)
            
            fig.savefig(save_path, bbox_inches='tight')
            plt.close(fig) 
            print(f"Saved: {save_path}")