import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_timeseries(json_path):
    """Load a rollout diagnostics JSON file and extract its timeseries block."""
    json_path = Path(json_path)

    with open(json_path, "r") as f:
        data = json.load(f)

    if "timeseries" not in data:
        raise ValueError(
            "The JSON file does not contain a 'timeseries' block. "
            "Run diagnose_rollout_dynamics.py with --save_timeseries or --render."
        )

    ts = data["timeseries"]

    if ts is None:
        raise ValueError("The 'timeseries' block is None.")

    return data, ts


def as_array(ts, key, default=np.nan):
    """Convert a timeseries field into a NumPy array."""
    values = ts.get(key, None)

    if values is None:
        return None

    cleaned = []
    for x in values:
        if x is None:
            cleaned.append(default)
        else:
            cleaned.append(x)

    return np.asarray(cleaned, dtype=np.float64)


def ensure_output_dir(output_dir):
    """Create output directory if needed."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def add_event_lines(ax, event_steps):
    """Add optional vertical event markers to an axis."""
    if event_steps is None:
        return

    for step in event_steps:
        ax.axvline(step, linestyle="--", linewidth=1.2, alpha=0.8)


def save_figure(fig, output_path):
    """Save and close a matplotlib figure."""
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_overview_timeseries(data, ts, output_dir, event_steps=None):
    """
    Main synchronized rollout timeline.

    This is the most important figure for the report:
        1. velocity/reward behavior
        2. action magnitude/saturation
        3. action roughness
        4. action shift from actor
        5. critic Q improvement
        6. posture/state proxies
    """
    t = as_array(ts, "t")

    x_velocity = as_array(ts, "x_velocity")
    scalar_reward = as_array(ts, "scalar_reward")
    cumulative_return = as_array(ts, "cumulative_return")

    actor_action_norm = as_array(ts, "actor_action_norm")
    final_action_norm = as_array(ts, "final_action_norm")
    action_saturation = as_array(ts, "action_saturation_fraction")

    delta_action_norm = as_array(ts, "delta_action_norm")
    action_jerk_norm = as_array(ts, "action_jerk_norm")

    action_shift = as_array(ts, "action_shift_from_actor")
    q_improvement = as_array(ts, "predicted_q_improvement")

    rootz = as_array(ts, "obs_0_rootz")
    torso_angle = as_array(ts, "obs_1_rooty_angle")
    x_vel_obs = as_array(ts, "obs_8_x_velocity_observation")
    angular_velocity_proxy = as_array(ts, "obs_10_candidate_angular_velocity")

    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)

    fig.suptitle(
        "Zero-shot rollout timeline: behavior, actions, critic signal, and posture proxies",
        fontsize=15,
        y=0.995,
    )

    # 1. Velocity and scalar reward.
    axes[0].plot(t, x_velocity, label="x_velocity")
    axes[0].plot(t, scalar_reward, label="scalar reward", alpha=0.8)
    axes[0].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Value")
    axes[0].set_title("Backward task signal: negative velocity is desirable")
    axes[0].legend(loc="upper right")
    add_event_lines(axes[0], event_steps)

    # 2. Cumulative return.
    axes[1].plot(t, cumulative_return, label="cumulative return")
    axes[1].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Return")
    axes[1].set_title("Accumulated backward-task return")
    axes[1].legend(loc="upper right")
    add_event_lines(axes[1], event_steps)

    # 3. Action magnitude and saturation.
    axes[2].plot(t, actor_action_norm, label="actor action norm", alpha=0.8)
    axes[2].plot(t, final_action_norm, label="final action norm")
    axes[2].axhline(np.sqrt(6), linestyle="--", linewidth=1.0, label="max norm sqrt(6)")
    axes[2].plot(t, action_saturation, label="saturation fraction", alpha=0.8)
    axes[2].set_ylabel("Action")
    axes[2].set_title("Action magnitude and saturation")
    axes[2].legend(loc="upper right")
    add_event_lines(axes[2], event_steps)

    # 4. Temporal roughness.
    axes[3].plot(t, delta_action_norm, label="||a_t - a_{t-1}||")
    axes[3].plot(t, action_jerk_norm, label="action jerk norm", alpha=0.85)
    axes[3].set_ylabel("Roughness")
    axes[3].set_title("Temporal irregularity of the optimized action sequence")
    axes[3].legend(loc="upper right")
    add_event_lines(axes[3], event_steps)

    # 5. Critic/action optimization.
    axes[4].plot(t, action_shift, label="||a_final - a_actor||")
    if q_improvement is not None and not np.all(np.isnan(q_improvement)):
        axes[4].plot(t, q_improvement, label="predicted Q improvement", alpha=0.85)
    axes[4].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[4].set_ylabel("Optimization")
    axes[4].set_title("How far action optimization moves away from the actor")
    axes[4].legend(loc="upper right")
    add_event_lines(axes[4], event_steps)

    # 6. State/posture proxies.
    axes[5].plot(t, rootz, label="obs[0] rootz")
    axes[5].plot(t, torso_angle, label="obs[1] rooty angle")
    axes[5].plot(t, x_vel_obs, label="obs[8] x-velocity obs", alpha=0.8)
    axes[5].plot(t, angular_velocity_proxy, label="obs[10] angular velocity proxy", alpha=0.8)
    axes[5].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[5].set_xlabel("Environment step")
    axes[5].set_ylabel("State")
    axes[5].set_title("Selected state coordinates / posture proxies")
    axes[5].legend(loc="upper right")
    add_event_lines(axes[5], event_steps)

    output_path = output_dir / "timeline_overview.png"
    save_figure(fig, output_path)


def plot_reward_decomposition(ts, output_dir, event_steps=None):
    """Plot reward decomposition over time."""
    t = as_array(ts, "t")
    velocity_term = as_array(ts, "velocity_reward_term")
    ctrl_term = as_array(ts, "ctrl_reward_term")
    scalar_reward = as_array(ts, "scalar_reward")

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(t, velocity_term, label="velocity reward term")
    ax.plot(t, ctrl_term, label="control reward term")
    ax.plot(t, scalar_reward, label="scalar reward", linewidth=2.0)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)

    add_event_lines(ax, event_steps)

    ax.set_title("Reward decomposition during zero-shot rollout")
    ax.set_xlabel("Environment step")
    ax.set_ylabel("Reward contribution")
    ax.legend(loc="upper right")

    output_path = output_dir / "reward_decomposition.png"
    save_figure(fig, output_path)


def plot_action_heatmap(ts, output_dir, action_key="final_actions", event_steps=None):
    """
    Plot a heatmap of action components over time.

    This is useful to visually inspect saturation, sign flips and temporal patterns.
    """
    t = as_array(ts, "t")
    actions = np.asarray(ts[action_key], dtype=np.float64)

    if actions.ndim != 2:
        raise ValueError(f"{action_key} must have shape [T, action_dim].")

    fig, ax = plt.subplots(figsize=(14, 4.5))

    im = ax.imshow(
        actions.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[t[0], t[-1], 0, actions.shape[1] - 1],
        vmin=-1.0,
        vmax=1.0,
    )

    if event_steps is not None:
        for step in event_steps:
            ax.axvline(step, linestyle="--", linewidth=1.2)

    ax.set_title(f"Action component heatmap: {action_key}")
    ax.set_xlabel("Environment step")
    ax.set_ylabel("Action dimension")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Action value")

    output_path = output_dir / f"{action_key}_heatmap.png"
    save_figure(fig, output_path)


def plot_action_components(ts, output_dir, action_key="final_actions", event_steps=None):
    """Plot each action component as a line over time."""
    t = as_array(ts, "t")
    actions = np.asarray(ts[action_key], dtype=np.float64)

    if actions.ndim != 2:
        raise ValueError(f"{action_key} must have shape [T, action_dim].")

    fig, ax = plt.subplots(figsize=(14, 6))

    for dim in range(actions.shape[1]):
        ax.plot(t, actions[:, dim], label=f"a[{dim}]", alpha=0.85)

    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.axhline(1.0, linestyle=":", linewidth=1.0)
    ax.axhline(-1.0, linestyle=":", linewidth=1.0)

    add_event_lines(ax, event_steps)

    ax.set_title(f"Action components over time: {action_key}")
    ax.set_xlabel("Environment step")
    ax.set_ylabel("Action value")
    ax.legend(loc="upper right", ncol=3)

    output_path = output_dir / f"{action_key}_components.png"
    save_figure(fig, output_path)


def plot_critic_trace(ts, output_dir, event_steps=None):
    """Plot predicted Q before/after action optimization and Q improvement."""
    t = as_array(ts, "t")

    q_before = as_array(ts, "predicted_q_before")
    q_after = as_array(ts, "predicted_q_after")
    q_improvement = as_array(ts, "predicted_q_improvement")
    action_shift = as_array(ts, "action_shift_from_actor")

    if q_before is None or np.all(np.isnan(q_before)):
        print("Skipping critic trace: no predicted Q values found.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(t, q_before, label="Q before")
    axes[0].plot(t, q_after, label="Q after")
    axes[0].set_title("Predicted critic value before/after action optimization")
    axes[0].set_ylabel("Predicted Q")
    axes[0].legend(loc="upper right")
    add_event_lines(axes[0], event_steps)

    axes[1].plot(t, q_improvement, label="Q improvement")
    axes[1].plot(t, action_shift, label="||a_final - a_actor||", alpha=0.85)
    axes[1].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[1].set_title("Local critic improvement and action displacement")
    axes[1].set_xlabel("Environment step")
    axes[1].set_ylabel("Value")
    axes[1].legend(loc="upper right")
    add_event_lines(axes[1], event_steps)

    output_path = output_dir / "critic_optimization_trace.png"
    save_figure(fig, output_path)


def plot_state_proxies(ts, output_dir, event_steps=None):
    """Plot selected observation coordinates and broad state norms."""
    t = as_array(ts, "t")

    rootz = as_array(ts, "obs_0_rootz")
    torso_angle = as_array(ts, "obs_1_rooty_angle")
    x_vel_obs = as_array(ts, "obs_8_x_velocity_observation")
    angular_velocity_proxy = as_array(ts, "obs_10_candidate_angular_velocity")
    state_l2 = as_array(ts, "state_l2_norm")
    velocity_block_l2 = as_array(ts, "state_velocity_block_l2_norm")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(t, rootz, label="obs[0] rootz")
    axes[0].plot(t, torso_angle, label="obs[1] rooty angle")
    axes[0].plot(t, x_vel_obs, label="obs[8] x-velocity obs", alpha=0.85)
    axes[0].plot(t, angular_velocity_proxy, label="obs[10] angular velocity proxy", alpha=0.85)
    axes[0].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[0].set_title("Selected HalfCheetah observation coordinates")
    axes[0].set_ylabel("Observation value")
    axes[0].legend(loc="upper right")
    add_event_lines(axes[0], event_steps)

    axes[1].plot(t, state_l2, label="state L2 norm")
    axes[1].plot(t, velocity_block_l2, label="velocity block L2 norm", alpha=0.85)
    axes[1].set_title("Broad state instability proxies")
    axes[1].set_xlabel("Environment step")
    axes[1].set_ylabel("Norm")
    axes[1].legend(loc="upper right")
    add_event_lines(axes[1], event_steps)

    output_path = output_dir / "state_posture_proxies.png"
    save_figure(fig, output_path)


def plot_phase_summary(ts, output_dir, event_steps=None):
    """
    Compact report-friendly figure with the most interpretable traces only.

    This is the one I would likely put in the final presentation/report.
    """
    t = as_array(ts, "t")

    x_velocity = as_array(ts, "x_velocity")
    scalar_reward = as_array(ts, "scalar_reward")
    final_action_norm = as_array(ts, "final_action_norm")
    delta_action_norm = as_array(ts, "delta_action_norm")
    action_jerk_norm = as_array(ts, "action_jerk_norm")
    action_shift = as_array(ts, "action_shift_from_actor")
    torso_angle = as_array(ts, "obs_1_rooty_angle")

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)

    fig.suptitle(
        "Synchronized zero-shot rollout diagnostics",
        fontsize=15,
        y=0.995,
    )

    axes[0].plot(t, x_velocity, label="x_velocity")
    axes[0].plot(t, scalar_reward, label="scalar reward", alpha=0.85)
    axes[0].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[0].set_title("Backward behavior signal")
    axes[0].set_ylabel("Value")
    axes[0].legend(loc="upper right")
    add_event_lines(axes[0], event_steps)

    axes[1].plot(t, final_action_norm, label="||a||")
    axes[1].axhline(np.sqrt(6), linestyle="--", linewidth=1.0, label="max norm sqrt(6)")
    axes[1].set_title("Action magnitude")
    axes[1].set_ylabel("Norm")
    axes[1].legend(loc="upper right")
    add_event_lines(axes[1], event_steps)

    axes[2].plot(t, delta_action_norm, label="||a_t - a_{t-1}||")
    axes[2].plot(t, action_jerk_norm, label="action jerk norm", alpha=0.85)
    axes[2].set_title("Temporal action roughness")
    axes[2].set_ylabel("Roughness")
    axes[2].legend(loc="upper right")
    add_event_lines(axes[2], event_steps)

    axes[3].plot(t, action_shift, label="||a_final - a_actor||")
    axes[3].plot(t, torso_angle, label="obs[1] rooty angle", alpha=0.85)
    axes[3].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[3].set_title("OOD action displacement and posture proxy")
    axes[3].set_xlabel("Environment step")
    axes[3].set_ylabel("Value")
    axes[3].legend(loc="upper right")
    add_event_lines(axes[3], event_steps)

    output_path = output_dir / "presentation_timeline_summary.png"
    save_figure(fig, output_path)


def parse_event_steps(raw_event_steps):
    """Parse comma-separated event steps from CLI."""
    if raw_event_steps is None or raw_event_steps.strip() == "":
        return None

    steps = []
    for item in raw_event_steps.split(","):
        item = item.strip()
        if item:
            steps.append(int(item))

    return steps if len(steps) > 0 else None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to rollout_dynamics JSON containing a timeseries block.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where figures will be saved. Defaults to <json_dir>/timeseries_plots.",
    )
    parser.add_argument(
        "--event_steps",
        type=str,
        default=None,
        help="Optional comma-separated timestep markers, e.g. '120,350,600'.",
    )
    parser.add_argument(
        "--make_all",
        action="store_true",
        help="If set, generate all available figures.",
    )

    args = parser.parse_args()

    json_path = Path(args.json_path)
    data, ts = load_timeseries(json_path)

    if args.output_dir is None:
        output_dir = json_path.parent / "timeseries_plots"
    else:
        output_dir = Path(args.output_dir)

    output_dir = ensure_output_dir(output_dir)
    event_steps = parse_event_steps(args.event_steps)

    print(f"Loaded JSON: {json_path}")
    print(f"Saving plots to: {output_dir}")

    if "episode_return_mean" in data:
        print(f"Episode return mean: {data['episode_return_mean']:.3f}")
    if "timeseries" in data:
        print(f"Timeseries length: {len(ts.get('t', []))}")

    # Always generate the two most useful figures.
    plot_phase_summary(ts, output_dir, event_steps=event_steps)
    plot_overview_timeseries(data, ts, output_dir, event_steps=event_steps)

    if args.make_all:
        plot_reward_decomposition(ts, output_dir, event_steps=event_steps)
        plot_action_heatmap(ts, output_dir, action_key="final_actions", event_steps=event_steps)
        plot_action_heatmap(ts, output_dir, action_key="actor_actions", event_steps=event_steps)
        # plot_action_components(ts, output_dir, action_key="final_actions", event_steps=event_steps)
        # plot_action_components(ts, output_dir, action_key="actor_actions", event_steps=event_steps)
        plot_critic_trace(ts, output_dir, event_steps=event_steps)
        plot_state_proxies(ts, output_dir, event_steps=event_steps)


if __name__ == "__main__":
    main()