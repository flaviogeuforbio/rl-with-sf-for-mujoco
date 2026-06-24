import os
os.environ.setdefault("MUJOCO_GL", "egl") # decomment this line if you want to run diagnostics with rendering on a headless server (e.g. Colab of Kaggle or Leonardo) (requires EGL support); keep it commented if you want to run it locally with rendering to a window

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import imageio.v2 as imageio

from ActorCritic import Actor, SFCritic, QCritic


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###########################################
# modifica
def sanitize_folder_value(value):
    return (
        str(value)
        .strip()
        .replace(".", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def build_diagnostics_dir(
    run_dir,
    gamma,
    phase,
    mode,
    model_type,
    task,
):
    gamma_name = sanitize_folder_value(gamma)
    mode_name = sanitize_folder_value(mode)
    model_name = sanitize_folder_value(model_type)
    task_name = sanitize_folder_value(task)

    folder_name = (
        f"gamma_{gamma_name}"
        f"__phase_{phase}"
        f"__mode_{mode_name}"
        f"__model_{model_name}"
        f"__task_{task_name}"
    )

    diagnostics_dir = run_dir / "diagnostics" / folder_name
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    return diagnostics_dir
#############################################################


def make_task_weights(device):
    """Create the forward and backward task vectors used in training."""
    w_forward = torch.tensor([[1.0], [1.0]], dtype=torch.float32, device=device)
    w_backward = torch.tensor([[-1.0], [1.0]], dtype=torch.float32, device=device)
    return w_forward, w_backward


def compute_phi_and_reward(info, task_weights):
    """
    Extract handcrafted HalfCheetah features and compute scalar reward.

    In Gymnasium HalfCheetah, reward_ctrl is already negative:
        reward_ctrl = -ctrl_cost

    Therefore:
        forward reward  =  x_velocity + reward_ctrl
        backward reward = -x_velocity + reward_ctrl
    """
    velocity = float(info.get("x_velocity", 0.0))
    ctrl_reward = float(info.get("reward_ctrl", 0.0))

    phi = np.array([velocity, ctrl_reward], dtype=np.float32)
    w_np = task_weights.detach().cpu().numpy().flatten()

    scalar_reward = float(phi @ w_np)

    velocity_term = float(phi[0] * w_np[0])
    ctrl_term = float(phi[1] * w_np[1])

    return phi, scalar_reward, velocity_term, ctrl_term


def summarize_array(name, values):
    """Compute robust summary statistics for a one-dimensional array."""
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        return {
            f"{name}_mean": None,
            f"{name}_std": None,
            f"{name}_min": None,
            f"{name}_max": None,
            f"{name}_median": None,
            f"{name}_p95": None,
        }

    return {
        f"{name}_mean": float(np.mean(values)),
        f"{name}_std": float(np.std(values)),
        f"{name}_min": float(np.min(values)),
        f"{name}_max": float(np.max(values)),
        f"{name}_median": float(np.median(values)),
        f"{name}_p95": float(np.percentile(values, 95)),
    }


def compute_action_diagnostics(actions, max_action):
    """
    Compute action magnitude, saturation and temporal roughness diagnostics.

    actions shape:
        [T, action_dim]
    """
    actions = np.asarray(actions, dtype=np.float64)

    if len(actions) == 0:
        return {}

    action_norms = np.linalg.norm(actions, axis=1)
    saturation_fraction = float(np.mean(np.abs(actions) > 0.95 * max_action))

    diagnostics = {}
    diagnostics.update(summarize_array("action_norm", action_norms))
    diagnostics["action_saturation_fraction"] = saturation_fraction

    if len(actions) >= 2:
        delta_actions = actions[1:] - actions[:-1]
        delta_norms = np.linalg.norm(delta_actions, axis=1)

        diagnostics.update(summarize_array("delta_action_norm", delta_norms))

        signs = np.sign(actions)
        sign_flips = signs[1:] != signs[:-1]
        diagnostics["action_sign_flip_rate"] = float(np.mean(sign_flips))
    else:
        diagnostics.update(summarize_array("delta_action_norm", []))
        diagnostics["action_sign_flip_rate"] = None

    if len(actions) >= 3:
        jerk_actions = actions[2:] - 2.0 * actions[1:-1] + actions[:-2]
        jerk_norms = np.linalg.norm(jerk_actions, axis=1)
        diagnostics.update(summarize_array("action_jerk_norm", jerk_norms))
    else:
        diagnostics.update(summarize_array("action_jerk_norm", []))

    per_dim_sat = np.mean(np.abs(actions) > 0.95 * max_action, axis=0)
    diagnostics["action_saturation_fraction_per_dim"] = [
        float(x) for x in per_dim_sat
    ]

    return diagnostics


def compute_state_diagnostics(states):
    """
    Compute broad state-space statistics.

    This does not assume a specific semantic meaning for each state dimension.
    It is meant to detect large excursions or instability.
    """
    states = np.asarray(states, dtype=np.float64)

    if len(states) == 0:
        return {}

    return {
        "state_mean_per_dim": [float(x) for x in np.mean(states, axis=0)],
        "state_std_per_dim": [float(x) for x in np.std(states, axis=0)],
        "state_min_per_dim": [float(x) for x in np.min(states, axis=0)],
        "state_max_per_dim": [float(x) for x in np.max(states, axis=0)],
        "state_abs_max_per_dim": [float(x) for x in np.max(np.abs(states), axis=0)],
    }


# ============================================================
# NEW: Spurious flipped-basin indicator
# ============================================================

def _to_float_array_with_nan(values):
    """Convert floats/None values to a float array with NaNs for missing values."""
    converted = []
    for value in values:
        if value is None:
            converted.append(np.nan)
        else:
            converted.append(float(value))
    return np.asarray(converted, dtype=np.float64)


def _summarize_optional_values(name, values):
    """Summarize scalar values while ignoring None and NaN entries."""
    clean_values = []
    for value in values:
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value):
            clean_values.append(value)

    return summarize_array(name, clean_values)


def _find_sustained_flip_start(flip_mask, window_size=50, min_fraction=0.8):
    """
    Find the first timestep where the flipped posture is sustained.

    A sustained flip is detected when at least min_fraction of a sliding
    window satisfies the flip condition.
    """
    flip_mask = np.asarray(flip_mask, dtype=bool)

    if len(flip_mask) == 0:
        return None

    if len(flip_mask) < window_size:
        if float(np.mean(flip_mask)) >= min_fraction:
            return 0
        return None

    for start_idx in range(0, len(flip_mask) - window_size + 1):
        window_fraction = float(np.mean(flip_mask[start_idx:start_idx + window_size]))
        if window_fraction >= min_fraction:
            return int(start_idx)

    return None


def _compute_normalized_q_jump(q_values, flip_start, local_window=50):
    """
    Compute the local normalized Q jump around a sustained flip transition.

    jump_norm = (mean(Q_post) - mean(Q_pre)) / std(Q_episode)
    """
    q_values = _to_float_array_with_nan(q_values)

    if flip_start is None:
        return None, None, None

    if len(q_values) == 0 or not np.any(np.isfinite(q_values)):
        return None, None, None

    t = int(flip_start)

    pre_values = q_values[max(0, t - local_window):t]
    post_values = q_values[t:min(len(q_values), t + local_window)]

    pre_values = pre_values[np.isfinite(pre_values)]
    post_values = post_values[np.isfinite(post_values)]

    if len(pre_values) == 0 or len(post_values) == 0:
        return None, None, None

    q_std = float(np.nanstd(q_values))

    pre_mean = float(np.mean(pre_values))
    post_mean = float(np.mean(post_values))

    if q_std < 1e-8:
        return None, pre_mean, post_mean

    jump_norm = float((post_mean - pre_mean) / (q_std + 1e-8))

    return jump_norm, pre_mean, post_mean


def compute_spurious_basin_indicator_for_episode(
    rooty_angles,
    q_before_values,
    q_after_values,
    flip_threshold=2.5,
    sustained_window=50,
    sustained_min_fraction=0.8,
    q_jump_window=50,
):
    """
    Compute the spurious flipped-basin indicator for one episode.

    Main score:
        B = flip_fraction * max(0, normalized Q_after jump at sustained flip)

    If there is no sustained flip, B is set to 0 when Q_after values exist.
    If Q_after values are not available, B is left undefined as None.
    """
    theta = _to_float_array_with_nan(rooty_angles)
    q_before = _to_float_array_with_nan(q_before_values)
    q_after = _to_float_array_with_nan(q_after_values)

    has_theta = len(theta) > 0 and np.any(np.isfinite(theta))
    has_q_before = len(q_before) > 0 and np.any(np.isfinite(q_before))
    has_q_after = len(q_after) > 0 and np.any(np.isfinite(q_after))

    result = {
        "flip_threshold": float(flip_threshold),
        "sustained_window": int(sustained_window),
        "sustained_min_fraction": float(sustained_min_fraction),
        "q_jump_window": int(q_jump_window),
        "num_steps": int(len(theta)),
        "flip_fraction": None,
        "sustained_flip": False,
        "sustained_flip_start": None,
        "q_after_pre_flip_mean": None,
        "q_after_post_flip_mean": None,
        "q_after_jump_norm_at_flip": None,
        "q_before_pre_flip_mean": None,
        "q_before_post_flip_mean": None,
        "q_before_jump_norm_at_flip": None,
        "spurious_basin_indicator": None,
        "spurious_basin_indicator_q_before": None,
    }

    if not has_theta:
        return result

    valid_theta = np.isfinite(theta)

    flip_mask = np.zeros_like(theta, dtype=bool)
    flip_mask[valid_theta] = np.abs(theta[valid_theta]) > flip_threshold

    flip_fraction = float(np.mean(flip_mask[valid_theta]))

    flip_start = _find_sustained_flip_start(
        flip_mask=flip_mask,
        window_size=sustained_window,
        min_fraction=sustained_min_fraction,
    )

    result["flip_fraction"] = flip_fraction
    result["sustained_flip"] = flip_start is not None
    result["sustained_flip_start"] = None if flip_start is None else int(flip_start)

    if flip_start is None:
        if has_q_after:
            result["spurious_basin_indicator"] = 0.0
        if has_q_before:
            result["spurious_basin_indicator_q_before"] = 0.0
        return result

    q_after_jump_norm, q_after_pre, q_after_post = _compute_normalized_q_jump(
        q_values=q_after_values,
        flip_start=flip_start,
        local_window=q_jump_window,
    )

    q_before_jump_norm, q_before_pre, q_before_post = _compute_normalized_q_jump(
        q_values=q_before_values,
        flip_start=flip_start,
        local_window=q_jump_window,
    )

    result["q_after_pre_flip_mean"] = q_after_pre
    result["q_after_post_flip_mean"] = q_after_post
    result["q_after_jump_norm_at_flip"] = q_after_jump_norm

    result["q_before_pre_flip_mean"] = q_before_pre
    result["q_before_post_flip_mean"] = q_before_post
    result["q_before_jump_norm_at_flip"] = q_before_jump_norm

    if q_after_jump_norm is not None:
        result["spurious_basin_indicator"] = float(
            flip_fraction * max(0.0, q_after_jump_norm)
        )

    if q_before_jump_norm is not None:
        result["spurious_basin_indicator_q_before"] = float(
            flip_fraction * max(0.0, q_before_jump_norm)
        )

    return result


def aggregate_spurious_basin_indicators(per_episode_results):
    """Aggregate spurious flipped-basin indicators over rollout episodes."""
    if len(per_episode_results) == 0:
        return {}

    diagnostics = {
        "per_episode": per_episode_results,
        "sustained_flip_rate": float(
            np.mean([bool(x.get("sustained_flip", False)) for x in per_episode_results])
        ),
    }

    keys_to_summarize = [
        "flip_fraction",
        "sustained_flip_start",
        "q_after_jump_norm_at_flip",
        "q_before_jump_norm_at_flip",
        "spurious_basin_indicator",
        "spurious_basin_indicator_q_before",
    ]

    for key in keys_to_summarize:
        diagnostics.update(
            _summarize_optional_values(
                name=key,
                values=[item.get(key) for item in per_episode_results],
            )
        )

    return diagnostics


def initialize_timeseries():
    """Initialize a dictionary used to store one diagnostic episode over time."""
    return {
        "t": [],

        # Reward and feature traces.
        "x_velocity": [],
        "ctrl_reward": [],
        "scalar_reward": [],
        "velocity_reward_term": [],
        "ctrl_reward_term": [],
        "cumulative_return": [],

        # Action magnitude and temporal roughness.
        "actor_action_norm": [],
        "final_action_norm": [],
        "delta_action_norm": [],
        "action_jerk_norm": [],
        "action_saturation_fraction": [],
        "action_sign_flip_fraction": [],
        "action_shift_from_actor": [],

        # Critic optimization traces.
        "predicted_q_before": [],
        "predicted_q_after": [],
        "predicted_q_improvement": [],

        # Selected observation coordinates.
        # Labels are based on common Gymnasium HalfCheetah-v5 documentation.
        # Keep the raw indices too, because XML/version details may differ.
        "obs_0_rootz": [],
        "obs_1_rooty_angle": [],
        "obs_8_x_velocity_observation": [],
        "obs_10_candidate_angular_velocity": [],

        # More general state instability proxies.
        "state_l2_norm": [],
        "state_velocity_block_l2_norm": [],

        # Full arrays for optional deeper plotting.
        "actor_actions": [],
        "final_actions": [],
        "states": [],
    }


def append_timeseries_step(
    timeseries,
    t,
    state,
    actor_action,
    final_action,
    previous_action,
    previous_previous_action,
    phi,
    scalar_reward,
    velocity_term,
    ctrl_term,
    cumulative_return,
    max_action,
    q_before=None,
    q_after=None,
    q_improvement=None,
):
    """Append one timestep to the diagnostic timeseries."""
    state = np.asarray(state, dtype=np.float64)
    actor_action = np.asarray(actor_action, dtype=np.float64)
    final_action = np.asarray(final_action, dtype=np.float64)

    if previous_action is None:
        delta_action_norm = 0.0
        sign_flip_fraction = 0.0
    else:
        previous_action = np.asarray(previous_action, dtype=np.float64)
        delta_action_norm = float(np.linalg.norm(final_action - previous_action))
        sign_flip_fraction = float(
            np.mean(np.sign(final_action) != np.sign(previous_action))
        )

    if previous_action is None or previous_previous_action is None:
        action_jerk_norm = 0.0
    else:
        previous_action = np.asarray(previous_action, dtype=np.float64)
        previous_previous_action = np.asarray(previous_previous_action, dtype=np.float64)
        jerk = final_action - 2.0 * previous_action + previous_previous_action
        action_jerk_norm = float(np.linalg.norm(jerk))

    saturation_fraction = float(np.mean(np.abs(final_action) > 0.95 * max_action))
    action_shift = float(np.linalg.norm(final_action - actor_action))

    timeseries["t"].append(int(t))

    timeseries["x_velocity"].append(float(phi[0]))
    timeseries["ctrl_reward"].append(float(phi[1]))
    timeseries["scalar_reward"].append(float(scalar_reward))
    timeseries["velocity_reward_term"].append(float(velocity_term))
    timeseries["ctrl_reward_term"].append(float(ctrl_term))
    timeseries["cumulative_return"].append(float(cumulative_return))

    timeseries["actor_action_norm"].append(float(np.linalg.norm(actor_action)))
    timeseries["final_action_norm"].append(float(np.linalg.norm(final_action)))
    timeseries["delta_action_norm"].append(delta_action_norm)
    timeseries["action_jerk_norm"].append(action_jerk_norm)
    timeseries["action_saturation_fraction"].append(saturation_fraction)
    timeseries["action_sign_flip_fraction"].append(sign_flip_fraction)
    timeseries["action_shift_from_actor"].append(action_shift)

    timeseries["predicted_q_before"].append(None if q_before is None else float(q_before))
    timeseries["predicted_q_after"].append(None if q_after is None else float(q_after))
    timeseries["predicted_q_improvement"].append(
        None if q_improvement is None else float(q_improvement)
    )

    # Selected coordinates. Use None if the observation shape is unexpected.
    timeseries["obs_0_rootz"].append(float(state[0]) if len(state) > 0 else None)
    timeseries["obs_1_rooty_angle"].append(float(state[1]) if len(state) > 1 else None)
    timeseries["obs_8_x_velocity_observation"].append(
        float(state[8]) if len(state) > 8 else None
    )
    timeseries["obs_10_candidate_angular_velocity"].append(
        float(state[10]) if len(state) > 10 else None
    )

    timeseries["state_l2_norm"].append(float(np.linalg.norm(state)))

    if len(state) > 8:
        velocity_block = state[8:]
        timeseries["state_velocity_block_l2_norm"].append(
            float(np.linalg.norm(velocity_block))
        )
    else:
        timeseries["state_velocity_block_l2_norm"].append(None)

    timeseries["actor_actions"].append([float(x) for x in actor_action])
    timeseries["final_actions"].append([float(x) for x in final_action])
    timeseries["states"].append([float(x) for x in state])


def optimize_action_with_sf(
    sf_critic,
    state_tensor,
    init_action,
    task_weights,
    max_action,
    n_steps=20,
    step_size=0.05,
    action_l2=1e-3,
    return_diagnostics=False,
):
    """
    Optimize action by gradient ascent on:
        Q_w(s, a) = psi(s, a)^T w

    If return_diagnostics=True, also return predicted Q before/after optimization.
    """
    with torch.no_grad():
        q_before = torch.matmul(
            sf_critic(state_tensor, init_action),
            task_weights,
        ).mean().item()

    action = init_action.clone().detach().requires_grad_(True)

    for _ in range(n_steps):
        psi = sf_critic(state_tensor, action)
        q_value = torch.matmul(psi, task_weights)

        action_penalty = action_l2 * (action ** 2).mean()
        loss = -(q_value.mean() - action_penalty)

        if action.grad is not None:
            action.grad.zero_()

        loss.backward()

        with torch.no_grad():
            action -= step_size * action.grad
            action.clamp_(-max_action, max_action)

        action = action.detach().requires_grad_(True)

    final_action = action.detach()

    if not return_diagnostics:
        return final_action

    with torch.no_grad():
        q_after = torch.matmul(
            sf_critic(state_tensor, final_action),
            task_weights,
        ).mean().item()

    diagnostics = {
        "q_before": float(q_before),
        "q_after": float(q_after),
        "q_improvement": float(q_after - q_before),
        "action_shift_from_actor": float(torch.norm(final_action - init_action).item()),
    }

    return final_action, diagnostics


def optimize_action_with_qcritic(
    q_critic,
    state_tensor,
    init_action,
    max_action,
    n_steps=20,
    step_size=0.05,
    action_l2=1e-3,
    return_diagnostics=False,
):
    """
    Optimize action by gradient ascent on the DDPG scalar Q critic.

    This is not true zero-shot reward reweighting, because the scalar critic
    is tied to its training reward.
    """
    with torch.no_grad():
        q_before = q_critic(state_tensor, init_action).mean().item()

    action = init_action.clone().detach().requires_grad_(True)

    for _ in range(n_steps):
        q_value = q_critic(state_tensor, action)

        action_penalty = action_l2 * (action ** 2).mean()
        loss = -(q_value.mean() - action_penalty)

        if action.grad is not None:
            action.grad.zero_()

        loss.backward()

        with torch.no_grad():
            action -= step_size * action.grad
            action.clamp_(-max_action, max_action)

        action = action.detach().requires_grad_(True)

    final_action = action.detach()

    if not return_diagnostics:
        return final_action

    with torch.no_grad():
        q_after = q_critic(state_tensor, final_action).mean().item()

    diagnostics = {
        "q_before": float(q_before),
        "q_after": float(q_after),
        "q_improvement": float(q_after - q_before),
        "action_shift_from_actor": float(torch.norm(final_action - init_action).item()),
    }

    return final_action, diagnostics


def load_models(run_dir, env_name, phase):
    """Load SF-DDPG and DDPG models from a saved artifact directory."""
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    feature_dim = 2

    env.close()

    sf_actor = Actor(state_dim, action_dim, max_action).to(device)
    sf_critic = SFCritic(state_dim, action_dim, feature_dim).to(device)

    sf_actor.load_state_dict(
        torch.load(run_dir / f"sf_actor_{phase}.pth", map_location=device)
    )
    sf_critic.load_state_dict(
        torch.load(run_dir / f"sf_critic_{phase}.pth", map_location=device)
    )

    q_actor = Actor(state_dim, action_dim, max_action).to(device)
    q_critic = QCritic(state_dim, action_dim).to(device)

    q_actor.load_state_dict(
        torch.load(run_dir / f"q_actor_{phase}.pth", map_location=device)
    )
    q_critic.load_state_dict(
        torch.load(run_dir / f"q_critic_{phase}.pth", map_location=device)
    )

    sf_actor.eval()
    sf_critic.eval()
    q_actor.eval()
    q_critic.eval()

    return sf_actor, sf_critic, q_actor, q_critic, max_action


def select_models_for_mode(mode, sf_actor, sf_critic, q_actor, q_critic):
    """Select the actor and critic associated with a diagnostic mode."""
    if mode == "sf_actor_only":
        return sf_actor, sf_critic, None, "actor_only"

    if mode == "sf_action_optimization":
        return sf_actor, sf_critic, None, "sf_action_optimization"

    if mode == "ddpg_actor_only":
        return q_actor, None, q_critic, "actor_only"

    if mode == "ddpg_q_action_optimization":
        return q_actor, None, q_critic, "q_action_optimization"

    raise ValueError(f"Unknown mode: {mode}")


def diagnose_rollout_dynamics(
    env_name,
    actor,
    task_weights,
    max_action,
    mode,
    sf_critic=None,
    q_critic=None,
    episodes=5,
    max_episode_steps=1000,
    opt_steps=20,
    opt_step_size=0.05,
    action_l2=1e-3,
    render=False,
    render_path=None,
    render_fps=30,
    save_timeseries=False,
    timeseries_episode=0,
):
    """
    Run diagnostic rollouts and measure:
        - reward/feature decomposition
        - action magnitude
        - action temporal roughness
        - action saturation
        - critic Q before/after action optimization
        - state-space excursions
        - spurious flipped-basin indicator over all episodes
        - optional per-timestep timeseries for one selected episode
    """
    env = gym.make(env_name, render_mode="rgb_array") if render else gym.make(env_name)

    frames = []

    episode_returns = []

    all_states = []
    all_actor_actions = []
    all_final_actions = []

    velocities = []
    ctrl_rewards = []
    scalar_rewards = []
    velocity_terms = []
    ctrl_terms = []

    q_before_values = []
    q_after_values = []
    q_improvements = []
    action_shifts = []

    selected_timeseries = None

    # NEW: Store one spurious-basin diagnostic result per episode.
    spurious_basin_episode_results = []

    for episode_idx in range(episodes):
        state, _ = env.reset()
        episode_return = 0.0

        # NEW: Minimal traces needed to compute the basin indicator for this episode.
        episode_rooty_angles = []
        episode_q_before_values = []
        episode_q_after_values = []

        should_render = render and episode_idx == timeseries_episode
        should_save_timeseries = save_timeseries and episode_idx == timeseries_episode

        if should_save_timeseries:
            selected_timeseries = initialize_timeseries()

        if should_render:
            frames.append(env.render())

        previous_action = None
        previous_previous_action = None

        for step_idx in range(max_episode_steps):
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            with torch.no_grad():
                actor_action_tensor = actor(state_tensor)

            q_before = None
            q_after = None
            q_improvement = None

            if mode == "actor_only":
                final_action_tensor = actor_action_tensor

            elif mode == "sf_action_optimization":
                if sf_critic is None:
                    raise ValueError("sf_critic must be provided for sf_action_optimization.")

                final_action_tensor, opt_diag = optimize_action_with_sf(
                    sf_critic=sf_critic,
                    state_tensor=state_tensor,
                    init_action=actor_action_tensor,
                    task_weights=task_weights,
                    max_action=max_action,
                    n_steps=opt_steps,
                    step_size=opt_step_size,
                    action_l2=action_l2,
                    return_diagnostics=True,
                )

                q_before = opt_diag["q_before"]
                q_after = opt_diag["q_after"]
                q_improvement = opt_diag["q_improvement"]

                q_before_values.append(q_before)
                q_after_values.append(q_after)
                q_improvements.append(q_improvement)
                action_shifts.append(opt_diag["action_shift_from_actor"])

            elif mode == "q_action_optimization":
                if q_critic is None:
                    raise ValueError("q_critic must be provided for q_action_optimization.")

                final_action_tensor, opt_diag = optimize_action_with_qcritic(
                    q_critic=q_critic,
                    state_tensor=state_tensor,
                    init_action=actor_action_tensor,
                    max_action=max_action,
                    n_steps=opt_steps,
                    step_size=opt_step_size,
                    action_l2=action_l2,
                    return_diagnostics=True,
                )

                q_before = opt_diag["q_before"]
                q_after = opt_diag["q_after"]
                q_improvement = opt_diag["q_improvement"]

                q_before_values.append(q_before)
                q_after_values.append(q_after)
                q_improvements.append(q_improvement)
                action_shifts.append(opt_diag["action_shift_from_actor"])

            else:
                raise ValueError(f"Unknown internal mode: {mode}")

            actor_action = actor_action_tensor.detach().cpu().numpy()[0]
            final_action = final_action_tensor.detach().cpu().numpy()[0]
            final_action = np.clip(final_action, -max_action, max_action)

            # NEW: Store posture and critic values for the current state.
            # q_before/q_after are defined at this state before env.step().
            episode_rooty_angles.append(float(state[1]) if len(state) > 1 else None)
            episode_q_before_values.append(q_before)
            episode_q_after_values.append(q_after)

            next_state, _, terminated, truncated, info = env.step(final_action)

            if should_render:
                frames.append(env.render())

            phi, scalar_reward, velocity_term, ctrl_term = compute_phi_and_reward(
                info=info,
                task_weights=task_weights,
            )

            velocity = float(phi[0])
            ctrl_reward = float(phi[1])

            episode_return += scalar_reward

            all_states.append(state.copy())
            all_actor_actions.append(actor_action.copy())
            all_final_actions.append(final_action.copy())

            velocities.append(velocity)
            ctrl_rewards.append(ctrl_reward)
            scalar_rewards.append(scalar_reward)
            velocity_terms.append(velocity_term)
            ctrl_terms.append(ctrl_term)

            if should_save_timeseries:
                append_timeseries_step(
                    timeseries=selected_timeseries,
                    t=step_idx,
                    state=state,
                    actor_action=actor_action,
                    final_action=final_action,
                    previous_action=previous_action,
                    previous_previous_action=previous_previous_action,
                    phi=phi,
                    scalar_reward=scalar_reward,
                    velocity_term=velocity_term,
                    ctrl_term=ctrl_term,
                    cumulative_return=episode_return,
                    max_action=max_action,
                    q_before=q_before,
                    q_after=q_after,
                    q_improvement=q_improvement,
                )

            previous_previous_action = None if previous_action is None else previous_action.copy()
            previous_action = final_action.copy()

            state = next_state

            if terminated or truncated:
                break

        # NEW: Compute the spurious flipped-basin indicator for this episode.
        spurious_basin_episode_results.append(
            compute_spurious_basin_indicator_for_episode(
                rooty_angles=episode_rooty_angles,
                q_before_values=episode_q_before_values,
                q_after_values=episode_q_after_values,
            )
        )

        episode_returns.append(float(episode_return))

    env.close()

    if render:
        if render_path is None:
            raise ValueError("render_path must be provided when render=True.")

        render_path = Path(render_path)
        render_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(render_path, frames, fps=render_fps)
        print(f"Saved render video to: {render_path}")

    states_np = np.asarray(all_states, dtype=np.float64)
    actor_actions_np = np.asarray(all_actor_actions, dtype=np.float64)
    final_actions_np = np.asarray(all_final_actions, dtype=np.float64)

    velocities_np = np.asarray(velocities, dtype=np.float64)
    ctrl_np = np.asarray(ctrl_rewards, dtype=np.float64)
    rewards_np = np.asarray(scalar_rewards, dtype=np.float64)
    velocity_terms_np = np.asarray(velocity_terms, dtype=np.float64)
    ctrl_terms_np = np.asarray(ctrl_terms, dtype=np.float64)

    mean_abs_velocity_term = np.mean(np.abs(velocity_terms_np))
    mean_abs_ctrl_term = np.mean(np.abs(ctrl_terms_np))
    total_abs_terms = mean_abs_velocity_term + mean_abs_ctrl_term + 1e-8

    results = {
        "episodes": int(episodes),
        "num_steps_collected": int(len(final_actions_np)),
        "episode_return_mean": float(np.mean(episode_returns)),
        "episode_return_std": float(np.std(episode_returns)),
        "episode_returns": [float(x) for x in episode_returns],

        "mean_velocity": float(np.mean(velocities_np)),
        "mean_ctrl_reward": float(np.mean(ctrl_np)),
        "mean_scalar_reward": float(np.mean(rewards_np)),

        "mean_abs_velocity_term": float(mean_abs_velocity_term),
        "mean_abs_ctrl_term": float(mean_abs_ctrl_term),
        "velocity_term_abs_fraction": float(mean_abs_velocity_term / total_abs_terms),
        "ctrl_term_abs_fraction": float(mean_abs_ctrl_term / total_abs_terms),

        "timeseries_episode": int(timeseries_episode),
        "timeseries_saved": bool(save_timeseries),
    }

    results.update(summarize_array("velocity", velocities_np))
    results.update(summarize_array("ctrl_reward", ctrl_np))
    results.update(summarize_array("scalar_reward", rewards_np))
    results.update(summarize_array("velocity_reward_term", velocity_terms_np))
    results.update(summarize_array("ctrl_reward_term", ctrl_terms_np))

    actor_action_diag = compute_action_diagnostics(actor_actions_np, max_action)
    final_action_diag = compute_action_diagnostics(final_actions_np, max_action)

    results["actor_action_diagnostics"] = actor_action_diag
    results["final_action_diagnostics"] = final_action_diag

    action_shift_from_actor = np.linalg.norm(final_actions_np - actor_actions_np, axis=1)
    results.update(summarize_array("final_minus_actor_action_norm", action_shift_from_actor))

    if len(q_before_values) > 0:
        q_before_np = np.asarray(q_before_values, dtype=np.float64)
        q_after_np = np.asarray(q_after_values, dtype=np.float64)
        q_improvement_np = np.asarray(q_improvements, dtype=np.float64)
        action_shift_np = np.asarray(action_shifts, dtype=np.float64)

        results.update(summarize_array("predicted_q_before", q_before_np))
        results.update(summarize_array("predicted_q_after", q_after_np))
        results.update(summarize_array("predicted_q_improvement", q_improvement_np))
        results.update(summarize_array("optimization_action_shift", action_shift_np))
    else:
        results["predicted_q_before_mean"] = None
        results["predicted_q_after_mean"] = None
        results["predicted_q_improvement_mean"] = None
        results["optimization_action_shift_mean"] = None

    results["state_diagnostics"] = compute_state_diagnostics(states_np)

    # NEW: Aggregate the spurious flipped-basin indicator over all episodes.
    results["spurious_basin_diagnostics"] = aggregate_spurious_basin_indicators(
        spurious_basin_episode_results
    )

    if save_timeseries:
        results["timeseries"] = selected_timeseries

    return results


def print_summary(results, mode):
    """Print compact diagnostic summary."""
    print("\n" + "=" * 80)
    print(f"Rollout dynamics diagnostics: {mode}")
    print("=" * 80)

    print(f"Episode return: {results['episode_return_mean']:.2f} ± {results['episode_return_std']:.2f}")
    print(f"Mean velocity: {results['mean_velocity']:.4f}")
    print(f"Mean ctrl reward: {results['mean_ctrl_reward']:.4f}")

    print("\nReward contribution fractions:")
    print(f"Velocity: {100.0 * results['velocity_term_abs_fraction']:.2f}%")
    print(f"Control:  {100.0 * results['ctrl_term_abs_fraction']:.2f}%")

    final_diag = results["final_action_diagnostics"]

    print("\nFinal action diagnostics:")
    print(f"Mean ||a||: {final_diag['action_norm_mean']:.4f}")
    print(f"Max  ||a||: {final_diag['action_norm_max']:.4f}")
    print(f"Saturation fraction: {100.0 * final_diag['action_saturation_fraction']:.2f}%")
    print(f"Mean ||Δa||: {final_diag['delta_action_norm_mean']}")
    print(f"Mean jerk: {final_diag['action_jerk_norm_mean']}")
    print(f"Sign flip rate: {final_diag['action_sign_flip_rate']}")

    if results["predicted_q_before_mean"] is not None:
        print("\nOptimization diagnostics:")
        print(f"Predicted Q before: {results['predicted_q_before_mean']:.4f}")
        print(f"Predicted Q after:  {results['predicted_q_after_mean']:.4f}")
        print(f"Predicted Q improvement: {results['predicted_q_improvement_mean']:.4f}")
        print(f"Mean ||a_final - a_actor||: {results['final_minus_actor_action_norm_mean']:.4f}")

    # NEW: Print compact spurious flipped-basin diagnostics.
    basin_diag = results.get("spurious_basin_diagnostics", {})
    if basin_diag:
        print("\nSpurious flipped-basin diagnostics:")
        print(f"Sustained flip rate: {basin_diag.get('sustained_flip_rate')}")
        print(
            "Spurious basin indicator: "
            f"{basin_diag.get('spurious_basin_indicator_mean')} ± "
            f"{basin_diag.get('spurious_basin_indicator_std')}"
        )
        print(f"Mean flip fraction: {basin_diag.get('flip_fraction_mean')}")
        print(f"Mean Q_after jump norm: {basin_diag.get('q_after_jump_norm_at_flip_mean')}")

    if results.get("timeseries_saved", False):
        print("\nTimeseries:")
        print(f"Saved timeseries for episode index: {results['timeseries_episode']}")
        if "timeseries" in results and results["timeseries"] is not None:
            print(f"Timeseries length: {len(results['timeseries']['t'])}")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, required=True)

    #################################################
    # modifica
    parser.add_argument(
        "--gamma",
        type=str,
        required=True,
        help="Gamma associato alla run.",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        choices=["sf", "ddpg"],
        required=True,
        help="Tipo di modello usato dalla diagnostica.",
    )
    ################################################

    parser.add_argument("--env_name", type=str, default="HalfCheetah-v5")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "sf_actor_only",
            "sf_action_optimization",
            "ddpg_actor_only",
            "ddpg_q_action_optimization",
        ],
    )
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--task", type=str, choices=["forward", "backward"], default="backward")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--opt_steps", type=int, default=250)
    parser.add_argument("--opt_step_size", type=float, default=0.2)
    parser.add_argument("--action_l2", type=float, default=1e-3)

    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render_fps", type=int, default=30)

    parser.add_argument(
        "--save_timeseries",
        action="store_true",
        help="If set, save per-timestep diagnostics for one selected episode.",
    )
    parser.add_argument(
        "--timeseries_episode",
        type=int,
        default=0,
        help="Episode index for timeseries/render synchronization.",
    )

    parser.add_argument("--output_name", type=str, default=None)

    args = parser.parse_args()

    ###################################################
    # modifica
    expected_model_type = (
        "sf"
        if args.mode.startswith("sf_")
        else "ddpg"
    )

    if args.model_type != expected_model_type:
        parser.error(
            f"La modalità '{args.mode}' richiede "
            f"--model_type {expected_model_type}, "
            f"non '{args.model_type}'."
        )
    ####################################################

    run_dir = Path("artifacts") / args.run_name

    ##################################################
    # modifica
    diagnostics_dir = build_diagnostics_dir(
        run_dir=run_dir,
        gamma=args.gamma,
        phase=args.phase,
        mode=args.mode,
        model_type=args.model_type,
        task=args.task,
    )
    ####################################################

    w_forward, w_backward = make_task_weights(device)
    task_weights = w_forward if args.task == "forward" else w_backward

    sf_actor, sf_critic, q_actor, q_critic, max_action = load_models(
        run_dir=run_dir,
        env_name=args.env_name,
        phase=args.phase,
    )

    actor, sf_critic_for_mode, q_critic_for_mode, internal_mode = select_models_for_mode(
        mode=args.mode,
        sf_actor=sf_actor,
        sf_critic=sf_critic,
        q_actor=q_actor,
        q_critic=q_critic,
    )

    if args.output_name is None:
        suffix = (
            "timeseries"
            if (args.save_timeseries or args.render)
            else "summary"
        )
        output_name = f"rollout_dynamics_{suffix}.json"
    else:
        output_name = args.output_name

    output_path = diagnostics_dir / output_name

    render_path = None
    if args.render:
        render_path = diagnostics_dir / (
            f"rollout_dynamics_episode_{args.timeseries_episode}.mp4"
        )

    # If rendering, save the matching timeseries automatically.
    save_timeseries = args.save_timeseries or args.render

    results = diagnose_rollout_dynamics(
        env_name=args.env_name,
        actor=actor,
        task_weights=task_weights,
        max_action=max_action,
        mode=internal_mode,
        sf_critic=sf_critic_for_mode,
        q_critic=q_critic_for_mode,
        episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
        opt_steps=args.opt_steps,
        opt_step_size=args.opt_step_size,
        action_l2=args.action_l2,
        render=args.render,
        render_path=render_path,
        render_fps=args.render_fps,
        save_timeseries=save_timeseries,
        timeseries_episode=args.timeseries_episode,
    )

    print_summary(results, args.mode)

    ###########################
    # modifica
    results["gamma"] = float(args.gamma)
    results["phase"] = int(args.phase)
    results["mode"] = args.mode
    results["model_type"] = args.model_type
    results["task"] = args.task
    results["run_name"] = args.run_name
    #################################

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved diagnostics to: {output_path}")


if __name__ == "__main__":
    main()