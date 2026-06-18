import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from ActorCritic import Actor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_task_weights(task_name):
    """Return handcrafted task weights for forward or backward HalfCheetah."""
    if task_name == "forward":
        return np.array([1.0, 1.0], dtype=np.float32)

    if task_name == "backward":
        return np.array([-1.0, 1.0], dtype=np.float32)

    raise ValueError("task_name must be either 'forward' or 'backward'.")


def load_actor(run_dir, env_name, model_type, phase):
    """Load a trained actor from the artifact directory."""
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    env.close()

    actor = Actor(state_dim, action_dim, max_action).to(device)

    if model_type == "sf":
        actor_path = run_dir / f"sf_actor_{phase}.pth"
    elif model_type == "ddpg":
        actor_path = run_dir / f"q_actor_{phase}.pth"
    else:
        raise ValueError("model_type must be either 'sf' or 'ddpg'.")

    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()

    return actor


def summarize_array(name, values):
    """Compute basic summary statistics for a 1D array."""
    values = np.asarray(values, dtype=np.float64)

    return {
        f"{name}_mean": float(np.mean(values)),
        f"{name}_std": float(np.std(values)),
        f"{name}_min": float(np.min(values)),
        f"{name}_max": float(np.max(values)),
        f"{name}_mean_abs": float(np.mean(np.abs(values))),
        f"{name}_median_abs": float(np.median(np.abs(values))),
    }


def diagnose_feature_scales(
    env_name,
    actor,
    task_weights,
    episodes=5,
    max_episode_steps=1000,
    deterministic=True,
):
    """
    Run rollouts with a trained actor and measure the empirical scale of:
        phi_1 = x_velocity
        phi_2 = reward_ctrl

    The scalar reward is computed as:
        r = phi^T w
    """

    env = gym.make(env_name)

    velocities = []
    ctrl_rewards = []
    scalar_rewards = []
    velocity_terms = []
    ctrl_terms = []
    action_norms = []

    episode_returns = []

    for _ in range(episodes):
        state, _ = env.reset()
        episode_return = 0.0

        for _ in range(max_episode_steps):
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            with torch.no_grad():
                action = actor(state_tensor).cpu().numpy()[0]

            next_state, _, terminated, truncated, info = env.step(action)

            velocity = float(info.get("x_velocity", 0.0))
            ctrl_reward = float(info.get("reward_ctrl", 0.0))

            phi = np.array([velocity, ctrl_reward], dtype=np.float32)

            velocity_term = float(phi[0] * task_weights[0])
            ctrl_term = float(phi[1] * task_weights[1])
            scalar_reward = float(phi @ task_weights)

            velocities.append(velocity)
            ctrl_rewards.append(ctrl_reward)
            velocity_terms.append(velocity_term)
            ctrl_terms.append(ctrl_term)
            scalar_rewards.append(scalar_reward)
            action_norms.append(float(np.linalg.norm(action)))

            episode_return += scalar_reward
            state = next_state

            if terminated or truncated:
                break

        episode_returns.append(episode_return)

    env.close()

    velocities_np = np.asarray(velocities, dtype=np.float64)
    ctrl_np = np.asarray(ctrl_rewards, dtype=np.float64)
    velocity_terms_np = np.asarray(velocity_terms, dtype=np.float64)
    ctrl_terms_np = np.asarray(ctrl_terms, dtype=np.float64)
    scalar_rewards_np = np.asarray(scalar_rewards, dtype=np.float64)
    action_norms_np = np.asarray(action_norms, dtype=np.float64)

    mean_abs_velocity = np.mean(np.abs(velocities_np))
    mean_abs_ctrl = np.mean(np.abs(ctrl_np))

    mean_abs_velocity_term = np.mean(np.abs(velocity_terms_np))
    mean_abs_ctrl_term = np.mean(np.abs(ctrl_terms_np))

    feature_scale_ratio = mean_abs_velocity / (mean_abs_ctrl + 1e-8)
    reward_term_ratio = mean_abs_velocity_term / (mean_abs_ctrl_term + 1e-8)

    total_abs_reward_terms = mean_abs_velocity_term + mean_abs_ctrl_term + 1e-8

    summary = {}

    summary.update(summarize_array("velocity", velocities_np))
    summary.update(summarize_array("ctrl_reward", ctrl_np))
    summary.update(summarize_array("velocity_reward_term", velocity_terms_np))
    summary.update(summarize_array("ctrl_reward_term", ctrl_terms_np))
    summary.update(summarize_array("scalar_reward", scalar_rewards_np))
    summary.update(summarize_array("action_norm", action_norms_np))

    summary["episodes"] = int(episodes)
    summary["num_steps_collected"] = int(len(velocities))
    summary["episode_return_mean"] = float(np.mean(episode_returns))
    summary["episode_return_std"] = float(np.std(episode_returns))
    summary["episode_returns"] = [float(x) for x in episode_returns]

    summary["mean_abs_velocity_over_mean_abs_ctrl_reward"] = float(feature_scale_ratio)
    summary["mean_abs_velocity_term_over_mean_abs_ctrl_term"] = float(reward_term_ratio)

    summary["velocity_term_abs_fraction"] = float(mean_abs_velocity_term / total_abs_reward_terms)
    summary["ctrl_term_abs_fraction"] = float(mean_abs_ctrl_term / total_abs_reward_terms)

    return summary


def print_summary(summary):
    """Pretty-print the most important diagnostic numbers."""
    print("\n" + "=" * 80)
    print("Feature scale diagnostics")
    print("=" * 80)

    print(f"Collected steps: {summary['num_steps_collected']}")
    print(f"Episode return: {summary['episode_return_mean']:.2f} ± {summary['episode_return_std']:.2f}")

    print("\nRaw feature scales:")
    print(
        "velocity mean/std/min/max: "
        f"{summary['velocity_mean']:.4f} / "
        f"{summary['velocity_std']:.4f} / "
        f"{summary['velocity_min']:.4f} / "
        f"{summary['velocity_max']:.4f}"
    )
    print(
        "ctrl_reward mean/std/min/max: "
        f"{summary['ctrl_reward_mean']:.4f} / "
        f"{summary['ctrl_reward_std']:.4f} / "
        f"{summary['ctrl_reward_min']:.4f} / "
        f"{summary['ctrl_reward_max']:.4f}"
    )

    print("\nMean absolute feature scale:")
    print(f"mean |velocity|: {summary['velocity_mean_abs']:.4f}")
    print(f"mean |ctrl_reward|: {summary['ctrl_reward_mean_abs']:.4f}")
    print(
        "|velocity| / |ctrl_reward| ratio: "
        f"{summary['mean_abs_velocity_over_mean_abs_ctrl_reward']:.4f}"
    )

    print("\nReward-term scale:")
    print(f"mean |velocity contribution|: {summary['velocity_reward_term_mean_abs']:.4f}")
    print(f"mean |control contribution|: {summary['ctrl_reward_term_mean_abs']:.4f}")
    print(
        "|velocity term| / |control term| ratio: "
        f"{summary['mean_abs_velocity_term_over_mean_abs_ctrl_term']:.4f}"
    )

    print("\nAbsolute reward-term fractions:")
    print(f"velocity fraction: {100.0 * summary['velocity_term_abs_fraction']:.2f}%")
    print(f"control fraction: {100.0 * summary['ctrl_term_abs_fraction']:.2f}%")

    print("\nAction norm:")
    print(
        "action_norm mean/std/min/max: "
        f"{summary['action_norm_mean']:.4f} / "
        f"{summary['action_norm_std']:.4f} / "
        f"{summary['action_norm_min']:.4f} / "
        f"{summary['action_norm_max']:.4f}"
    )

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--env_name", type=str, default="HalfCheetah-v5")
    parser.add_argument("--model_type", type=str, choices=["sf", "ddpg"], required=True)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--task", type=str, choices=["forward", "backward"], default="forward")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--output_name", type=str, default=None)

    ######################################
    #modifica
    parser.add_argument(
    "--gamma",
    type=str,
    required=True,
    help="Gamma associato alla run.",
)

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="Modalità associata alla diagnostica.",
    )

    ######################################

    args = parser.parse_args()

    run_dir = Path("artifacts") / args.run_name

    actor = load_actor(
        run_dir=run_dir,
        env_name=args.env_name,
        model_type=args.model_type,
        phase=args.phase,
    )

    task_weights = make_task_weights(args.task)

    summary = diagnose_feature_scales(
        env_name=args.env_name,
        actor=actor,
        task_weights=task_weights,
        episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
    )

    print_summary(summary)

    ############################################
    #modifica

    def sanitize_folder_value(value):
        return (
            str(value)
            .strip()
            .replace(".", "_")
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )
    
    gamma_name = sanitize_folder_value(args.gamma)
    mode_name = sanitize_folder_value(args.mode)
    model_name = sanitize_folder_value(args.model_type)
    task_name = sanitize_folder_value(args.task)

    diagnostic_folder_name = (
        f"gamma_{gamma_name}"
        f"__phase_{args.phase}"
        f"__mode_{mode_name}"
        f"__model_{model_name}"
        f"__task_{task_name}"
    )

    diagnostics_dir = run_dir / "diagnostics" / diagnostic_folder_name
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    if args.output_name is None:
        output_name = "feature_scale_diagnostics.json"
    else:
        output_name = args.output_name

    summary["gamma"] = float(args.gamma)
    summary["phase"] = int(args.phase)
    summary["mode"] = args.mode
    summary["model_type"] = args.model_type
    summary["task"] = args.task
    summary["run_name"] = args.run_name

    output_path = diagnostics_dir / output_name
    

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Saved diagnostics to: {output_path}")


if __name__ == "__main__":
    main()