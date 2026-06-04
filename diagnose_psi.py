import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from ActorCritic import Actor, SFCritic


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pearson_corr(x, y):
    """Compute Pearson correlation without scipy."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x = x - x.mean()
    y = y - y.mean()

    denom = np.sqrt((x ** 2).sum()) * np.sqrt((y ** 2).sum())

    if denom < 1e-12:
        return float("nan")

    return float((x * y).sum() / denom)


def cosine_similarity_matrix_rows(x, y):
    """Compute mean row-wise cosine similarity between two matrices."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    num = np.sum(x * y, axis=1)
    den = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)

    valid = den > 1e-12

    if valid.sum() == 0:
        return float("nan")

    return float(np.mean(num[valid] / den[valid]))


def discounted_feature_returns(phis, gamma):
    """
    Compute discounted feature returns:

        G_t = phi_t + gamma * phi_{t+1} + gamma^2 * phi_{t+2} + ...

    phis has shape [T, feature_dim].
    """
    phis = np.asarray(phis, dtype=np.float32)

    returns = np.zeros_like(phis, dtype=np.float32)
    running = np.zeros(phis.shape[1], dtype=np.float32)

    for t in reversed(range(len(phis))):
        running = phis[t] + gamma * running
        returns[t] = running

    return returns


def load_models(run_dir, env_name, phase):
    """Load SF actor and SF critic from a previous run."""
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    feature_dim = 2

    env.close()

    actor = Actor(state_dim, action_dim, max_action).to(device)
    critic = SFCritic(state_dim, action_dim, feature_dim).to(device)

    actor_path = run_dir / f"sf_actor_{phase}.pth"
    critic_path = run_dir / f"sf_critic_{phase}.pth"

    actor.load_state_dict(torch.load(actor_path, map_location=device))
    critic.load_state_dict(torch.load(critic_path, map_location=device))

    actor.eval()
    critic.eval()

    return actor, critic


def collect_rollouts(env_name, actor, episodes, max_episode_steps):
    """
    Collect states, actions and handcrafted features using the trained actor.
    """
    env = gym.make(env_name)

    all_states = []
    all_actions = []
    all_phis = []

    episode_returns_forward = []
    episode_returns_backward = []

    for _ in range(episodes):
        state, _ = env.reset()

        states = []
        actions = []
        phis = []

        forward_return = 0.0
        backward_return = 0.0

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

            # Store transition-level data.
            states.append(state.copy())
            actions.append(action.copy())
            phis.append(phi.copy())

            # Scalar rewards for diagnostic purposes.
            forward_return += velocity + ctrl_reward
            backward_return += -velocity + ctrl_reward

            state = next_state

            if terminated or truncated:
                break

        all_states.append(np.asarray(states, dtype=np.float32))
        all_actions.append(np.asarray(actions, dtype=np.float32))
        all_phis.append(np.asarray(phis, dtype=np.float32))

        episode_returns_forward.append(float(forward_return))
        episode_returns_backward.append(float(backward_return))

    env.close()

    return all_states, all_actions, all_phis, episode_returns_forward, episode_returns_backward


def predict_psi(critic, states, actions, batch_size=4096):
    """Predict psi(s,a) for all collected transitions."""
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)

    preds = []

    for start in range(0, len(states), batch_size):
        end = start + batch_size

        state_batch = torch.tensor(
            states[start:end],
            dtype=torch.float32,
            device=device,
        )

        action_batch = torch.tensor(
            actions[start:end],
            dtype=torch.float32,
            device=device,
        )

        with torch.no_grad():
            psi = critic(state_batch, action_batch).cpu().numpy()

        preds.append(psi)

    return np.concatenate(preds, axis=0)


def analyze_psi(run_dir, env_name, phase, episodes, max_episode_steps, gamma):
    """Main psi decomposition diagnostic."""
    actor, critic = load_models(
        run_dir=run_dir,
        env_name=env_name,
        phase=phase,
    )

    all_states, all_actions, all_phis, ret_f, ret_b = collect_rollouts(
        env_name=env_name,
        actor=actor,
        episodes=episodes,
        max_episode_steps=max_episode_steps,
    )

    states_flat = []
    actions_flat = []
    phis_flat = []
    feature_returns_flat = []

    for states, actions, phis in zip(all_states, all_actions, all_phis):
        feature_returns = discounted_feature_returns(phis, gamma)

        states_flat.append(states)
        actions_flat.append(actions)
        phis_flat.append(phis)
        feature_returns_flat.append(feature_returns)

    states_flat = np.concatenate(states_flat, axis=0)
    actions_flat = np.concatenate(actions_flat, axis=0)
    phis_flat = np.concatenate(phis_flat, axis=0)
    feature_returns_flat = np.concatenate(feature_returns_flat, axis=0)

    psi_pred = predict_psi(
        critic=critic,
        states=states_flat,
        actions=actions_flat,
    )

    error = psi_pred - feature_returns_flat

    # Component-wise metrics.
    mse_velocity = float(np.mean(error[:, 0] ** 2))
    mse_ctrl = float(np.mean(error[:, 1] ** 2))

    mae_velocity = float(np.mean(np.abs(error[:, 0])))
    mae_ctrl = float(np.mean(np.abs(error[:, 1])))

    corr_velocity = pearson_corr(psi_pred[:, 0], feature_returns_flat[:, 0])
    corr_ctrl = pearson_corr(psi_pred[:, 1], feature_returns_flat[:, 1])

    mean_cosine = cosine_similarity_matrix_rows(psi_pred, feature_returns_flat)

    # Scalar projected diagnostics.
    w_forward = np.array([1.0, 1.0], dtype=np.float32)
    w_backward = np.array([-1.0, 1.0], dtype=np.float32)

    q_pred_forward = psi_pred @ w_forward
    q_true_forward = feature_returns_flat @ w_forward

    q_pred_backward = psi_pred @ w_backward
    q_true_backward = feature_returns_flat @ w_backward

    q_forward_mse = float(np.mean((q_pred_forward - q_true_forward) ** 2))
    q_backward_mse = float(np.mean((q_pred_backward - q_true_backward) ** 2))

    q_forward_corr = pearson_corr(q_pred_forward, q_true_forward)
    q_backward_corr = pearson_corr(q_pred_backward, q_true_backward)

    results = {
        "run_dir": str(run_dir),
        "phase": int(phase),
        "episodes": int(episodes),
        "gamma": float(gamma),
        "num_transitions": int(len(states_flat)),

        "episode_forward_return_mean": float(np.mean(ret_f)),
        "episode_forward_return_std": float(np.std(ret_f)),
        "episode_backward_return_mean": float(np.mean(ret_b)),
        "episode_backward_return_std": float(np.std(ret_b)),

        "psi_velocity_mse": mse_velocity,
        "psi_ctrl_mse": mse_ctrl,
        "psi_velocity_mae": mae_velocity,
        "psi_ctrl_mae": mae_ctrl,
        "psi_velocity_corr": corr_velocity,
        "psi_ctrl_corr": corr_ctrl,
        "psi_mean_cosine_similarity": mean_cosine,

        "q_forward_mse": q_forward_mse,
        "q_backward_mse": q_backward_mse,
        "q_forward_corr": q_forward_corr,
        "q_backward_corr": q_backward_corr,

        "psi_velocity_mean": float(np.mean(psi_pred[:, 0])),
        "psi_velocity_std": float(np.std(psi_pred[:, 0])),
        "psi_ctrl_mean": float(np.mean(psi_pred[:, 1])),
        "psi_ctrl_std": float(np.std(psi_pred[:, 1])),

        "true_feature_return_velocity_mean": float(np.mean(feature_returns_flat[:, 0])),
        "true_feature_return_velocity_std": float(np.std(feature_returns_flat[:, 0])),
        "true_feature_return_ctrl_mean": float(np.mean(feature_returns_flat[:, 1])),
        "true_feature_return_ctrl_std": float(np.std(feature_returns_flat[:, 1])),
    }

    return results


def print_results(results):
    """Print the most important diagnostics."""
    print("\n" + "=" * 80)
    print("PSI DECOMPOSITION DIAGNOSTICS")
    print("=" * 80)

    print(f"Run dir: {results['run_dir']}")
    print(f"Phase: {results['phase']}")
    print(f"Transitions: {results['num_transitions']}")

    print("\nRollout returns under loaded actor:")
    print(
        f"Forward return:  {results['episode_forward_return_mean']:.2f} "
        f"± {results['episode_forward_return_std']:.2f}"
    )
    print(
        f"Backward return: {results['episode_backward_return_mean']:.2f} "
        f"± {results['episode_backward_return_std']:.2f}"
    )

    print("\nComponent-wise psi vs true discounted feature returns:")
    print(f"velocity corr: {results['psi_velocity_corr']:.4f}")
    print(f"control  corr: {results['psi_ctrl_corr']:.4f}")
    print(f"velocity MSE:  {results['psi_velocity_mse']:.4f}")
    print(f"control  MSE:  {results['psi_ctrl_mse']:.4f}")
    print(f"mean cosine:   {results['psi_mean_cosine_similarity']:.4f}")

    print("\nProjected Q diagnostics:")
    print(f"forward  Q corr: {results['q_forward_corr']:.4f}")
    print(f"backward Q corr: {results['q_backward_corr']:.4f}")
    print(f"forward  Q MSE:  {results['q_forward_mse']:.4f}")
    print(f"backward Q MSE:  {results['q_backward_mse']:.4f}")

    print("\nPsi scale:")
    print(
        f"psi velocity mean/std: "
        f"{results['psi_velocity_mean']:.4f} / {results['psi_velocity_std']:.4f}"
    )
    print(
        f"psi control  mean/std: "
        f"{results['psi_ctrl_mean']:.4f} / {results['psi_ctrl_std']:.4f}"
    )

    print("\nTrue discounted feature-return scale:")
    print(
        f"true velocity return mean/std: "
        f"{results['true_feature_return_velocity_mean']:.4f} / "
        f"{results['true_feature_return_velocity_std']:.4f}"
    )
    print(
        f"true control return mean/std: "
        f"{results['true_feature_return_ctrl_mean']:.4f} / "
        f"{results['true_feature_return_ctrl_std']:.4f}"
    )

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--env_name", type=str, default="HalfCheetah-v5")
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--output_name", type=str, default=None)

    args = parser.parse_args()

    run_dir = Path("artifacts") / args.run_name

    results = analyze_psi(
        run_dir=run_dir,
        env_name=args.env_name,
        phase=args.phase,
        episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
        gamma=args.gamma,
    )

    print_results(results)

    if args.output_name is None:
        output_path = run_dir / f"psi_diagnostics_phase_{args.phase}.json"
    else:
        output_path = run_dir / args.output_name

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved diagnostics to: {output_path}")


if __name__ == "__main__":
    main()