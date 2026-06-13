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


def summarize_array(name, values):
    """Compute basic summary statistics for a one-dimensional array."""
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


def make_task_weights(eval_task, device):
    """Create task weights for forward or backward evaluation."""
    if eval_task == "forward":
        return torch.tensor([[1.0], [1.0]], dtype=torch.float32, device=device)

    if eval_task == "backward":
        return torch.tensor([[-1.0], [1.0]], dtype=torch.float32, device=device)

    raise ValueError("eval_task must be either 'forward' or 'backward'.")


def optimize_action_with_sf(
    sf_critic,
    state_tensor,
    init_action,
    task_weights,
    max_action,
    n_steps=250,
    step_size=0.2,
    action_l2=1e-3,
):
    """
    Optimize the action by gradient ascent on:
        Q_w(s, a) = psi(s, a)^T w

    This is used only for diagnostic rollouts.
    Network parameters are never updated.
    """

    action = init_action.clone().detach().requires_grad_(True)

    for _ in range(n_steps):
        psi = sf_critic(state_tensor, action)
        q_value = torch.matmul(psi, task_weights)

        # Penalize large actions to reduce immediate torque saturation.
        action_penalty = action_l2 * (action ** 2).mean()

        # Minimize negative objective = maximize Q minus action penalty.
        loss = -(q_value.mean() - action_penalty)

        if action.grad is not None:
            action.grad.zero_()

        loss.backward()

        with torch.no_grad():
            action -= step_size * action.grad
            action.clamp_(-max_action, max_action)

        action = action.detach().requires_grad_(True)

    return action.detach()


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

    return actor, critic, max_action


def collect_rollouts(
    env_name,
    actor,
    critic,
    task_weights,
    max_action,
    episodes,
    max_episode_steps,
    policy_mode="actor_only",
    opt_steps=250,
    opt_step_size=0.2,
    action_l2=1e-3,
):
    """
    Collect states, actions and handcrafted features using either:
        - actor_only
        - sf_action_optimization

    The second mode is important because it tests whether psi remains accurate
    on the actions selected by zero-shot action optimization.
    """

    env = gym.make(env_name)

    all_states = []
    all_actions = []
    all_actor_actions = []
    all_phis = []

    episode_returns_forward = []
    episode_returns_backward = []

    action_shift_from_actor = []

    for _ in range(episodes):
        state, _ = env.reset()

        states = []
        actions = []
        actor_actions = []
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
                actor_action_tensor = actor(state_tensor)

            if policy_mode == "actor_only":
                action_tensor = actor_action_tensor

            elif policy_mode == "sf_action_optimization":
                action_tensor = optimize_action_with_sf(
                    sf_critic=critic,
                    state_tensor=state_tensor,
                    init_action=actor_action_tensor,
                    task_weights=task_weights,
                    max_action=max_action,
                    n_steps=opt_steps,
                    step_size=opt_step_size,
                    action_l2=action_l2,
                )

            else:
                raise ValueError(
                    "policy_mode must be either 'actor_only' or 'sf_action_optimization'."
                )

            with torch.no_grad():
                shift = torch.norm(action_tensor - actor_action_tensor).item()

            actor_action = actor_action_tensor.detach().cpu().numpy()[0]
            action = action_tensor.detach().cpu().numpy()[0]
            action = np.clip(action, -max_action, max_action)

            next_state, _, terminated, truncated, info = env.step(action)

            velocity = float(info.get("x_velocity", 0.0))
            ctrl_reward = float(info.get("reward_ctrl", 0.0))

            phi = np.array([velocity, ctrl_reward], dtype=np.float32)

            # Store transition-level data.
            states.append(state.copy())
            actions.append(action.copy())
            actor_actions.append(actor_action.copy())
            phis.append(phi.copy())
            action_shift_from_actor.append(float(shift))

            # Scalar rewards for diagnostic purposes.
            forward_return += velocity + ctrl_reward
            backward_return += -velocity + ctrl_reward

            state = next_state

            if terminated or truncated:
                break

        all_states.append(np.asarray(states, dtype=np.float32))
        all_actions.append(np.asarray(actions, dtype=np.float32))
        all_actor_actions.append(np.asarray(actor_actions, dtype=np.float32))
        all_phis.append(np.asarray(phis, dtype=np.float32))

        episode_returns_forward.append(float(forward_return))
        episode_returns_backward.append(float(backward_return))

    env.close()

    return (
        all_states,
        all_actions,
        all_actor_actions,
        all_phis,
        episode_returns_forward,
        episode_returns_backward,
        action_shift_from_actor,
    )


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


def compute_action_basic_stats(actions, max_action):
    """Compute simple action statistics for the collected policy actions."""
    actions = np.asarray(actions, dtype=np.float64)

    if len(actions) == 0:
        return {}

    action_norms = np.linalg.norm(actions, axis=1)
    saturation_fraction = np.mean(np.abs(actions) > 0.95 * max_action)

    stats = {}
    stats.update(summarize_array("action_norm", action_norms))
    stats["action_saturation_fraction"] = float(saturation_fraction)

    return stats


def analyze_psi(
    run_dir,
    env_name,
    phase,
    episodes,
    max_episode_steps,
    gamma,
    policy_mode="actor_only",
    eval_task="backward",
    opt_steps=250,
    opt_step_size=0.2,
    action_l2=1e-3,
):
    """Main psi decomposition diagnostic."""
    actor, critic, max_action = load_models(
        run_dir=run_dir,
        env_name=env_name,
        phase=phase,
    )

    task_weights_torch = make_task_weights(eval_task, device)
    task_weights_np = task_weights_torch.detach().cpu().numpy().flatten()

    (
        all_states,
        all_actions,
        all_actor_actions,
        all_phis,
        ret_f,
        ret_b,
        action_shift_from_actor,
    ) = collect_rollouts(
        env_name=env_name,
        actor=actor,
        critic=critic,
        task_weights=task_weights_torch,
        max_action=max_action,
        episodes=episodes,
        max_episode_steps=max_episode_steps,
        policy_mode=policy_mode,
        opt_steps=opt_steps,
        opt_step_size=opt_step_size,
        action_l2=action_l2,
    )

    states_flat = []
    actions_flat = []
    actor_actions_flat = []
    phis_flat = []
    feature_returns_flat = []

    for states, actions, actor_actions, phis in zip(
        all_states,
        all_actions,
        all_actor_actions,
        all_phis,
    ):
        feature_returns = discounted_feature_returns(phis, gamma)

        states_flat.append(states)
        actions_flat.append(actions)
        actor_actions_flat.append(actor_actions)
        phis_flat.append(phis)
        feature_returns_flat.append(feature_returns)

    states_flat = np.concatenate(states_flat, axis=0)
    actions_flat = np.concatenate(actions_flat, axis=0)
    actor_actions_flat = np.concatenate(actor_actions_flat, axis=0)
    phis_flat = np.concatenate(phis_flat, axis=0)
    feature_returns_flat = np.concatenate(feature_returns_flat, axis=0)

    psi_pred = predict_psi(
        critic=critic,
        states=states_flat,
        actions=actions_flat,
    )

    error = psi_pred - feature_returns_flat

    # Component-wise diagnostics.
    mse_velocity = float(np.mean(error[:, 0] ** 2))
    mse_ctrl = float(np.mean(error[:, 1] ** 2))

    mae_velocity = float(np.mean(np.abs(error[:, 0])))
    mae_ctrl = float(np.mean(np.abs(error[:, 1])))

    corr_velocity = pearson_corr(psi_pred[:, 0], feature_returns_flat[:, 0])
    corr_ctrl = pearson_corr(psi_pred[:, 1], feature_returns_flat[:, 1])

    mean_cosine = cosine_similarity_matrix_rows(psi_pred, feature_returns_flat)

    # Projected Q diagnostics.
    w_forward = np.array([1.0, 1.0], dtype=np.float32)
    w_backward = np.array([-1.0, 1.0], dtype=np.float32)

    q_pred_forward = psi_pred @ w_forward
    q_true_forward = feature_returns_flat @ w_forward

    q_pred_backward = psi_pred @ w_backward
    q_true_backward = feature_returns_flat @ w_backward

    q_pred_eval = psi_pred @ task_weights_np
    q_true_eval = feature_returns_flat @ task_weights_np

    q_forward_mse = float(np.mean((q_pred_forward - q_true_forward) ** 2))
    q_backward_mse = float(np.mean((q_pred_backward - q_true_backward) ** 2))
    q_eval_mse = float(np.mean((q_pred_eval - q_true_eval) ** 2))

    q_forward_corr = pearson_corr(q_pred_forward, q_true_forward)
    q_backward_corr = pearson_corr(q_pred_backward, q_true_backward)
    q_eval_corr = pearson_corr(q_pred_eval, q_true_eval)

    # Scale ratios.
    true_velocity_abs_mean = np.mean(np.abs(feature_returns_flat[:, 0]))
    true_ctrl_abs_mean = np.mean(np.abs(feature_returns_flat[:, 1]))
    psi_velocity_abs_mean = np.mean(np.abs(psi_pred[:, 0]))
    psi_ctrl_abs_mean = np.mean(np.abs(psi_pred[:, 1]))

    action_shift_arr = np.asarray(action_shift_from_actor, dtype=np.float64)

    results = {
        "run_dir": str(run_dir),
        "phase": int(phase),
        "policy_mode": policy_mode,
        "eval_task": eval_task,
        "episodes": int(episodes),
        "gamma": float(gamma),
        "num_transitions": int(len(states_flat)),

        "opt_steps": int(opt_steps),
        "opt_step_size": float(opt_step_size),
        "action_l2": float(action_l2),

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
        "q_eval_mse": q_eval_mse,
        "q_forward_corr": q_forward_corr,
        "q_backward_corr": q_backward_corr,
        "q_eval_corr": q_eval_corr,

        "psi_velocity_mean": float(np.mean(psi_pred[:, 0])),
        "psi_velocity_std": float(np.std(psi_pred[:, 0])),
        "psi_ctrl_mean": float(np.mean(psi_pred[:, 1])),
        "psi_ctrl_std": float(np.std(psi_pred[:, 1])),

        "true_feature_return_velocity_mean": float(np.mean(feature_returns_flat[:, 0])),
        "true_feature_return_velocity_std": float(np.std(feature_returns_flat[:, 0])),
        "true_feature_return_ctrl_mean": float(np.mean(feature_returns_flat[:, 1])),
        "true_feature_return_ctrl_std": float(np.std(feature_returns_flat[:, 1])),

        "true_abs_velocity_return_over_abs_ctrl_return": float(
            true_velocity_abs_mean / (true_ctrl_abs_mean + 1e-8)
        ),
        "psi_abs_velocity_over_abs_ctrl": float(
            psi_velocity_abs_mean / (psi_ctrl_abs_mean + 1e-8)
        ),
    }

    # Diagnostics for how far optimized actions move away from the actor.
    results.update(summarize_array("action_shift_from_actor", action_shift_arr))

    # Basic action magnitude/saturation diagnostics.
    final_action_stats = compute_action_basic_stats(actions_flat, max_action)
    actor_action_stats = compute_action_basic_stats(actor_actions_flat, max_action)

    results["final_action_stats"] = final_action_stats
    results["actor_action_stats"] = actor_action_stats

    return results


def print_results(results):
    """Print the most important diagnostics."""
    print("\n" + "=" * 80)
    print("PSI DECOMPOSITION DIAGNOSTICS")
    print("=" * 80)

    print(f"Run dir: {results['run_dir']}")
    print(f"Phase: {results['phase']}")
    print(f"Policy mode: {results['policy_mode']}")
    print(f"Eval task: {results['eval_task']}")
    print(f"Transitions: {results['num_transitions']}")

    print("\nRollout returns under diagnostic policy:")
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
    print(f"eval-task Q corr: {results['q_eval_corr']:.4f}")
    print(f"forward  Q MSE:  {results['q_forward_mse']:.4f}")
    print(f"backward Q MSE:  {results['q_backward_mse']:.4f}")
    print(f"eval-task Q MSE: {results['q_eval_mse']:.4f}")

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

    print("\nAction shift from actor:")
    print(
        f"mean: {results['action_shift_from_actor_mean']:.4f}, "
        f"p95: {results['action_shift_from_actor_p95']:.4f}, "
        f"max: {results['action_shift_from_actor_max']:.4f}"
    )

    print("\nFinal action stats:")
    print(
        f"mean ||a||: {results['final_action_stats']['action_norm_mean']:.4f}, "
        f"saturation: "
        f"{100.0 * results['final_action_stats']['action_saturation_fraction']:.2f}%"
    )

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, required=True, help="Name of the run directory containing the saved models.")
    parser.add_argument("--env_name", type=str, default="HalfCheetah-v5")
    parser.add_argument("--phase", type=int, default=0, help="0 for Task 1, 1 for Task 2")
    parser.add_argument("--episodes", type=int, default=5, help="Number of diagnostic episodes to collect.")
    parser.add_argument("--max_episode_steps", type=int, default=1000, help="Maximum steps per diagnostic episode.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for computing feature returns.")

    parser.add_argument(
        "--policy_mode",
        type=str,
        choices=["actor_only", "sf_action_optimization"], # for gamma ablation we need only sf_action_optimization
        default="actor_only",
     help="Whether to use the actor's actions or optimize actions with the SF critic for diagnostics.")
    parser.add_argument(
        "--eval_task",
        type=str,
        choices=["forward", "backward"],
        default="backward",
        help="Whether to evaluate on the forward or backward task for diagnostics.",
    )
    parser.add_argument("--opt_steps", type=int, default=250, help="Number of optimization steps for sf_action_optimization mode.")
    parser.add_argument("--opt_step_size", type=float, default=0.2, help="Step size for optimization in sf_action_optimization mode.")
    parser.add_argument("--action_l2", type=float, default=1e-3, help="L2 regularization strength for actions.")

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
        policy_mode=args.policy_mode,
        eval_task=args.eval_task,
        opt_steps=args.opt_steps,
        opt_step_size=args.opt_step_size,
        action_l2=args.action_l2,
    )

    print_results(results)

    if args.output_name is None:
        output_path = (
            run_dir
            / f"psi_diagnostics_phase_{args.phase}_{args.policy_mode}_{args.eval_task}.json"
        )
    else:
        output_path = run_dir / args.output_name

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved diagnostics to: {output_path}")


if __name__ == "__main__":
    main()