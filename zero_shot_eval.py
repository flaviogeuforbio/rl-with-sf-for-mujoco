import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from ActorCritic import Actor, SFCritic, QCritic


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_task_weights(device):
    """Create the forward and backward task vectors used in training."""
    w_forward = torch.tensor([[1.0], [1.0]], dtype=torch.float32, device=device)
    w_backward = torch.tensor([[-1.0], [1.0]], dtype=torch.float32, device=device)

    return w_forward, w_backward


def compute_scalar_reward(info, task_weights):
    """Compute the scalar reward from HalfCheetah info and task weights."""
    velocity = info.get("x_velocity", 0.0)
    ctrl_reward = info.get("reward_ctrl", 0.0)

    phi = np.array([velocity, ctrl_reward], dtype=np.float32)
    w_np = task_weights.detach().cpu().numpy().flatten()

    return float(phi @ w_np)


def optimize_action_with_sf(
    sf_critic,
    state_tensor,
    init_action,
    task_weights,
    max_action,
    n_steps=20,
    step_size=0.05,
    action_l2=1e-3,
):
    """
    Optimize the action by gradient ascent on:
        Q_w(s, a) = psi(s, a)^T w

    This does not update actor or critic parameters.
    Only the action tensor is optimized at evaluation time.
    """

    action = init_action.clone().detach().requires_grad_(True)

    for _ in range(n_steps):
        psi = sf_critic(state_tensor, action)
        q_value = torch.matmul(psi, task_weights)

        # Penalize overly large actions to reduce immediate saturation.
        action_penalty = action_l2 * (action ** 2).mean()

        # Minimize negative objective = maximize Q minus action penalty.
        loss = -(q_value.mean() - action_penalty)

        if action.grad is not None:
            action.grad.zero_()

        loss.backward()

        with torch.no_grad():
            # Gradient descent on -Q is gradient ascent on Q.
            action -= step_size * action.grad
            action.clamp_(-max_action, max_action)

        action = action.detach().requires_grad_(True)

    return action.detach()


def optimize_action_with_qcritic(
    q_critic,
    state_tensor,
    init_action,
    max_action,
    n_steps=20,
    step_size=0.05,
    action_l2=1e-3,
):
    """
    Optimize the action by gradient ascent on the DDPG scalar Q critic.

    Important: this is not zero-shot reward transfer.
    The DDPG critic is tied to the reward used during its own training.
    """

    action = init_action.clone().detach().requires_grad_(True)

    for _ in range(n_steps):
        q_value = q_critic(state_tensor, action)

        # Penalize overly large actions to reduce immediate saturation.
        action_penalty = action_l2 * (action ** 2).mean()

        loss = -(q_value.mean() - action_penalty)

        if action.grad is not None:
            action.grad.zero_()

        loss.backward()

        with torch.no_grad():
            action -= step_size * action.grad
            action.clamp_(-max_action, max_action)

        action = action.detach().requires_grad_(True)

    return action.detach()


def evaluate_policy(
    env_name,
    actor,
    task_weights,
    max_action,
    mode,
    sf_critic=None,
    q_critic=None,
    episodes=10,
    max_episode_steps=1000,
    opt_steps=20,
    opt_step_size=0.05,
    action_l2=1e-3,
):
    """
    Evaluate one policy mode on a given task.

    Supported modes:
        - "actor_only"
        - "sf_action_optimization"
        - "q_action_optimization"
    """

    env = gym.make(env_name)
    returns = []

    actor.eval()

    if sf_critic is not None:
        sf_critic.eval()

    if q_critic is not None:
        q_critic.eval()

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
                init_action = actor(state_tensor)

            if mode == "actor_only":
                action_tensor = init_action

            elif mode == "sf_action_optimization":
                if sf_critic is None:
                    raise ValueError("sf_critic must be provided for SF action optimization.")

                action_tensor = optimize_action_with_sf(
                    sf_critic=sf_critic,
                    state_tensor=state_tensor,
                    init_action=init_action,
                    task_weights=task_weights,
                    max_action=max_action,
                    n_steps=opt_steps,
                    step_size=opt_step_size,
                    action_l2=action_l2,
                )

            elif mode == "q_action_optimization":
                if q_critic is None:
                    raise ValueError("q_critic must be provided for Q action optimization.")

                action_tensor = optimize_action_with_qcritic(
                    q_critic=q_critic,
                    state_tensor=state_tensor,
                    init_action=init_action,
                    max_action=max_action,
                    n_steps=opt_steps,
                    step_size=opt_step_size,
                    action_l2=action_l2,
                )

            else:
                raise ValueError(f"Unknown mode: {mode}")

            action = action_tensor.detach().cpu().numpy()[0]
            action = np.clip(action, -max_action, max_action)

            next_state, _, terminated, truncated, info = env.step(action)

            reward = compute_scalar_reward(info, task_weights)
            episode_return += reward

            state = next_state

            if terminated or truncated:
                break

        returns.append(episode_return)

    env.close()

    return {
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "episodes": [float(x) for x in returns],
    }


def load_models(run_dir, env_name, phase):
    """Load SF-DDPG and DDPG models from a saved artifact directory."""

    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    feature_dim = 2

    env.close()

    # SF-DDPG models.
    sf_actor = Actor(state_dim, action_dim, max_action).to(device)
    sf_critic = SFCritic(state_dim, action_dim, feature_dim).to(device)

    sf_actor.load_state_dict(
        torch.load(run_dir / f"sf_actor_{phase}.pth", map_location=device)
    )
    sf_critic.load_state_dict(
        torch.load(run_dir / f"sf_critic_{phase}.pth", map_location=device)
    )

    # Standard DDPG models.
    q_actor = Actor(state_dim, action_dim, max_action).to(device)
    q_critic = QCritic(state_dim, action_dim).to(device)

    q_actor.load_state_dict(
        torch.load(run_dir / f"q_actor_{phase}.pth", map_location=device)
    )
    q_critic.load_state_dict(
        torch.load(run_dir / f"q_critic_{phase}.pth", map_location=device)
    )

    return sf_actor, sf_critic, q_actor, q_critic, max_action


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--env_name", type=str, default="HalfCheetah-v5")
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--opt_steps", type=int, default=20)
    parser.add_argument("--opt_step_size", type=float, default=0.05)
    parser.add_argument("--action_l2", type=float, default=1e-3)
    parser.add_argument("--output_name", type=str, default="zero_shot_eval_results.json")

    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    w_forward, w_backward = make_task_weights(device)

    sf_actor, sf_critic, q_actor, q_critic, max_action = load_models(
        run_dir=run_dir,
        env_name=args.env_name,
        phase=args.phase,
    )

    # The key zero-shot test is backward evaluation after phase 0,
    # i.e. after training only on the forward task.
    eval_task_weights = w_backward

    results = {}

    print("=" * 80)
    print("Zero-shot evaluation on backward task")
    print(f"Run directory: {run_dir}")
    print(f"Loaded phase: {args.phase}")
    print("=" * 80)

    print("Evaluating SF actor only...")
    results["sf_actor_only"] = evaluate_policy(
        env_name=args.env_name,
        actor=sf_actor,
        task_weights=eval_task_weights,
        max_action=max_action,
        mode="actor_only",
        episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
    )

    print("Evaluating SF actor + action-space gradient ascent...")
    results["sf_action_optimization"] = evaluate_policy(
        env_name=args.env_name,
        actor=sf_actor,
        sf_critic=sf_critic,
        task_weights=eval_task_weights,
        max_action=max_action,
        mode="sf_action_optimization",
        episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
        opt_steps=args.opt_steps,
        opt_step_size=args.opt_step_size,
        action_l2=args.action_l2,
    )

    print("Evaluating DDPG actor only...")
    results["ddpg_actor_only"] = evaluate_policy(
        env_name=args.env_name,
        actor=q_actor,
        task_weights=eval_task_weights,
        max_action=max_action,
        mode="actor_only",
        episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
    )

    print("Evaluating DDPG actor + Q-gradient ascent...")
    results["ddpg_q_action_optimization"] = evaluate_policy(
        env_name=args.env_name,
        actor=q_actor,
        q_critic=q_critic,
        task_weights=eval_task_weights,
        max_action=max_action,
        mode="q_action_optimization",
        episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
        opt_steps=args.opt_steps,
        opt_step_size=args.opt_step_size,
        action_l2=args.action_l2,
    )

    print("\nResults:")
    for key, value in results.items():
        print(f"{key}: {value['mean']:.2f} ± {value['std']:.2f}")

    output_path = run_dir / args.output_name

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()