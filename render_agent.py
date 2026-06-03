import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import imageio.v2 as imageio

from ActorCritic import QActor, QCritic, SFCritic


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_task_weights(device):
    """Create the forward and backward task vectors."""
    w_forward = torch.tensor([[1.0], [1.0]], dtype=torch.float32, device=device)
    w_backward = torch.tensor([[-1.0], [1.0]], dtype=torch.float32, device=device)
    return w_forward, w_backward


def load_actor(run_dir, env_name, model_type, phase):
    """Load an actor saved after a given training phase."""

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    env.close()

    # In the current minimal setup, both SF-DDPG and DDPG actors have the same architecture.
    actor = QActor(state_dim, action_dim, max_action).to(device)

    if model_type == "sf":
        actor_path = run_dir / f"sf_actor_{phase}.pth"
    elif model_type == "ddpg":
        actor_path = run_dir / f"q_actor_{phase}.pth"
    else:
        raise ValueError("model_type must be either 'sf' or 'ddpg'.")

    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()

    return actor, max_action


def render_actor_policy(
    env_name,
    actor,
    output_path,
    fps=30,
    max_steps=1000,
):
    """Render one episode using actor(state) -> action and save it as an mp4 video."""

    env = gym.make(env_name, render_mode="rgb_array")

    frames = []

    state, _ = env.reset()

    for _ in range(max_steps):
        frame = env.render()
        frames.append(frame)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        with torch.no_grad():
            action = actor(state_tensor).cpu().numpy()[0]

        next_state, _, terminated, truncated, _ = env.step(action)

        state = next_state

        if terminated or truncated:
            break

    env.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)

    print(f"Saved video to: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--env_name", type=str, default="HalfCheetah-v5")
    parser.add_argument("--model_type", type=str, choices=["sf", "ddpg"], required=True)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    actor, _ = load_actor(
        run_dir=run_dir,
        env_name=args.env_name,
        model_type=args.model_type,
        phase=args.phase,
    )

    if args.output is None:
        output_path = run_dir / f"render_{args.model_type}_phase_{args.phase}.mp4"
    else:
        output_path = Path(args.output)

    render_actor_policy(
        env_name=args.env_name,
        actor=actor,
        output_path=output_path,
        fps=args.fps,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()