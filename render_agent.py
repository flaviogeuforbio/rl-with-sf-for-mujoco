import os
#os.environ["MUJOCO_GL"] = "egl" # decomment this line if you want to use EGL rendering (works on headless servers like kaggle or Colab or Leonardo, but not on Windows local machines)

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import imageio.v2 as imageio

from ActorCritic import Actor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_actor(run_dir, env_name, model_type, phase):
    """Load an actor saved after a given training phase."""

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    env.close()

    # In the current minimal setup, both SF-DDPG and DDPG actors have the same architecture.
    actor = Actor(state_dim, action_dim, max_action).to(device)

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

    parser.add_argument("--run_dir", type=str, required=True, help="Directory where the trained model is saved.")
    parser.add_argument("--env_name", type=str, default="HalfCheetah-v5", help="Name of the Gym environment to render, e.g. HalfCheetah-v5 or Walker2d-v5.")
    parser.add_argument("--model_type", type=str, choices=["sf", "ddpg"], required=True, help="Type of the model to render.")
    parser.add_argument("--phase", type=int, default=0, help="Phase of the training process (0 or 1).")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video.")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum number of steps to render.")
    parser.add_argument("--output", type=str, default=None, help="Output path for the rendered video.")

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