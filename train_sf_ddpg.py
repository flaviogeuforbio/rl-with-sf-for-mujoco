import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path
import json

# Import from the previously created file
from ActorCritic import Actor, SFCritic, SFtrain_actor, SFtrain_critic, QCritic, Qtrain_actor, Qtrain_critic  
from utils import ReplayBuffer, soft_update, plot_results, evaluate_zero_shot_transfer_learning_sf, set_seed

WARMUP_STEPS = 1000  # Number of initial steps to take random actions for exploration

# ---------------------------------------------------------
# Device Setup
# ---------------------------------------------------------
# We automatically use the GPU if available; otherwise we fall back to the CPU.
# This makes the implementation portable and avoids device mismatch errors.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------
def train_sf_ddpg(
    run_dir: Path | str,
    env_name="HalfCheetah-v5",
    steps_per_phase=50000,
    batch_size=64,
    lambda_q=1.0,
    lambda_vec=0.05,
    seed=42,  # Default seed value for reproducibility if not provided via command line (Homage to Douglas Adams' "Answer to the Ultimate Question of Life, The Universe, and Everything")
    gamma=0.99,  # Discount factor for future rewards/features
    backward_only=False
):

    env = gym.make(env_name)

    #################################################
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    #################################################


    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    max_action = float(env.action_space.high[0])

    feature_dim = 3  

    # ---------------------------------------------------------
    # Initialize Networks
    # ---------------------------------------------------------
    actor = Actor(state_dim, action_dim, max_action).to(device)

    actor_target = deepcopy(actor).to(device)
    # Set target networks to eval mode. This is a no-op for plain MLPs
    # (no BatchNorm or Dropout), but is good practice: it signals clearly
    # that these networks are used for inference only and should never
    # be updated by an optimizer or behave differently at train time.
    actor_target.eval()

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)

    sf_critic = SFCritic(state_dim, action_dim, feature_dim).to(device)

    sf_critic_target = deepcopy(sf_critic).to(device)
    sf_critic_target.eval()  # same reasoning as actor_target above

    critic_optimizer = optim.Adam(sf_critic.parameters(), lr=1e-3)

    replay_buffer = ReplayBuffer()

    # ---------------------------------------------------------
    # Task weights setup
    # ---------------------------------------------------------

    if backward_only:
        phases = [1]  # 1 corresponds to backward
    else:
        phases = [0, 1]  # 0: forward, 1: backward

    returns_history = []

    for phase in phases:
        print(f"--- Starting Phase {phase} ---")

        # Initialize w as a learnable PyTorch parameter for the new phase
        # We use small random values instead of standard normal (randn) 
        # to prevent massive initial Q-value spikes before regression converges.
        w_param = torch.nn.Parameter(
            torch.rand(feature_dim, 1, dtype=torch.float32, device=device) * 0.1
        )
        w_optimizer = optim.Adam([w_param], lr=1e-2)

        # Reset Replay Buffer on task switch (as per Chua 2024 distribution shift protocol)
        replay_buffer.clear()
        
        if phase == phases[0]: # If it's the first phase, we reset the environment with a fixed seed for reproducibility. This works also for the backward_only case, as it will be the only phase. It would not work if we write `if phase == 0:` because in the backward_only case, phase would be 1, and we would not reset with a seed.
            state, _ = env.reset(seed=seed)
        else:
            state, _ = env.reset()

        episode_return = 0
        episode_returns = []
        critic_loss_val = 0.0
        q_loss_val = 0.0
        vec_loss_val = 0.0

        for step in range(steps_per_phase):

            # ---------------------------------------------------------
            # 1. Select Action (with exploration noise)
            # ---------------------------------------------------------
            if len(replay_buffer.storage) < WARMUP_STEPS:
                action = env.action_space.sample() # take random actions for exploration during warmup
            else:
                state_tensor = torch.tensor(
                    state,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)

                with torch.no_grad():
                    action = actor(state_tensor).cpu().numpy()[0]

                noise = np.random.normal(0, 0.1, size=action_dim)
                action = (action + noise).clip(-max_action, max_action)

            # ---------------------------------------------------------
            # 2. Step Environment
            # ---------------------------------------------------------
            next_state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # ---------------------------------------------------------
            # 3. Extract 3D Basis Features (phi) and True Reward
            # ---------------------------------------------------------
            velocity = info.get("x_velocity", 0.0)
            
            # Split velocity into two strictly positive features
            pos_fwd_vel = max(velocity, 0.0)
            pos_bwd_vel = max(-velocity, 0.0)
            
            ctrl_reward = info.get("reward_ctrl", 0.0)

            # 3D phi vector
            phi = np.array([pos_fwd_vel, pos_bwd_vel, ctrl_reward], dtype=np.float32)

            # Ground truth scalar reward provided by the environment
            if phase == 0:
                true_reward = pos_fwd_vel + ctrl_reward # forward task: maximize forward velocity and control reward
            else:
                true_reward = pos_bwd_vel + ctrl_reward # backward task: maximize backward velocity and control reward

            episode_return += true_reward

            # ---------------------------------------------------------
            # Store in Buffer
            # ---------------------------------------------------------
            replay_buffer.add(
                state,
                action,
                phi,
                next_state,
                float(terminated)  
            )

            state = next_state

            # ---------------------------------------------------------
            # 4. Train Networks
            # ---------------------------------------------------------
            if len(replay_buffer.storage) >= WARMUP_STEPS:

                batch_states, batch_actions, batch_phis, batch_next_states, batch_term = replay_buffer.sample(batch_size)

                # ---------------------------------------------------------
                # 4a. Train Task Weights (w) via Linear Regression
                # ---------------------------------------------------------
                with torch.no_grad():
                    if phase == 0:
                        batch_true_rewards = batch_phis[:, 0:1] + batch_phis[:, 2:3] # forward task: maximize forward velocity and control reward
                    else:
                        batch_true_rewards = batch_phis[:, 1:2] + batch_phis[:, 2:3] # backward task: maximize backward velocity and control reward

                # predicted reward = phi^T w
                predicted_rewards = torch.matmul(batch_phis, w_param)
                # MSE loss between predicted rewards and true rewards
                w_loss = F.mse_loss(predicted_rewards, batch_true_rewards)

                w_optimizer.zero_grad()
                w_loss.backward()
                w_optimizer.step()

                # Detach the learned weights so SF and Actor updates don't backpropagate into w
                learned_w = w_param.detach()

                # ---------------------------------------------------------
                # 4b. Train Critic
                # ---------------------------------------------------------
                critic_loss_val, q_loss_val, vec_loss_val = SFtrain_critic(
                    sf_critic,
                    sf_critic_target,
                    learned_w, 
                    actor_target,
                    batch_states,
                    batch_actions,
                    batch_phis,
                    batch_next_states,
                    batch_term,
                    critic_optimizer,
                    lambda_q=lambda_q, 
                    lambda_vec=lambda_vec,
                    gamma=gamma
                )

                # ---------------------------------------------------------
                # 4c. Train Actor
                # ---------------------------------------------------------
                SFtrain_actor(
                    actor,
                    sf_critic,
                    learned_w,
                    batch_states,
                    actor_optimizer
                )

                # ---------------------------------------------------------
                # Soft update target networks
                # ---------------------------------------------------------
                soft_update(sf_critic_target, sf_critic)
                soft_update(actor_target, actor)

            # ---------------------------------------------------------
            # Episode Reset Logic
            # ---------------------------------------------------------
            if done:
                state, _ = env.reset()
                episode_returns.append(episode_return)
                episode_return = 0

                if len(episode_returns) % 10 == 0:
                    avg_ret = np.mean(episode_returns[-10:])
                    print(
                        f"Step: {step} | "
                        f"Episodes: {len(episode_returns)} | "
                        f"Moving Avg Return: {avg_ret:.2f} | "
                        f"Critic Loss: {critic_loss_val:.4f} | "
                        f"Q Critic Loss: {q_loss_val:.4f} | "
                        f"Vec Critic Loss: {vec_loss_val:.4f}"  
                    ) 

        returns_history.append(episode_returns)

        torch.save(actor.state_dict(), run_dir / f"sf_actor_{phase}.pth")
        torch.save(sf_critic.state_dict(), run_dir / f"sf_critic_{phase}.pth")
    env.close()

    return returns_history


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# DDPG Baseline (for comparison)
# ---------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------
def train_ddpg(
    run_dir: Path | str,
    env_name="HalfCheetah-v5",
    steps_per_phase=50000,
    batch_size=64,
    seed=42,  # Default seed value for reproducibility if not provided via command line
    gamma=0.99,  # Discount factor for future rewards/features
    backward_only=False
):

    env = gym.make(env_name)

    #################################################
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    #################################################

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # ---------------------------------------------------------
    # Initialize Networks
    # ---------------------------------------------------------
    actor = Actor(state_dim, action_dim, max_action).to(device)

    actor_target = deepcopy(actor).to(device)
    actor_target.eval()
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)

    q_critic = QCritic(state_dim, action_dim).to(device)

    q_critic_target = deepcopy(q_critic).to(device)
    q_critic_target.eval() 
    critic_optimizer = optim.Adam(q_critic.parameters(), lr=1e-3)

    replay_buffer = ReplayBuffer()

    # ---------------------------------------------------------
    # Task weights setup (Updated to 3D Features)
    # ---------------------------------------------------------

   
    # The HalfCheetah-v5 environment does not know how to run backward. 
    # It only has one built-in reward function: v_{fwd} + r_{ctrl}.
    # If we want the cheetah to run backward, we have to "hack" the environment's reward signal.
    # We have to look at the (info), calculate the backward reward ourselves, and feed that new number to the DDPG agent.
    # The static w vector is used externally to project the 3D physical feature vector phi = [v_fwd, v_bwd, r_ctrl] into a single scalar reward.
    # The DDPG agent never sees w or phi; it simply learns to maximize the resulting scalar reward via standard Q-learning.

    # Task 1: maximize forward velocity (ignore backward) and maximize control reward
    w_forward = torch.tensor(
        [[1.0], [0.0], [1.0]],
        dtype=torch.float32,
        device=device
    )

    # Task 2: maximize backward velocity (ignore forward) and maximize control reward
    w_backward = torch.tensor(
        [[0.0], [1.0], [1.0]],
        dtype=torch.float32,
        device=device
    )

    if backward_only:
        tasks = [
            {"name": "Task 2 (Backward) from Scratch", "w": w_backward, "phase_idx": 1}
        ]
    else:
        tasks = [
            {"name": "Task 1 (Forward)", "w": w_forward, "phase_idx": 0},
            {"name": "Task 2 (Backward)", "w": w_backward, "phase_idx": 1}
        ]

    returns_history = []

    for task in tasks:

        print(f"--- Starting {task['name']} ---")

        w_current = task["w"]
        phase_idx = task["phase_idx"]

        # Reset Replay Buffer on task switch
        replay_buffer.clear()
        
        # Reset environment with seed only on the first phase executed
        if task == tasks[0]:
            state, _ = env.reset(seed=seed)
        else:
            state, _ = env.reset()
            
        episode_return = 0
        episode_returns = []
        critic_loss_val = 0.0

        for step in range(steps_per_phase):

            # ---------------------------------------------------------
            # 1. Select Action (with exploration noise)
            # ---------------------------------------------------------
            if len(replay_buffer.storage) < WARMUP_STEPS:
                action = env.action_space.sample()
            else:
                state_tensor = torch.tensor(
                    state,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)

                with torch.no_grad():
                    action = actor(state_tensor).cpu().numpy()[0]

                noise = np.random.normal(0, 0.1, size=action_dim)
                action = (action + noise).clip(-max_action, max_action)

            # ---------------------------------------------------------
            # 2. Step Environment
            # ---------------------------------------------------------
            next_state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # ---------------------------------------------------------
            # 3. Extract 3D Basis Features (phi)
            # ---------------------------------------------------------
            velocity = info.get("x_velocity", 0.0)
            
            # Split velocity into two strictly positive features
            pos_fwd_vel = max(velocity, 0.0)
            pos_bwd_vel = max(-velocity, 0.0)
            
            ctrl_reward = info.get("reward_ctrl", 0.0)

            phi = np.array(
                [pos_fwd_vel, pos_bwd_vel, ctrl_reward],
                dtype=np.float32
            )

            # ---------------------------------------------------------
            # Calculate actual scalar reward for logging purposes
            # ---------------------------------------------------------
            scalar_reward = float(
                phi @ w_current.detach().cpu().numpy().flatten()
            )

            episode_return += scalar_reward

            # ---------------------------------------------------------
            # Store in Buffer
            # ---------------------------------------------------------
            replay_buffer.add(
                state,
                action,
                phi,
                next_state,
                float(terminated)  
            )

            state = next_state

            # ---------------------------------------------------------
            # 4. Train Networks
            # ---------------------------------------------------------
            if len(replay_buffer.storage) >= WARMUP_STEPS:

                batch_states, batch_actions, batch_phis, batch_next_states, batch_term = replay_buffer.sample(batch_size)

                critic_loss_val = Qtrain_critic(
                    q_critic,
                    q_critic_target,
                    actor_target,
                    batch_states,
                    batch_actions,
                    batch_phis,
                    w_current, 
                    batch_next_states,
                    batch_term,
                    critic_optimizer,
                    gamma = gamma
                )

                Qtrain_actor(
                    actor,
                    q_critic,
                    batch_states,
                    actor_optimizer
                )

                soft_update(q_critic_target, q_critic)
                soft_update(actor_target, actor)

            # ---------------------------------------------------------
            # Episode Reset Logic
            # ---------------------------------------------------------
            if done:
                state, _ = env.reset()
                episode_returns.append(episode_return)
                episode_return = 0

                if len(episode_returns) % 10 == 0:
                    avg_ret = np.mean(episode_returns[-10:])
                    print(
                        f"Step: {step} | "
                        f"Episodes: {len(episode_returns)} | "
                        f"Moving Avg Return: {avg_ret:.2f} | "
                        f"Critic Loss: {critic_loss_val:.4f}"  
                    )

        returns_history.append(episode_returns)

        torch.save(actor.state_dict(), run_dir / f"q_actor_{phase_idx}.pth")
        torch.save(q_critic.state_dict(), run_dir / f"q_critic_{phase_idx}.pth")

    env.close()

    return returns_history

def parse_arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps_per_phase", type=int, default=50000)
    parser.add_argument("--lambda_q", type=float, default=1.0, help = "Q-SF-TD loss term weight")
    parser.add_argument("--lambda_vec", type=float, default=0.05, help = "Vectorial TD loss (standard SF-TD) term weight")
    parser.add_argument("--baseline", action= "store_true", help="If True, runs also DDPG baseline after SF-DDPG, for comparison.")
    parser.add_argument("--run_name", type=str, default="default_run", help="A name for this run, used to save results and plots with unique identifiers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards/features")
    parser.add_argument("--backward_only", action="store_true", help="Train ONLY Task 2 from scratch")
    args = parser.parse_args()
    return args
# ---------------------------------------------------------
# Main 
# ---------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    args = parse_arg()

    set_seed(args.seed) # <--- Set the seed globally

    # Save inside a subfolder specific to this seed
    run_dir = Path("artifacts") / args.run_name / f"seed_{args.seed}"
    run_dir.mkdir(exist_ok=True, parents=True)

    print("="*50)
    print("Training SF-DDPG...")
    print("="*50)
    SFreturns = train_sf_ddpg(
        run_dir=run_dir,
        steps_per_phase=args.steps_per_phase,
        lambda_q = args.lambda_q,
        lambda_vec = args.lambda_vec,
        seed=args.seed,  # Pass the seed to the training function to ensure reproducibility of environment interactions
        gamma=args.gamma,
        backward_only=args.backward_only
    )
    #plot_results(SFreturns, "SF-DDPG Sequential Training Adaptation (HalfCheetah)", run_dir / "sf_ddpg_results.pdf")
    with open(run_dir / "sf_ddpg_returns.json", "w") as f:
        json.dump(SFreturns, f)


    # print("Debug: SF-DDPG returns history:")
    # print(SFreturns)
    # print(f"N. episodes for each task: {len(SFreturns[0])}, {len(SFreturns[1])}")
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    if getattr(args, "baseline", False):  

        #############################
        # MODIFICA 2
        set_seed(args.seed)
        #############################

        print("="*50)
        print("Training DDPG...")  
        print("="*50)
        Qreturns = train_ddpg(
            run_dir=run_dir,
            steps_per_phase=args.steps_per_phase,
            seed=args.seed,  # Pass the seed to the training function to ensure reproducibility of environment interactions
            gamma=args.gamma,
            backward_only=args.backward_only
        )
        #plot_results(Qreturns, "Standard DDPG Sequential Training Adaptation (HalfCheetah)" , run_dir / "ddpg_results.pdf")
        with open(run_dir / "ddpg_returns.json", "w") as f:
            json.dump(Qreturns, f)

        # print("Debug: DDPG returns history:")
        # print(Qreturns)
        # print(f"N. episodes for each task: {len(Qreturns[0])}, {len(Qreturns[1])}")


