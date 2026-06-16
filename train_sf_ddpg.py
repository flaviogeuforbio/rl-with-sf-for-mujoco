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
    #MODIFICA 1
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    #################################################




    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    max_action = float(env.action_space.high[0])

    feature_dim = 2  # [velocity, control_reward]

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

    # Task 1: maximize forward velocity and maximize control reward (minimize penalty)
    w_forward = torch.tensor(
        [[1.0], [1.0]],
        dtype=torch.float32,
        device=device
    )

    # Task 2: maximize backward velocity (minimize forward) and maximize control reward
    w_backward = torch.tensor(
        [[-1.0], [1.0]],
        dtype=torch.float32,
        device=device
    )

    if backward_only:
        tasks = [
            {"name": "Task 2 (Backward) from Scratch", "w": w_backward}
        ]
    else:
        tasks = [
            {"name": "Task 1 (Forward)", "w": w_forward},
            {"name": "Task 2 (Backward)", "w": w_backward}
        ]

    returns_history = []

    for phase, task in enumerate(tasks):

        print(f"--- Starting {task['name']} ---")

        w_current = task["w"]

        # Reset Replay Buffer on task switch (as per Chua 2024 distribution shift protocol)
        replay_buffer.clear()
        
        if phase == 0:
            state, _ = env.reset(seed=seed)
        else:
            state, _ = env.reset()

        episode_return = 0

        episode_returns = []

        # initialise to 0.0 so that the logging print
        # inside `if terminated:` never raises an UnboundLocalError on episodes
        # that finish before the replay buffer has enough samples to trigger
        # the first training update.
        critic_loss_val = 0.0

        for step in range(steps_per_phase):

            # ---------------------------------------------------------
            # 1. Select Action (with exploration noise)
            # ---------------------------------------------------------

            # Pure Exploration:
            # For the first few thousand steps, the agent flails randomly
            # to gather diverse data.
            if len(replay_buffer.storage) < WARMUP_STEPS:

                action = env.action_space.sample()

            else:

                # Convert state to tensor and add batch dimension.
                state_tensor = torch.tensor(
                    state,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)

                # The Actor network outputs a deterministic action based on the current state.
                # We use torch.no_grad() to avoid building the autograd graph entirely
                # during inference, which is cleaner and slightly faster than calling
                # .detach() after the fact. The result is converted to a NumPy array
                # for interaction with the environment.
                with torch.no_grad():
                    action = actor(state_tensor).cpu().numpy()[0]

                # Exploration Noise:
                # Once the Actor starts making decisions, we add Gaussian noise
                # to its actions. The Actor is deterministic; without this noise,
                # it would execute the exact same movement every time and get stuck
                # in a local minimum.
                noise = np.random.normal(0, 0.1, size=action_dim)

                action = (action + noise).clip(-max_action, max_action)

            # ---------------------------------------------------------
            # 2. Step Environment
            # ---------------------------------------------------------

            # The chosen action (joint torques) is fed to MuJoCo.
            # MuJoCo computes the physics over a fraction of a second
            # and returns the next_state (new joint angles/velocities).
            next_state, _, terminated, truncated, info = env.step(action)

            # The episode ends if the Cheetah falls (terminated)
            # or if we reach a time limit (truncated).
            done = terminated or truncated

            # ---------------------------------------------------------
            # Extract Basis Features (phi)
            # ---------------------------------------------------------

            # Instead of taking MuJoCo's scalar reward,
            # we extract the raw physical metrics from the info dictionary
            # to build our basis features phi.

            # The forward velocity of the Cheetah,
            # which is the primary feature for both tasks.
            # For Task 1, we want to maximize this.
            # For Task 2, we want to minimize it
            # (or maximize backward velocity).
            velocity = info.get("x_velocity", 0.0)

            # The control reward (negative of the sum of squared torques)
            # is the second feature.
            # Both tasks want to maximize this
            # (i.e., minimize control effort).
            ctrl_reward = info.get("reward_ctrl", 0.0)

            # The phi vector is constructed by directly taking the relevant
            # features from the MuJoCo environment.
            # This is a key aspect of the SF framework:
            # we define a set of basis features (phi)
            # that capture the essential aspects of the environment
            # relevant to our tasks. (We don't need an encoder)
            phi = np.array(
                [velocity, ctrl_reward],
                dtype=np.float32
            )

            # s (long-term future rewards) = phi (short-term future reward) + gamma * psi'
            # psi (...) = phi + gamma * phi' + gamma^2 * phi'' + ... (expected total future reward) = G_t

            # ---------------------------------------------------------
            # Calculate actual scalar reward for logging purposes
            # ---------------------------------------------------------

            # For logging and plotting, we compute the actual scalar reward
            # that the agent receives based on the current task's weights.
            # This is done by taking the dot product of the phi vector
            # with the current task's weight vector w.
            scalar_reward = float( # R_t+1
                phi @ w_current.detach().cpu().numpy().flatten()
            )

            # We accumulate the scalar reward for the current episode
            # to track the episode return, which is what we will plot
            # in the results.
            episode_return += scalar_reward

            # ---------------------------------------------------------
            # Store in Buffer
            # ---------------------------------------------------------

            # After taking the action and observing the next state and reward,
            # we store this transition in the replay buffer.
            # The replay buffer is a data structure that holds past experiences
            # (state, action, phi, next_state, terminated)
            # that the agent can later sample from to learn.
            #
            # we store `terminated` rather than `done`
            # as the terminal flag. When `truncated=True` (time limit reached),
            # the episode ends administratively but the next state is still
            # physically valid. Bootstrapping should continue from it.
            # Storing `done=True` on truncation would incorrectly zero out the
            # TD target's future-return term, causing the critic to
            # underestimate Q-values near episode boundaries.
            replay_buffer.add(
                state,
                action,
                phi,
                next_state,
                float(terminated)  
            )

            # We update the current state to the next state
            # for the next iteration of the loop.
            # This is important because the agent's decision
            # at the next timestep will be based on this new state.
            state = next_state

            # ---------------------------------------------------------
            # 4. Train Networks (end of inference -- eval --- phase)
            # ---------------------------------------------------------

            # We only start training once we have enough samples
            # in the replay buffer to form a batch.
            if len(replay_buffer.storage) >= WARMUP_STEPS:

                # We sample a random batch of transitions
                # from the replay buffer.
                batch_states, batch_actions, batch_phis, batch_next_states, batch_term = replay_buffer.sample(batch_size)

                # ---------------------------------------------------------
                # Train Critic to predict Expected Features
                # ---------------------------------------------------------
                # train_critic returns the loss value
                # so we can monitor critic convergence during training.
                critic_loss_val, q_loss_val, vec_loss_val = SFtrain_critic(
                    sf_critic,
                    sf_critic_target,
                    w_current, 
                    actor_target,
                    batch_states,
                    batch_actions,
                    batch_phis,
                    batch_next_states,
                    batch_term,
                    critic_optimizer,
                    lambda_q = lambda_q, 
                    lambda_vec = lambda_vec,
                    gamma = gamma

                )
                # At first, we train the critic to have meaningful successor features predictions, then we train the Actor
                # ---------------------------------------------------------
                # Train Actor to maximize:
                # Q(s,a) = ψ(s,a)^T w
                # ---------------------------------------------------------
                SFtrain_actor(
                    actor,
                    sf_critic,
                    w_current,
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

            # When the episode ends, we log the episode return
            # and reset the environment for the next episode.
            if done:

                state, _ = env.reset()

                episode_returns.append(episode_return)

                # Reset episode return *after* appending, so the completed
                # episode's return is always recorded before clearing.
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
    seed=42,  # Default seed value for reproducibility if not provided via command line (Homage to Douglas Adams' "Answer to the Ultimate Question of Life, The Universe, and Everything")
    gamma=0.99,  # Discount factor for future rewards/features
    backward_only=False
):

    env = gym.make(env_name)

    #################################################
    #MODIFICA 1
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
    # Set target networks to eval mode. This is a no-op for plain MLPs
    # (no BatchNorm or Dropout), but is good practice: it signals clearly
    # that these networks are used for inference only and should never
    # be updated by an optimizer or behave differently at train time.
    actor_target.eval()

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)

    q_critic = QCritic(state_dim, action_dim).to(device)

    q_critic_target = deepcopy(q_critic).to(device)
    q_critic_target.eval()  # same reasoning as actor_target above

    critic_optimizer = optim.Adam(q_critic.parameters(), lr=1e-3)

    replay_buffer = ReplayBuffer()

    # ---------------------------------------------------------
    # Task weights setup
    # ---------------------------------------------------------

    # Task 1: maximize forward velocity and maximize control reward (minimize penalty)
    w_forward = torch.tensor(
        [[1.0], [1.0]],
        dtype=torch.float32,
        device=device
    )

    # Task 2: maximize backward velocity (minimize forward) and maximize control reward
    w_backward = torch.tensor(
        [[-1.0], [1.0]],
        dtype=torch.float32,
        device=device
    )

    if backward_only:
        tasks = [
            {"name": "Task 2 (Backward) from Scratch", "w": w_backward}
        ]
    else:
        tasks = [
            {"name": "Task 1 (Forward)", "w": w_forward},
            {"name": "Task 2 (Backward)", "w": w_backward}
        ]

    returns_history = []

    for phase, task in enumerate(tasks):

        print(f"--- Starting {task['name']} ---")

        w_current = task["w"]

        # Reset Replay Buffer on task switch (as per Chua 2024 distribution shift protocol)
        replay_buffer.clear()
        
        if phase == 0:
            state, _ = env.reset(seed=seed)
        else:
            state, _ = env.reset()
        episode_return = 0

        episode_returns = []

        # initialise to 0.0 so that the logging print
        # inside `if terminated:` never raises an UnboundLocalError on episodes
        # that finish before the replay buffer has enough samples to trigger
        # the first training update.
        critic_loss_val = 0.0

        for step in range(steps_per_phase):

            # ---------------------------------------------------------
            # 1. Select Action (with exploration noise)
            # ---------------------------------------------------------

            # Pure Exploration:
            # For the first few thousand steps, the agent flails randomly
            # to gather diverse data.
            if len(replay_buffer.storage) < WARMUP_STEPS:

                action = env.action_space.sample()

            else:

                # Convert state to tensor and add batch dimension.
                state_tensor = torch.tensor(
                    state,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)

                # The Actor network outputs a deterministic action based on the current state.
                # We use torch.no_grad() to avoid building the autograd graph entirely
                # during inference, which is cleaner and slightly faster than calling
                # .detach() after the fact. The result is converted to a NumPy array
                # for interaction with the environment.
                with torch.no_grad():
                    action = actor(state_tensor).cpu().numpy()[0]

                # Exploration Noise:
                # Once the Actor starts making decisions, we add Gaussian noise
                # to its actions. The Actor is deterministic; without this noise,
                # it would execute the exact same movement every time and get stuck
                # in a local minimum.
                noise = np.random.normal(0, 0.1, size=action_dim)

                action = (action + noise).clip(-max_action, max_action)

            # ---------------------------------------------------------
            # 2. Step Environment
            # ---------------------------------------------------------

            # The chosen action (joint torques) is fed to MuJoCo.
            # MuJoCo computes the physics over a fraction of a second
            # and returns the next_state (new joint angles/velocities).
            next_state, _, terminated, truncated, info = env.step(action)

            # The episode ends if the Cheetah falls (terminated)
            # or if we reach a time limit (truncated).
            done = terminated or truncated

            # ---------------------------------------------------------
            # Extract Basis Features (phi)
            # ---------------------------------------------------------

            # Instead of taking MuJoCo's scalar reward,
            # we extract the raw physical metrics from the info dictionary
            # to build our basis features phi.

            # The forward velocity of the Cheetah,
            # which is the primary feature for both tasks.
            # For Task 1, we want to maximize this.
            # For Task 2, we want to minimize it
            # (or maximize backward velocity).
            velocity = info.get("x_velocity", 0.0)

            # The control reward (negative of the sum of squared torques)
            # is the second feature.
            # Both tasks want to maximize this
            # (i.e., minimize control effort).
            ctrl_reward = info.get("reward_ctrl", 0.0)

            # The phi vector is constructed by directly taking the relevant
            # features from the MuJoCo environment.
            # This is a key aspect of the SF framework:
            # we define a set of basis features (phi)
            # that capture the essential aspects of the environment
            # relevant to our tasks. (We don't need an encoder)
            phi = np.array(
                [velocity, ctrl_reward],
                dtype=np.float32
            )

            # ---------------------------------------------------------
            # Calculate actual scalar reward for logging purposes
            # ---------------------------------------------------------

            # For logging and plotting, we compute the actual scalar reward
            # that the agent receives based on the current task's weights.
            # This is done by taking the dot product of the phi vector
            # with the current task's weight vector w.
            scalar_reward = float( # R_t+1
                phi @ w_current.detach().cpu().numpy().flatten()
            )

            # We accumulate the scalar reward for the current episode
            # to track the episode return, which is what we will plot
            # in the results.
            episode_return += scalar_reward

            # ---------------------------------------------------------
            # Store in Buffer
            # ---------------------------------------------------------

            # After taking the action and observing the next state and reward,
            # we store this transition in the replay buffer.
            # The replay buffer is a data structure that holds past experiences
            # (state, action, phi, next_state, terminated)
            # that the agent can later sample from to learn.
            #
            # we store `terminated` rather than `done`
            # as the terminal flag. When `truncated=True` (time limit reached),
            # the episode ends administratively but the next state is still
            # physically valid. Bootstrapping should continue from it.
            # Storing `done=True` on truncation would incorrectly zero out the
            # TD target's future-return term, causing the critic to
            # underestimate Q-values near episode boundaries.
            replay_buffer.add(
                state,
                action,
                phi,
                next_state,
                float(terminated)  
            )

            # We update the current state to the next state
            # for the next iteration of the loop.
            # This is important because the agent's decision
            # at the next timestep will be based on this new state.
            state = next_state

            # ---------------------------------------------------------
            # 4. Train Networks (end of inference -- eval --- phase)
            # ---------------------------------------------------------

            # We only start training once we have enough samples
            # in the replay buffer to form a batch.
            if len(replay_buffer.storage) >= WARMUP_STEPS:

                # We sample a random batch of transitions
                # from the replay buffer.
                batch_states, batch_actions, batch_phis, batch_next_states, batch_term = replay_buffer.sample(batch_size)

                # ---------------------------------------------------------
                # Train Critic to predict Expected Features
                # ---------------------------------------------------------
                # train_critic returns the loss value
                # so we can monitor critic convergence during training.
                critic_loss_val = Qtrain_critic(
                    q_critic,
                    q_critic_target,
                    actor_target,
                    batch_states,
                    batch_actions,
                    batch_phis,
                    w_current, # we need to pass also the current task weights to compute the TD target for the Q critic
                    batch_next_states,
                    batch_term,
                    critic_optimizer,
                    gamma = gamma
                )
                # At first, we train the critic to have meaningful successor features predictions, then we train the Actor
                # ---------------------------------------------------------
                # Train Actor to maximize:
                # Q(s,a) = ψ(s,a)^T w
                # ---------------------------------------------------------
                Qtrain_actor(
                    actor,
                    q_critic,
                    w_current,
                    batch_states,
                    actor_optimizer
                )

                # ---------------------------------------------------------
                # Soft update target networks
                # ---------------------------------------------------------
                soft_update(q_critic_target, q_critic)

                soft_update(actor_target, actor)

            # ---------------------------------------------------------
            # Episode Reset Logic
            # ---------------------------------------------------------

            # When the episode ends, we log the episode return
            # and reset the environment for the next episode.
            if done:

                state, _ = env.reset()

                episode_returns.append(episode_return)

                # Reset episode return *after* appending, so the completed
                # episode's return is always recorded before clearing.
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

        torch.save(actor.state_dict(), run_dir / f"q_actor_{phase}.pth")
        torch.save(q_critic.state_dict(), run_dir / f"q_critic_{phase}.pth")

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


