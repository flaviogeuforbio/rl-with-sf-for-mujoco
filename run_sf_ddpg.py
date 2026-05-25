import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from copy import deepcopy

# Import from the previously created file
from ActorCritic import Actor, SFCritic, train_actor, train_critic

# ---------------------------------------------------------
# Device Setup
# ---------------------------------------------------------
# We automatically use the GPU if available; otherwise we fall back to the CPU.
# This makes the implementation portable and avoids device mismatch errors.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------
# A simple replay buffer to store transitions and sample batches for training.
class ReplayBuffer:
    def __init__(self, max_size=1e5):
        self.storage = [] # A list to hold the transitions (state, action, phi, next_state, terminated)
        self.max_size = int(max_size) # Maximum number of transitions the buffer can hold. Once we exceed this, we will start overwriting old transitions in a circular manner.
        self.ptr = 0 # A pointer to keep track of where to insert the next transition when the buffer is full. It starts at 0 and increments with each new transition, wrapping around to the beginning of the buffer when it reaches max_size.

    # The add method is used to store a new transition in the replay buffer.
    # Each transition consists of:
    # - state: The state of the environment before taking the action.
    # - action: The action taken by the agent.
    # - phi: The feature vector representing the immediate rewards.
    # - next_state: The state of the environment after taking the action.
    # - terminated: A boolean indicating whether the episode has ended.
    def add(self, state, action, phi, next_state, terminated):

        data = (state, action, phi, next_state, terminated)

        # If the buffer is not yet full, append the new transition.
        if len(self.storage) < self.max_size:
            self.storage.append(data)

        # Otherwise overwrite the oldest transition using circular indexing.
        else:
            self.storage[self.ptr] = data

        # Move the pointer forward regardless of whether we appended or overwrote.
        # This ensures that once the buffer is full, old experiences are replaced uniformly.
        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size): # The sample method is used to randomly sample a batch of transitions from the replay buffer for training.

        # Safety check to ensure that we do not try to sample more elements than available.
        assert len(self.storage) >= batch_size, "Not enough samples in replay buffer."

        indices = np.random.randint(0, len(self.storage), size=batch_size) # Python Mechanics: Generates an array of batch_size (e.g., 64) random integers.
        # The integers range from 0 to the current number of items in the memory bank (len(self.storage)).
        # RL Purpose: This implements uniform random sampling. Neural networks require independent
        # and identically distributed (i.i.d.) data to learn effectively.
        # If you train on steps sequentially (step 1, step 2, step 3), the strong temporal correlation will cause
        # the network's weights to diverge or oscillate. Random sampling breaks this correlation.

        batch = [self.storage[i] for i in indices] # We create a list of sampled transitions by indexing into the storage with the randomly generated indices. Each entry in this list is a tuple of (state, action, phi, next_state, done).

        states, actions, phis, next_states, term_values = map(np.array, zip(*batch)) # We use zip(*) to transpose the list
        # of transitions into separate lists for states, actions, phis, next_states, and terminated.
        # Then we convert each of these lists into NumPy arrays for easier manipulation during training.

        return (
            torch.tensor(states, dtype=torch.float32, device=device),
            torch.tensor(actions, dtype=torch.float32, device=device),
            torch.tensor(phis, dtype=torch.float32, device=device),
            torch.tensor(next_states, dtype=torch.float32, device=device),
            torch.tensor(term_values, dtype=torch.float32, device=device).unsqueeze(1)
        )

        # We return the sampled batch as PyTorch tensors, which will be used for training the neural networks.

    def clear(self): # A method to clear the replay buffer, which is useful when we switch tasks to ensure that the agent learns from the new distribution of experiences.
        self.storage = []
        self.ptr = 0


# ---------------------------------------------------------
# Soft Target Network Update
# ---------------------------------------------------------
# Instead of copying the online network directly into the target network,
# we slowly blend the parameters. This stabilizes temporal-difference learning.

# target networks are used to compute the target values for the critic's loss function. By keeping a separate target network that is updated more slowly than the online network, we prevent the target values from changing too rapidly during training, which can lead to more stable learning and better convergence properties.
# target slowly adapts to the online network, providing a more stable target for the critic's updates. This technique is commonly used in deep reinforcement learning algorithms like DDPG and TD3 to improve training stability.
def soft_update(target, source, tau=0.005):

    for target_param, param in zip(target.parameters(), source.parameters()):

        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

# ---------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------
def train_sf_ddpg(
    env_name="HalfCheetah-v5",
    steps_per_phase=50000,
    batch_size=64
):

    env = gym.make(env_name)

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
            if len(replay_buffer.storage) < batch_size * 10:

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
            if len(replay_buffer.storage) >= batch_size:

                # We sample a random batch of transitions
                # from the replay buffer.
                batch_states, batch_actions, batch_phis, batch_next_states, batch_term = replay_buffer.sample(batch_size)

                # ---------------------------------------------------------
                # Train Critic to predict Expected Features
                # ---------------------------------------------------------
                # train_critic returns the loss value
                # so we can monitor critic convergence during training.
                critic_loss_val = train_critic(
                    sf_critic,
                    sf_critic_target,
                    actor_target,
                    batch_states,
                    batch_actions,
                    batch_phis,
                    batch_next_states,
                    batch_term,
                    critic_optimizer
                )
                # At first, we train the critic to have meaningful successor features predictions, then we train the Actor
                # ---------------------------------------------------------
                # Train Actor to maximize:
                # Q(s,a) = ψ(s,a)^T w
                # ---------------------------------------------------------
                train_actor(
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
                        f"Critic Loss: {critic_loss_val:.4f}"  
                    )

        returns_history.append(episode_returns)

    env.close()

    return returns_history

# ---------------------------------------------------------
# Plotting logic
# ---------------------------------------------------------
def plot_results(returns_history):

    plt.figure(figsize=(10, 5))

    # Calculate moving averages.
    window = 10

    # Safety checks in case too few episodes were completed.
    if len(returns_history[0]) < window or len(returns_history[1]) < window:

        print("Not enough episodes for moving average plotting.")

        return

    phase1_ma = np.convolve(
        returns_history[0],
        np.ones(window) / window,
        mode='valid'
    )

    phase2_ma = np.convolve(
        returns_history[1],
        np.ones(window) / window,
        mode='valid'
    )

    plt.plot(
        np.arange(len(phase1_ma)),
        phase1_ma,
        label='Task 1 (Forward)',
        color='blue'
    )

    plt.plot(
        np.arange(len(phase1_ma), len(phase1_ma) + len(phase2_ma)),
        phase2_ma,
        label='Task 2 (Backward)',
        color='red'
    )

    plt.axvline(
        x=len(phase1_ma),
        color='black',
        linestyle='--',
        label='Task Switch'
    )

    plt.title('SF-DDPG Zero-Shot Adaptation (HalfCheetah)')

    plt.xlabel('Episodes')

    plt.ylabel('Episode Return')

    plt.legend()

    plt.grid(True)

    plt.show()

# ---------------------------------------------------------
# Main 
# ---------------------------------------------------------
if __name__ == "__main__":

    # Note:
    # 50,000 steps per phase is set for computational speed
    # to verify the script runs.
    #
    # To see actual asymptotic convergence as in Figure 4a,
    # increase to 1e6 steps.
    returns = train_sf_ddpg(
        steps_per_phase=50000
    )

    plot_results(returns)
