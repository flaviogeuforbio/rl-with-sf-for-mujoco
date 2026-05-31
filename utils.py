from pathlib import Path 
import numpy as np
import torch
import matplotlib.pyplot as plt

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
# Plotting logic
# ---------------------------------------------------------
def plot_results(returns_history, figName = None):

    plt.figure(figsize=(10, 5))

    # Calculate moving averages.
    window = 10

    # Safety checks in case too few episodes were completed.
    if len(returns_history[0]) < window or len(returns_history[1]) < window:

        print("Not enough episodes for moving average plotting.")

        return
    # First Task (Forward)
    phase1_ma = np.convolve( # Computes the average return over a sliding window of 10 episodes for the first task (forward). The mode='valid' argument ensures that we only get averages for complete windows, so the resulting array will be shorter than the original by window-1.
        returns_history[0],
        np.ones(window) / window,
        mode='valid'
    )
    # Second Task (Backward)
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

    save_dir = Path("artifacts")
    save_dir.mkdir(exist_ok=True, parents=True)

    if figName:
        plt.savefig(save_dir / figName)
    else:
        plt.show()

