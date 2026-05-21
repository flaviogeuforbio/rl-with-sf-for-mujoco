import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from copy import deepcopy

# Import from the previously created file
from ActorCritic import Actor, SFCritic, train_actor

# ---------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------
class ReplayBuffer:
    def __init__(self, max_size=1e5):
        self.storage = []
        self.max_size = int(max_size)
        self.ptr = 0

    def add(self, state, action, phi, next_state, done):
        if len(self.storage) < self.max_size:
            self.storage.append((state, action, phi, next_state, done))
        else:
            self.storage[self.ptr] = (state, action, phi, next_state, done)
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.storage), size=batch_size)
        batch = [self.storage[i] for i in indices]
        states, actions, phis, next_states, dones = map(np.array, zip(*batch))
        return (torch.FloatTensor(states), torch.FloatTensor(actions),
                torch.FloatTensor(phis), torch.FloatTensor(next_states),
                torch.FloatTensor(dones).unsqueeze(1))

    def clear(self):
        self.storage = []
        self.ptr = 0

# ---------------------------------------------------------
# Critic Training Logic (TD Learning for SF)
# ---------------------------------------------------------
def train_critic(sf_critic, sf_critic_target, actor_target, state, action, phi, next_state, done, optimizer, gamma=0.99):
    with torch.no_grad():
        next_action = actor_target(next_state)
        # Target psi = immediate features (phi) + expected future features
        target_psi = phi + gamma * (1 - done) * sf_critic_target(next_state, next_action)
    
    current_psi = sf_critic(state, action)
    critic_loss = F.mse_loss(current_psi, target_psi)
    
    optimizer.zero_grad()
    critic_loss.backward()
    optimizer.step()

def soft_update(target, source, tau=0.005):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# ---------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------
def train_sf_ddpg(env_name="HalfCheetah-v5", steps_per_phase=50000, batch_size=64):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    feature_dim = 2  # [velocity, control_reward]

    # Initialize Networks
    actor = Actor(state_dim, action_dim, max_action)
    actor_target = deepcopy(actor)
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)

    sf_critic = SFCritic(state_dim, action_dim, feature_dim)
    sf_critic_target = deepcopy(sf_critic)
    critic_optimizer = optim.Adam(sf_critic.parameters(), lr=1e-3)

    replay_buffer = ReplayBuffer()

    # Task weights setup
    # Task 1: maximize forward velocity and maximize control reward (minimize penalty)
    w_forward = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
    # Task 2: maximize backward velocity (minimize forward) and maximize control reward
    w_backward = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)

    tasks = [
        {"name": "Task 1 (Forward)", "w": w_forward},
        {"name": "Task 2 (Backward)", "w": w_backward}
    ]

    returns_history = []

    for phase, task in enumerate(tasks):
        print(f"--- Starting {task['name']} ---")
        w_current = task['w']
        
        # Reset Replay Buffer on task switch (as per Chua 2024 distribution shift protocol)
        replay_buffer.clear()
        
        state, _ = env.reset()
        episode_return = 0
        episode_returns = []

        for step in range(steps_per_phase):
            # 1. Select Action (with exploration noise)
            if len(replay_buffer.storage) < batch_size * 10:
                action = env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = actor(state_tensor).detach().numpy()[0]
                action = (action + np.random.normal(0, 0.1, size=action_dim)).clip(-max_action, max_action)

            # 2. Step Environment
            next_state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Extract Basis Features (phi) directly from MuJoCo physics
            velocity = info.get('x_velocity', 0.0)
            ctrl_reward = info.get('reward_ctrl', 0.0)
            phi = np.array([velocity, ctrl_reward], dtype=np.float32)

            # Calculate actual scalar reward for logging purposes based on current task
            scalar_reward = float(np.dot(phi, w_current.numpy())[0])
            episode_return += scalar_reward

            # 3. Store in Buffer
            replay_buffer.add(state, action, phi, next_state, float(done))
            state = next_state

            # 4. Train Networks
            if len(replay_buffer.storage) > batch_size:
                batch_states, batch_actions, batch_phis, batch_next_states, batch_dones = replay_buffer.sample(batch_size)
                
                # Train Critic to predict Expected Features
                train_critic(sf_critic, sf_critic_target, actor_target, 
                             batch_states, batch_actions, batch_phis, batch_next_states, batch_dones, 
                             critic_optimizer)
                
                # Train Actor to maximize Q = psi^T * w
                train_actor(actor, sf_critic, w_current, batch_states, actor_optimizer)
                
                # Soft update target networks
                soft_update(sf_critic_target, sf_critic)
                soft_update(actor_target, actor)

            if done:
                state, _ = env.reset()
                episode_returns.append(episode_return)
                episode_return = 0
                
                if len(episode_returns) % 10 == 0:
                    avg_ret = np.mean(episode_returns[-10:])
                    print(f"Step: {step} | Episodes: {len(episode_returns)} | Moving Avg Return: {avg_ret:.2f}")

        returns_history.append(episode_returns)

    env.close()
    return returns_history

# ---------------------------------------------------------
# Plotting logic
# ---------------------------------------------------------
def plot_results(returns_history):
    plt.figure(figsize=(10, 5))
    
    # Calculate moving averages
    window = 10
    phase1_ma = np.convolve(returns_history[0], np.ones(window)/window, mode='valid')
    phase2_ma = np.convolve(returns_history[1], np.ones(window)/window, mode='valid')
    
    plt.plot(np.arange(len(phase1_ma)), phase1_ma, label='Task 1 (Forward)', color='blue')
    plt.plot(np.arange(len(phase1_ma), len(phase1_ma) + len(phase2_ma)), phase2_ma, label='Task 2 (Backward)', color='red')
    
    plt.axvline(x=len(phase1_ma), color='black', linestyle='--', label='Task Switch')
    plt.title('SF-DDPG Zero-Shot Adaptation (HalfCheetah)')
    plt.xlabel('Episodes')
    plt.ylabel('Episode Return')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Note: 50,000 steps per phase is set for computational speed to verify the script runs. 
    # To see actual asymptotic convergence as in Figure 4a, increase to 1e6 steps.
    returns = train_sf_ddpg(steps_per_phase=50000)
    plot_results(returns)