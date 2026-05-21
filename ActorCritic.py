import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# 1. The Actor: Selects the deterministic action
# ---------------------------------------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        # Standard fully connected neural network
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        # max_action is the motor force limit of the Cheetah (e.g., 1.0)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        # Tanh maps the output to [-1, 1], which is then scaled by max_action
        return self.max_action * torch.tanh(self.l3(a))

# ---------------------------------------------------------
# 2. The SF Critic (Successor Feature Critic)
# Instead of predicting a 1D Q-value, it predicts the psi vector (feature_dim)
# ---------------------------------------------------------
class SFCritic(nn.Module):
    def __init__(self, state_dim, action_dim, feature_dim):
        super(SFCritic, self).__init__()
        # The critic's input is the concatenation of state and action
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        # The output is the feature dimensionality (psi), not a scalar!
        self.l3 = nn.Linear(256, feature_dim)

    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        # Returns the successor features vector psi(s, a)
        return self.l3(q)

# ---------------------------------------------------------
# 3. Actor Training Logic (The derivative "trick")
# ---------------------------------------------------------
def train_actor(actor, sf_critic, task_weights, state_batch, actor_optimizer):
    """
    actor_optimizer: Optimizer (e.g., Adam) for the Actor network
    task_weights: The 'w' vector defining the current task (shape: [feature_dim, 1])
    state_batch: Batch of states sampled from the replay buffer
    """
    
    # 1. The Actor decides the action based on the state
    actions = actor(state_batch)
    
    # 2. The SF-Critic predicts the psi vector for those actions
    psi_values = sf_critic(state_batch, actions)
    
    # 3. Compute the Q-value as the dot product between psi and the task weights w
    # (Equivalent to Q(s, a) = psi(s,a)^T * w)
    q_values = torch.matmul(psi_values, task_weights)
    
    # 4. Actor Loss Computation
    # We want to maximize the Q-value, so we minimize its negative value (-Q).
    # PyTorch's autograd automatically handles the chain rule to compute 
    # the derivative of Q with respect to the continuous action 'a'.
    actor_loss = -q_values.mean()
    
    # 5. Backpropagation and Actor weight update
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    
    return actor_loss.item()