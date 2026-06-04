import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# 1. The Actor: Selects the deterministic action
# ---------------------------------------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
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
        return self.max_action * torch.tanh(self.l3(a)) # 6D action vector for the 6 joints of the HalfCheetah

# ---------------------------------------------------------
# 2. The SF Critic (Successor Feature Critic)
# Instead of predicting a 1D Q-value, it predicts the psi vector (feature_dim)
# ---------------------------------------------------------
class SFCritic(nn.Module):
    def __init__(self, state_dim, action_dim, feature_dim):
        super().__init__()
        # The critic's input is the concatenation of state and action
        self.l1 = nn.Linear(state_dim + action_dim, 256) # Input layer takes both state and action
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
def SFtrain_actor(actor, sf_critic, task_weights, state_batch, actor_optimizer):
    """
    actor_optimizer: Optimizer (e.g., Adam) for the Actor network
    task_weights: The 'w' vector defining the current task (shape: [feature_dim, 1])
    state_batch: Batch of states sampled from the replay buffer
    """
    actor_optimizer.zero_grad()
    # Logical scheme: action + state -> critic -> psi -> dot-product with w -> Q-value -> maximize Q-value by changing Actor weights (changing output action)  
    # The critic is fixed, so the actor learns to output actions that maximize the critic's output (psi), which in turn should lead to higher rewards as defined by the task weights (w).

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
    actor_loss = -q_values.mean() # Average over batch.
    
    # 5. Backpropagation and Actor weight update 
    actor_loss.backward()
    actor_optimizer.step()
    return actor_loss.item()

# ---------------------------------------------------------
# Critic Training Logic (TD Learning for SF)
# ---------------------------------------------------------
# The critic is trained to predict the expected feature vector (psi) for a given state-action pair.
def SFtrain_critic(
    sf_critic,
    sf_critic_target,
    task_weights, 
    actor_target, # We use the target actor to compute the next action for stability.
    state,
    action,
    phi,
    next_state,
    done,
    optimizer,
    gamma=0.99,
    lambda_q=1.0,
    lambda_vec=0.05
):
# The critic is trained on the Temporal Difference.

# Actor: the policy (given that the actions are continuous, we use a deterministic policy, so the actor directly outputs the action for a given state).
# Critic: the value function approximator (predicts psi). The critic is trained using TD learning, where the target is the immediate features (phi) plus the discounted expected future features (predicted by the target critic and the target actor).
# The critic loss is the mean squared error between the predicted psi and the target psi, which is then backpropagated to update the critic's weights.
# In a discrete space, the actor would simply be argmax over the action space, but in a continuous space, we need to use the actor network to output the action directly, and we train it using the critic's feedback. We need to use ANNs to approximate both the actor and the critic because of the continuous nature of the state and action spaces in MuJoCo environments.


    with torch.no_grad(): # We use torch.no_grad() to indicate that we do not want to compute gradients for the operations within this block. This is because we are calculating the target values for the critic, and we do not want to backpropagate through the target network or the actor when computing these targets.

        # The target for the critic is the immediate features (phi) plus the discounted expected future features.
        next_action = actor_target(next_state) # We are using data from the experience replay buffer (state, action, next_state)
        # phi is already observed, it is the immediate features after taking the action in the current state. We want to predict the expected future features (psi) given the current state and action, and we use the target critic to predict the expected future features for the next state and next action. The target for the critic is then the immediate features (phi) plus the discounted expected future features.

        # phi enters in the training of the critic because it represents the immediate features observed after taking the action in the current state. The critic is trained to predict the expected future features (psi) given a state-action pair, and the target for this prediction includes both the immediate features (phi) and the discounted expected future features.
        # Target q = immediate features (phi) + discounted expected future features.
        target_psi = phi + gamma * (1 - done) * sf_critic_target(next_state, next_action)

        target_q = phi @ task_weights + gamma * (1 - done) * (sf_critic_target(next_state, next_action) @ task_weights)# if the episode is terminated, we don't add the FUTURE features, hence the (1 - done) term.
        # TD target for the critic: immediate features + discounted expected future features

    current_psi = sf_critic_target(state, action)
    vec_loss = F.mse_loss(current_psi, target_psi)

    current_q = sf_critic(state, action) @ task_weights # TD Learning of the critic's current prediction of psi for the given state and action. 
    q_loss = F.mse_loss(current_q, target_q)

    loss = lambda_q * q_loss + lambda_vec * vec_loss

    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping improves numerical stability and helps avoid exploding gradients.
    torch.nn.utils.clip_grad_norm_(sf_critic.parameters(), max_norm=1.0)

    optimizer.step()

    # Return the critic loss so it can be logged during training,
    # making it easier to detect if the critic is diverging or failing to converge.
    return loss.item(), q_loss.item(), vec_loss.item()

# We are doing something quite different from the paper since we are not predicting the phi with another branch of the network, 
# but we are defining phi such that it is already optimised to give the correct reward in the scalar products
# while they use another encoder network. Here the world is simply given by a hand-made encoding (torques, velocity etc),
# so we don't need another branch of an NN to optimise phi, we can already craft the optimised one. 

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
############# Baseline: Standard Actor-Critic with Q-value prediction #############
class QCritic(nn.Module):
    """Standard DDPG critic: outputs a scalar Q-value."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)  # scalar output, not feature_dim

    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        return self.l3(q)  # returns Q(s,a) directly

# ---------------------------------------------------------
# 3. Actor Training Logic (The derivative "trick")
# ---------------------------------------------------------
def Qtrain_actor(actor, q_critic, task_weights, state_batch, actor_optimizer):
    """
    actor_optimizer: Optimizer (e.g., Adam) for the Actor network
    task_weights: The 'w' vector defining the current task (shape: [feature_dim, 1])
    state_batch: Batch of states sampled from the replay buffer
    """
    actor_optimizer.zero_grad()
    # Logical scheme: action + state -> critic -> psi -> dot-product with w -> Q-value -> maximize Q-value by changing Actor weights (changing output action)  
    # The critic is fixed, so the actor learns to output actions that maximize the critic's output (psi), which in turn should lead to higher rewards as defined by the task weights (w).

    # 1. The Actor decides the action based on the state
    actions = actor(state_batch) 
    
    # 2. The q-Critic predicts the q-value for those actions
    q_values = q_critic(state_batch, actions)

    # 3. Actor Loss Computation
    # We want to maximize the Q-value, so we minimize its negative value (-Q).
    # PyTorch's autograd automatically handles the chain rule to compute 
    # the derivative of Q with respect to the continuous action 'a'.
    actor_loss = -q_values.mean() # Average over batch.
    
    # 4. Backpropagation and Actor weight update 
    actor_loss.backward()
    actor_optimizer.step()
    return actor_loss.item()

# ---------------------------------------------------------
# Critic Training Logic (TD Learning for SF)
# ---------------------------------------------------------
def Qtrain_critic(
    q_critic,
    q_critic_target,
    actor_target, # We use the target actor to compute the next action for stability.
    state,
    action,
    phi,
    w, # we need also w to compute the reward from phi, since the reward is r = phi^T * w
    next_state,
    done,
    optimizer,
    gamma=0.99
):
# The critic is trained on the Temporal Difference.

# Actor: the policy (given that the actions are continuous, we use a deterministic policy, so the actor directly outputs the action for a given state).
# Critic: the value function approximator (predicts psi). The critic is trained using TD learning, where the target is the immediate features (phi) plus the discounted expected future features (predicted by the target critic and the target actor).
# The critic loss is the mean squared error between the predicted psi and the target psi, which is then backpropagated to update the critic's weights.
# In a discrete space, the actor would simply be argmax over the action space, but in a continuous space, we need to use the actor network to output the action directly, and we train it using the critic's feedback. We need to use ANNs to approximate both the actor and the critic because of the continuous nature of the state and action spaces in MuJoCo environments.


    with torch.no_grad(): # We use torch.no_grad() to indicate that we do not want to compute gradients for the operations within this block. This is because we are calculating the target values for the critic, and we do not want to backpropagate through the target network or the actor when computing these targets.

        # The target for the critic is the immediate features (phi) plus the discounted expected future features.
        next_action = actor_target(next_state) # We are using data from the experience replay buffer (state, action, next_state)
        # phi is already observed, it is the immediate features after taking the action in the current state. We want to predict the expected future features (psi) given the current state and action, and we use the target critic to predict the expected future features for the next state and next action. The target for the critic is then the immediate features (phi) plus the discounted expected future features.

        # phi enters in the training of the critic because it represents the immediate features observed after taking the action in the current state. The critic is trained to predict the expected future features (psi) given a state-action pair, and the target for this prediction includes both the immediate features (phi) and the discounted expected future features.
        # Target psi = immediate features (phi) + discounted expected future features.
        target_q = phi@w + gamma * (1 - done) * q_critic_target(next_state, next_action) # if the episode is terminated, we don't add the FUTURE features, hence the (1 - done) term.
        # TD target for the critic: immediate features + discounted expected future features
    current_q = q_critic(state, action) # TD Learning of the critic's current prediction of q-value for the given state and action.
    critic_loss = F.mse_loss(current_q, target_q)

    optimizer.zero_grad()
    critic_loss.backward()

    # Gradient clipping improves numerical stability and helps avoid exploding gradients.
    torch.nn.utils.clip_grad_norm_(q_critic.parameters(), max_norm=1.0)

    optimizer.step()

    # Return the critic loss so it can be logged during training,
    # making it easier to detect if the critic is diverging or failing to converge.
    return critic_loss.item()

