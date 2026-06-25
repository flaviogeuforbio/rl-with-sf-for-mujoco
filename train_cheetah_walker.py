import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path
import json
import os
import random

# Import from the previously created file
from ActorCritic import Actor, SFCritic, SFtrain_actor, SFtrain_critic, QCritic, Qtrain_actor, Qtrain_critic  
from utils import ReplayBuffer, soft_update, plot_results, evaluate_zero_shot_transfer_learning_sf, set_seed

WARMUP_STEPS = 1000  

# ---------------------------------------------------------
# Device Setup
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# Main Training Loop (SF-DDPG)
# ---------------------------------------------------------
def train_sf_ddpg(
    run_dir: Path | str,
    steps_per_phase=50000,
    batch_size=64,
    lambda_q=1.0,
    lambda_vec=0.05,
    seed=42, 
    gamma=0.99,
    walker_only=False,
    resume_dir=None,     # Directory to look for checkpoint
    save_freq=50000      # How often to save the checkpoint
):

    env_cheetah = gym.make("HalfCheetah-v5")
    env_walker = gym.make("Walker2d-v5")

    state_dim = env_cheetah.observation_space.shape[0]
    action_dim = env_cheetah.action_space.shape[0]
    max_action = float(env_cheetah.action_space.high[0])
    feature_dim = 3 

    actor = Actor(state_dim, action_dim, max_action).to(device)
    actor_target = deepcopy(actor).to(device)
    actor_target.eval()
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)

    sf_critic = SFCritic(state_dim, action_dim, feature_dim).to(device)
    sf_critic_target = deepcopy(sf_critic).to(device)
    sf_critic_target.eval()  
    critic_optimizer = optim.Adam(sf_critic.parameters(), lr=1e-3)

    replay_buffer = ReplayBuffer()

    if walker_only:
        phases = [1]  
    else:
        phases = [0, 1]  

    # ---------------------------------------------------------
    # CHECKPOINT LOADING (Networks)
    # ---------------------------------------------------------
    start_phase_idx = 0 
    resume_path = Path(resume_dir) / "checkpoint_sf.pt" if resume_dir else None

    if resume_path and resume_path.exists():
        print(f"--> [RESUME] Loading SF networks from {resume_path}")
        # Save the checkpoint with safe device loading
        checkpoint = torch.load(resume_path, map_location=device) # this contains the state_dicts for actor, sf_critic, their targets, optimizers, and other training states
        
        actor.load_state_dict(checkpoint['actor'])
        sf_critic.load_state_dict(checkpoint['sf_critic'])
        actor_target.load_state_dict(checkpoint['actor_target'])
        sf_critic_target.load_state_dict(checkpoint['sf_critic_target'])
        actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

        # Explicitly cast optimizer states to device
        for opt in [actor_optimizer, critic_optimizer]:
            for opt_state in opt.state.values():
                for k, v in opt_state.items():
                    if isinstance(v, torch.Tensor):
                        opt_state[k] = v.to(device)
        
        saved_phase = checkpoint['phase']
        
        # Very Important: we need to set the starting phase index based on the checkpoint's phase.
        # This is because if we are resuming from a checkpoint that was saved during the Walker task, we want to skip the Cheetah task and start directly from the Walker task.

        # Handle walker_only mismatch
        if saved_phase in phases:
            start_phase_idx = phases.index(saved_phase)
        else:
            start_phase_idx = 0
            
        # If the loaded checkpoint was saved at the exact end of a phase, advance to the next task
        if checkpoint.get('phase_completed', False):
            start_phase_idx += 1
            
        returns_history = checkpoint['returns_history']
    else:
        returns_history = []
    
    # --- If a checkpoint is loaded where the final phase (Walker) is already marked as phase_completed: True,
    #  The execution must be explicitly terminated. ---
    if start_phase_idx >= len(phases):
        print("Training already completed.")
        env_cheetah.close()
        env_walker.close()
        return returns_history
    # -------------------------

    for p_idx, phase in enumerate(phases):
        if p_idx < start_phase_idx:
            continue

        print(f"--- Starting Phase {phase} ({'Cheetah' if phase == 0 else 'Walker'}) ---")
        env = env_cheetah if phase == 0 else env_walker
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        # ---------------------------------------------------------
        # CHECKPOINT LOADING (Buffer & Iterators)
        # ---------------------------------------------------------
        if p_idx == start_phase_idx and resume_path and resume_path.exists():
            print(f"--> [RESUME] Restoring buffer and iterators at step {checkpoint['step']}")
           
            w_param = torch.nn.Parameter(checkpoint['w_param'].clone().to(device))
            w_optimizer = optim.Adam([w_param], lr=1e-2)

            # We must not forget to store the pointer ptr to the replay buffer location where the next experience will start from.
            w_optimizer.load_state_dict(checkpoint['w_optimizer'])
            # Explicitly cast w_optimizer states to device
            for opt_state in w_optimizer.state.values():
                for k, v in opt_state.items():
                    if isinstance(v, torch.Tensor):
                        opt_state[k] = v.to(device)

            replay_buffer.storage = checkpoint['replay_buffer_storage']
            replay_buffer.ptr = checkpoint.get('replay_buffer_ptr', len(checkpoint['replay_buffer_storage']))
            state = checkpoint['state']
            
            # --- Restore RNG (Random Number Generator) States ---
            np.random.set_state(checkpoint['numpy_rng'])
            torch.set_rng_state(checkpoint['torch_rng'])
            random.setstate(checkpoint['python_rng'])
            if checkpoint.get('cuda_rng') is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(checkpoint['cuda_rng'])
            
            env.reset(seed=seed)
            env.unwrapped.set_state(checkpoint['qpos'], checkpoint['qvel'])
            
            episode_return = checkpoint['episode_return']
            episode_returns = checkpoint['episode_returns']
            start_step = checkpoint['step'] + 1
            critic_loss_val = q_loss_val = vec_loss_val = 0.0
        else:
            w_param = torch.nn.Parameter(
                torch.rand(feature_dim, 1, dtype=torch.float32, device=device) * 0.1
            )
            w_optimizer = optim.Adam([w_param], lr=1e-2)
            replay_buffer.clear()

            # Always seed the environment reset, regardless of phase
            state, _ = env.reset(seed=seed)

            episode_return = 0
            episode_returns = []
            start_step = 0
            critic_loss_val = q_loss_val = vec_loss_val = 0.0

        for step in range(start_step, steps_per_phase):

            if len(replay_buffer.storage) < WARMUP_STEPS:
                action = env.action_space.sample() 
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action = actor(state_tensor).cpu().numpy()[0]
                noise = np.random.normal(0, 0.1, size=action_dim)
                action = (action + noise).clip(-max_action, max_action)

            next_state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            velocity = info.get("x_velocity", 0.0)
            pos_fwd_vel = max(velocity, 0.0)
            pos_bwd_vel = max(-velocity, 0.0)
            ctrl_reward = info.get("reward_ctrl", 0.0)

            phi = np.array([pos_fwd_vel, pos_bwd_vel, ctrl_reward], dtype=np.float32)
            true_reward = pos_fwd_vel + ctrl_reward
            episode_return += true_reward

            replay_buffer.add(state, action, phi, next_state, float(terminated))
            state = next_state

            if len(replay_buffer.storage) >= WARMUP_STEPS:
                batch_states, batch_actions, batch_phis, batch_next_states, batch_term = replay_buffer.sample(batch_size)

                with torch.no_grad():
                    batch_true_rewards = batch_phis[:, 0:1] + batch_phis[:, 2:3]

                predicted_rewards = torch.matmul(batch_phis, w_param)
                w_loss = F.mse_loss(predicted_rewards, batch_true_rewards)

                w_optimizer.zero_grad()
                w_loss.backward()
                w_optimizer.step()

                learned_w = w_param.detach()

                critic_loss_val, q_loss_val, vec_loss_val = SFtrain_critic(
                    sf_critic, sf_critic_target, learned_w, actor_target,
                    batch_states, batch_actions, batch_phis, batch_next_states,
                    batch_term, critic_optimizer, lambda_q=lambda_q, 
                    lambda_vec=lambda_vec, gamma=gamma
                )

                SFtrain_actor(actor, sf_critic, learned_w, batch_states, actor_optimizer)

                soft_update(sf_critic_target, sf_critic)
                soft_update(actor_target, actor)

            if done:
                state, _ = env.reset()
                episode_returns.append(float(episode_return))
                episode_return = 0

                if len(episode_returns) % 10 == 0:
                    avg_ret = np.mean(episode_returns[-10:])
                    print(f"Step: {step} | Episodes: {len(episode_returns)} | "
                          f"Avg Return: {avg_ret:.2f} | C Loss: {critic_loss_val:.4f} | "
                          f"Q Loss: {q_loss_val:.4f} | Vec Loss: {vec_loss_val:.4f}")

            # ---------------------------------------------------------
            # CHECKPOINT SAVE LOGIC
            # ---------------------------------------------------------
            if step > 0 and step % save_freq == 0:
                checkpoint = {
                    'phase': phase,
                    'phase_completed': False,
                    'step': step,
                    'actor': actor.state_dict(),
                    'sf_critic': sf_critic.state_dict(),
                    'actor_target': actor_target.state_dict(),
                    'sf_critic_target': sf_critic_target.state_dict(),
                    'actor_optimizer': actor_optimizer.state_dict(),
                    'critic_optimizer': critic_optimizer.state_dict(),
                    'w_param': w_param.detach().cpu(),
                    'w_optimizer': w_optimizer.state_dict(),
                    'replay_buffer_storage': replay_buffer.storage,
                    'replay_buffer_ptr': getattr(replay_buffer, 'ptr', 0),
                    'state': state,
                    'qpos': env.unwrapped.data.qpos.copy(),
                    'qvel': env.unwrapped.data.qvel.copy(),
                    'numpy_rng': np.random.get_state(),
                    'torch_rng': torch.get_rng_state(),
                    'python_rng': random.getstate(),
                    'cuda_rng': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                    'returns_history': returns_history,
                    'episode_returns': episode_returns,
                    'episode_return': episode_return
                }
                # Save to a temporary file first, then rename to avoid corruption (if the process is interrupted during saving)
                tmp_path = Path(run_dir) / "checkpoint_sf.tmp"
                ckpt_path = Path(run_dir) / "checkpoint_sf.pt"
                torch.save(checkpoint, tmp_path)
                os.replace(tmp_path, ckpt_path)
                print(f"--> [SAVE] Checkpoint saved at Phase {phase}, Step {step}")

        returns_history.append(episode_returns)
        torch.save(actor.state_dict(), Path(run_dir) / f"sf_actor_{phase}.pth")
        torch.save(sf_critic.state_dict(), Path(run_dir) / f"sf_critic_{phase}.pth")
        
        final_checkpoint = {
            'phase': phase,
            'phase_completed': True,
            'step': steps_per_phase,
            'actor': actor.state_dict(),
            'sf_critic': sf_critic.state_dict(),
            'actor_target': actor_target.state_dict(),
            'sf_critic_target': sf_critic_target.state_dict(),
            'actor_optimizer': actor_optimizer.state_dict(),
            'critic_optimizer': critic_optimizer.state_dict(),
            'w_param': w_param.detach().cpu(),
            'w_optimizer': w_optimizer.state_dict(),
            'replay_buffer_storage': replay_buffer.storage,
            'replay_buffer_ptr': getattr(replay_buffer, 'ptr', 0),
            'state': state,
            'qpos': env.unwrapped.data.qpos.copy(),
            'qvel': env.unwrapped.data.qvel.copy(),
            'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(),
            'python_rng': random.getstate(),
            'cuda_rng': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'returns_history': returns_history,
            'episode_returns': episode_returns,
            'episode_return': episode_return
        }
        # Save to a temporary file first, then rename to avoid corruption (if the process is interrupted during saving)
        tmp_path = Path(run_dir) / "checkpoint_sf.tmp"
        ckpt_path = Path(run_dir) / "checkpoint_sf.pt"
        torch.save(final_checkpoint, tmp_path)
        os.replace(tmp_path, ckpt_path)

    env_cheetah.close()
    env_walker.close()
    
    return returns_history


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# DDPG Baseline (for comparison)
# ---------------------------------------------------------
def train_ddpg(
    run_dir: Path | str,
    steps_per_phase=50000,
    batch_size=64,
    seed=42, 
    gamma=0.99, 
    walker_only=False,
    resume_dir=None,    
    save_freq=50000     
):

    env_cheetah = gym.make("HalfCheetah-v5")
    env_walker = gym.make("Walker2d-v5")

    state_dim = env_cheetah.observation_space.shape[0]
    action_dim = env_cheetah.action_space.shape[0]
    max_action = float(env_cheetah.action_space.high[0])

    actor = Actor(state_dim, action_dim, max_action).to(device)
    actor_target = deepcopy(actor).to(device)
    actor_target.eval()
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)

    q_critic = QCritic(state_dim, action_dim).to(device)
    q_critic_target = deepcopy(q_critic).to(device)
    q_critic_target.eval() 
    critic_optimizer = optim.Adam(q_critic.parameters(), lr=1e-3)

    replay_buffer = ReplayBuffer()

    w_forward = torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float32, device=device)

    if walker_only:
        tasks = [{"name": "Task 2 (Walker Forward) from Scratch", "w": w_forward, "phase_idx": 1}]
    else:
        tasks = [
            {"name": "Task 1 (Cheetah Forward)", "w": w_forward, "phase_idx": 0},
            {"name": "Task 2 (Walker Forward)", "w": w_forward, "phase_idx": 1}
        ]

    # ---------------------------------------------------------
    # CHECKPOINT LOAD LOGIC (Networks)
    # ---------------------------------------------------------
    start_phase_idx = 0
    resume_path = Path(resume_dir) / "checkpoint_q.pt" if resume_dir else None

    if resume_path and resume_path.exists():
        print(f"--> [RESUME] Loading Q networks from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        actor.load_state_dict(checkpoint['actor'])
        q_critic.load_state_dict(checkpoint['q_critic'])
        actor_target.load_state_dict(checkpoint['actor_target'])
        q_critic_target.load_state_dict(checkpoint['q_critic_target'])
        actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        # Explicitly cast optimizer states to device
        for opt in [actor_optimizer, critic_optimizer]:
            for opt_state in opt.state.values():
                for k, v in opt_state.items():
                    if isinstance(v, torch.Tensor):
                        opt_state[k] = v.to(device)

        returns_history = checkpoint['returns_history']
        
        saved_phase = checkpoint['phase_idx']
        for i, t in enumerate(tasks):
            if t['phase_idx'] == saved_phase:
                start_phase_idx = i
                break
                
        if checkpoint.get('phase_completed', False):
            start_phase_idx += 1
    else:
        returns_history = []
    
    # --- Check if training is already completed ---
    if start_phase_idx >= len(tasks):
        print("Training already completed.")
        env_cheetah.close()
        env_walker.close()
        return returns_history
    # -------------------------


    for p_idx, task in enumerate(tasks):
        if p_idx < start_phase_idx:
            continue

        print(f"--- Starting {task['name']} ---")

        w_current = task["w"]
        phase_idx = task["phase_idx"]

        env = env_cheetah if phase_idx == 0 else env_walker
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        # ---------------------------------------------------------
        # CHECKPOINT LOAD LOGIC (Buffer & Iterators)
        # ---------------------------------------------------------
        if p_idx == start_phase_idx and resume_path and resume_path.exists():
            print(f"--> [RESUME] Restoring buffer and iterators at step {checkpoint['step']}")
            replay_buffer.storage = checkpoint['replay_buffer_storage']
            replay_buffer.ptr = checkpoint.get('replay_buffer_ptr', len(checkpoint['replay_buffer_storage']))
            state = checkpoint['state']
            
            # --- Restore RNG States ---
            np.random.set_state(checkpoint['numpy_rng'])
            torch.set_rng_state(checkpoint['torch_rng'])
            random.setstate(checkpoint['python_rng'])
            if checkpoint.get('cuda_rng') is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(checkpoint['cuda_rng'])
            
            env.reset(seed=seed)
            env.unwrapped.set_state(checkpoint['qpos'], checkpoint['qvel'])
            
            episode_return = checkpoint['episode_return']
            episode_returns = checkpoint['episode_returns']
            start_step = checkpoint['step'] + 1
            critic_loss_val = 0.0
        else:
            replay_buffer.clear()
            
            # Fix 9: Always seed the environment reset, regardless of phase
            state, _ = env.reset(seed=seed)
            
            episode_return = 0
            episode_returns = []
            start_step = 0
            critic_loss_val = 0.0

        for step in range(start_step, steps_per_phase):

            if len(replay_buffer.storage) < WARMUP_STEPS:
                action = env.action_space.sample()
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action = actor(state_tensor).cpu().numpy()[0]
                noise = np.random.normal(0, 0.1, size=action_dim)
                action = (action + noise).clip(-max_action, max_action)

            next_state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            velocity = info.get("x_velocity", 0.0)
            pos_fwd_vel = max(velocity, 0.0)
            pos_bwd_vel = max(-velocity, 0.0)
            ctrl_reward = info.get("reward_ctrl", 0.0)

            phi = np.array([pos_fwd_vel, pos_bwd_vel, ctrl_reward], dtype=np.float32)
            scalar_reward = float(phi @ w_current.detach().cpu().numpy().flatten())
            episode_return += scalar_reward

            replay_buffer.add(state, action, phi, next_state, float(terminated))
            state = next_state

            if len(replay_buffer.storage) >= WARMUP_STEPS:
                batch_states, batch_actions, batch_phis, batch_next_states, batch_term = replay_buffer.sample(batch_size)

                critic_loss_val = Qtrain_critic(
                    q_critic, q_critic_target, actor_target, batch_states,
                    batch_actions, batch_phis, w_current, batch_next_states,
                    batch_term, critic_optimizer, gamma=gamma
                )

                Qtrain_actor(actor, q_critic, batch_states, actor_optimizer)

                soft_update(q_critic_target, q_critic)
                soft_update(actor_target, actor)

            if done:
                state, _ = env.reset()
                episode_returns.append(float(episode_return))
                episode_return = 0

                if len(episode_returns) % 10 == 0:
                    avg_ret = np.mean(episode_returns[-10:])
                    print(f"Step: {step} | Episodes: {len(episode_returns)} | "
                          f"Avg Return: {avg_ret:.2f} | C Loss: {critic_loss_val:.4f}")

            # ---------------------------------------------------------
            # CHECKPOINT SAVE LOGIC
            # ---------------------------------------------------------
            if step > 0 and step % save_freq == 0:
                checkpoint = {
                    'phase_idx': phase_idx,
                    'phase_completed': False,
                    'step': step,
                    'actor': actor.state_dict(),
                    'q_critic': q_critic.state_dict(),
                    'actor_target': actor_target.state_dict(),
                    'q_critic_target': q_critic_target.state_dict(),
                    'actor_optimizer': actor_optimizer.state_dict(),
                    'critic_optimizer': critic_optimizer.state_dict(),
                    'replay_buffer_storage': replay_buffer.storage,
                    'replay_buffer_ptr': getattr(replay_buffer, 'ptr', 0),
                    'state': state,
                    'qpos': env.unwrapped.data.qpos.copy(),
                    'qvel': env.unwrapped.data.qvel.copy(),
                    'numpy_rng': np.random.get_state(),
                    'torch_rng': torch.get_rng_state(),
                    'python_rng': random.getstate(),
                    'cuda_rng': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                    'returns_history': returns_history,
                    'episode_returns': episode_returns,
                    'episode_return': episode_return
                }
                # Save to a temporary file first, then rename to avoid corruption (if the process is interrupted during saving)
                tmp_path = Path(run_dir) / "checkpoint_q.tmp"
                ckpt_path = Path(run_dir) / "checkpoint_q.pt"
                torch.save(checkpoint, tmp_path)
                os.replace(tmp_path, ckpt_path)
                print(f"--> [SAVE] Checkpoint saved at Phase {phase_idx}, Step {step}")

        returns_history.append(episode_returns)
        torch.save(actor.state_dict(), Path(run_dir) / f"q_actor_{phase_idx}.pth")
        torch.save(q_critic.state_dict(), Path(run_dir) / f"q_critic_{phase_idx}.pth")
        
        # Explicitly construct the final checkpoint to avoid UnboundLocalError 
        final_checkpoint = {
            'phase_idx': phase_idx,
            'phase_completed': True,
            'step': steps_per_phase,
            'actor': actor.state_dict(),
            'q_critic': q_critic.state_dict(),
            'actor_target': actor_target.state_dict(),
            'q_critic_target': q_critic_target.state_dict(),
            'actor_optimizer': actor_optimizer.state_dict(),
            'critic_optimizer': critic_optimizer.state_dict(),
            'replay_buffer_storage': replay_buffer.storage,
            'replay_buffer_ptr': getattr(replay_buffer, 'ptr', 0),
            'state': state,
            'qpos': env.unwrapped.data.qpos.copy(),
            'qvel': env.unwrapped.data.qvel.copy(),
            'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(),
            'python_rng': random.getstate(),
            'cuda_rng': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'returns_history': returns_history,
            'episode_returns': episode_returns,
            'episode_return': episode_return
        }
        # Save to a temporary file first, then rename to avoid corruption (if the process is interrupted during saving)
        tmp_path = Path(run_dir) / "checkpoint_q.tmp"
        ckpt_path = Path(run_dir) / "checkpoint_q.pt"
        torch.save(final_checkpoint, tmp_path)
        os.replace(tmp_path, ckpt_path)

    env_cheetah.close()
    env_walker.close()
    return returns_history

def parse_arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps_per_phase", type=int, default=50000)
    parser.add_argument("--lambda_q", type=float, default=0.2, help="Q-SF-TD loss term weight")
    parser.add_argument("--lambda_vec", type=float, default=1.0, help="Vectorial TD loss term weight")
    parser.add_argument("--baseline", action="store_true", help="Run DDPG baseline")
    parser.add_argument("--run_name", type=str, default="default_run", help="Unique identifier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--walker_only", action="store_true", help="Train ONLY Walker2d-v5")
    
    # NEW ARGUMENTS FOR CHECKPOINTING
    parser.add_argument("--resume_dir", type=str, default=None, help="Directory containing .pt checkpoints")
    parser.add_argument("--save_freq", type=int, default=50000, help="Steps between saves")
    
    args = parser.parse_args()
    return args

# ---------------------------------------------------------
# Main 
# ---------------------------------------------------------
if __name__ == "__main__":
    args = parse_arg()
    set_seed(args.seed) 

    run_dir = Path("artifacts/walker") / args.run_name / f"seed_{args.seed}"
    run_dir.mkdir(exist_ok=True, parents=True)

    print("="*50)
    print("Training SF-DDPG...")
    print("="*50)
    SFreturns = train_sf_ddpg(
        run_dir=run_dir,
        steps_per_phase=args.steps_per_phase,
        lambda_q=args.lambda_q,
        lambda_vec=args.lambda_vec,
        seed=args.seed,  
        gamma=args.gamma,
        walker_only=args.walker_only,
        resume_dir=args.resume_dir,
        save_freq=args.save_freq
    )
    with open(run_dir / "sf_ddpg_returns.json", "w") as f:
        json.dump(SFreturns, f)

    if getattr(args, "baseline", False):  
        set_seed(args.seed)

        print("="*50)
        print("Training DDPG...")  
        print("="*50)
        Qreturns = train_ddpg(
            run_dir=run_dir,
            steps_per_phase=args.steps_per_phase,
            seed=args.seed,  
            gamma=args.gamma,
            walker_only=args.walker_only,
            resume_dir=args.resume_dir,
            save_freq=args.save_freq
        )
        with open(run_dir / "ddpg_returns.json", "w") as f:
            json.dump(Qreturns, f)