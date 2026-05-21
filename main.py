import gymnasium as gym

# 1. Initialize the Environment
# "HalfCheetah-v4" is the standard 2D bipedal/cheetah MuJoCo environment.
# render_mode="human" forces the simulation to open a graphical window.
env = gym.make("HalfCheetah-v4", render_mode="human")

# 2. Reset the Environment
# Always call reset() before starting a new episode.
# observation: A vector representing the initial physical state (joint angles, angular velocities).
# info: A dictionary containing auxiliary diagnostic information.
observation, info = env.reset(seed=42)

# Define the length of our baseline test
num_steps = 100

# 3. Simulation Loop
for step in range(num_steps):
    
    # 4. Generate Random Action
    # The action space for HalfCheetah is continuous (a 6D vector of torques applied to the joints).
    # .sample() generates a random valid action within the permitted torque ranges.
    action = env.action_space.sample()
    
    # 5. Step the Environment
    # Feed the chosen action into the simulation to advance it by one timestep.
    # observation: The new physical state of the robot.
    # reward: The scalar reward achieved (based on forward velocity and control costs).
    # terminated: True if the environment reaches a fail state (not applicable in default HalfCheetah).
    # truncated: True if the episode reaches its maximum time limit.
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Optional debugging print to track the simulation progress
    print(f"Step: {step+1:03d} | Reward: {reward:+.4f}")
    
    # 6. Handle Episode End
    # If the simulation ends for any reason, reset it to continue looping safely.
    if terminated or truncated:
        observation, info = env.reset()

# 7. Cleanup
# Safely close the environment and destroy the render window.
env.close()