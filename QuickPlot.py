import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# 1. Define the correct path
run_dir = Path(r"C:\Users\mauro\Documents\RLProject\rl-with-sf-for-mujoco\artifacts\walker\transfer_learning_long_run_4_million_steps\3DFeatures_Walker_transfer_gamma_0_99_lq_1_0_lvec_1_0\seed_1")

def smooth(data, window=50):
    """Applies a moving average to smooth the learning curve."""
    if len(data) < window: return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def safe_load_phase_0(filepath):
    """Loads the JSON and safely extracts only Phase 0, even if Phase 1 doesn't exist."""
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return []
    with open(filepath, "r") as f:
        data = json.load(f)
    # If it's a nested list [[phase0], [phase1]], grab index 0. Otherwise, just use the flat list.
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        return data[0]
    return data

# 2. Load the data
sf_returns = safe_load_phase_0(run_dir / "sf_ddpg_returns.json")
ddpg_returns = safe_load_phase_0(run_dir / "ddpg_returns.json")

# 3. Plot the data
plt.figure(figsize=(10, 6))

if len(sf_returns) > 0:
    plt.plot(smooth(sf_returns), label="SF-DDPG", color="blue")
if len(ddpg_returns) > 0:
    plt.plot(smooth(ddpg_returns), label="Standard DDPG", color="orange")

plt.title("Current Training Progress (Phase 0 Only)")
plt.xlabel("Episodes (Smoothed over 50 eps)")
plt.ylabel("Return")
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Save and Show
save_path = run_dir / "phase_0_progress.pdf"
plt.savefig(save_path)
print(f"Plot saved to: {save_path}")
plt.show()