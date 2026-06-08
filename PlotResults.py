import json
from pathlib import Path
from utils import plot_results
import matplotlib.pyplot as plt

run_dir = Path("artifacts/comparison_001/comparison_001") # Update with your actual run_name directory

# Regenerate SF-DDPG Plot
with open(run_dir / "sf_ddpg_returns.json", "r") as f:
    sf_returns = json.load(f)
plot_results(
    sf_returns, 
    figName=run_dir / "sf_ddpg_results.pdf", 
    title="SF-DDPG Sequential Training Adaptation (HalfCheetah)"
)

# Regenerate Standard DDPG Plot
with open(run_dir / "ddpg_returns.json", "r") as f:
    ddpg_returns = json.load(f)
fig1, ax1 = plot_results(
    ddpg_returns, 
    figName=run_dir / "ddpg_results.pdf", 
    title="Standard DDPG Sequential Training Adaptation (HalfCheetah)"
)
