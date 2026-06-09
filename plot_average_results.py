import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_data(base_path, algo_name):
    """Loads episode returns across all seeds for a specific algorithm."""
    all_seeds_phase_0 = []
    all_seeds_phase_1 = []
    
    for seed_dir in os.listdir(base_path):
        if seed_dir.startswith("seed_"):
            file_path = os.path.join(base_path, seed_dir, f"{algo_name}_returns.json")
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    data = json.load(f)
                    all_seeds_phase_0.append(data[0])
                    all_seeds_phase_1.append(data[1])
                    
    return all_seeds_phase_0, all_seeds_phase_1

def smooth_and_pad(data_list, window=20):
    """Smooths data with a moving average and truncates to the minimum length across seeds."""
    smoothed_data = []
    for run in data_list:
        smoothed = np.convolve(run, np.ones(window)/window, mode='valid')
        smoothed_data.append(smoothed)
        
    # Truncate to the minimum length to create a uniform 2D array for statistical operations
    min_len = min(len(run) for run in smoothed_data)
    truncated_data = np.array([run[:min_len] for run in smoothed_data])
    
    return truncated_data

def plot_phase(ax, sf_data, baseline_data, title):
    """Plots the mean and ±1 standard deviation shaded region."""
    sf_mean = np.mean(sf_data, axis=0)
    sf_std = np.std(sf_data, axis=0)
    
    base_mean = np.mean(baseline_data, axis=0)
    base_std = np.std(baseline_data, axis=0)
    
    x_sf = np.arange(len(sf_mean))
    x_base = np.arange(len(base_mean))
    
    # SF-DDPG Curve
    ax.plot(x_sf, sf_mean, label="SF-DDPG", color='blue')
    ax.fill_between(x_sf, sf_mean - sf_std, sf_mean + sf_std, color='blue', alpha=0.2)
    
    # Baseline DDPG Curve
    ax.plot(x_base, base_mean, label="Standard DDPG", color='orange')
    ax.fill_between(x_base, base_mean - base_std, base_mean + base_std, color='orange', alpha=0.2)
    
    ax.set_title(title)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Return (Smoothed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

if __name__ == "__main__":
    BASE_DIR = os.path.join("artifacts", "final_eval")
    WINDOW_SIZE = 20  # Adjust this value to increase/decrease smoothing
    
    print("Aggregating data...")
    sf_phase_0, sf_phase_1 = load_data(BASE_DIR, "sf_ddpg")
    base_phase_0, base_phase_1 = load_data(BASE_DIR, "ddpg")
    
    print("Processing statistics...")
    sf_p0_clean = smooth_and_pad(sf_phase_0, WINDOW_SIZE)
    sf_p1_clean = smooth_and_pad(sf_phase_1, WINDOW_SIZE)
    base_p0_clean = smooth_and_pad(base_phase_0, WINDOW_SIZE)
    base_p1_clean = smooth_and_pad(base_phase_1, WINDOW_SIZE)
    
    print("Generating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    plot_phase(axes[0], sf_p0_clean, base_p0_clean, "Phase 0: Task 1 (Forward)")
    plot_phase(axes[1], sf_p1_clean, base_p1_clean, "Phase 1: Task 2 (Backward)")
    
    plt.tight_layout()
    plt.savefig("learning_curves_comparison.png", dpi=300)
    plt.show()