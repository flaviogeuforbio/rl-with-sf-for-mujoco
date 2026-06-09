import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_data(base_path, algo_name):
    all_seeds_phase_0 = []
    all_seeds_phase_1 = []
    
    for seed_dir in os.listdir(base_path):
        if seed_dir.startswith("seed_"):
            file_path = os.path.join(base_path, seed_dir, f"{algo_name}_returns.json")
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    data = json.load(f)
                    all_seeds_phase_0.append(data[0])
                    if len(data) > 1:
                        all_seeds_phase_1.append(data[1])
                    
    return all_seeds_phase_0, all_seeds_phase_1

def smooth_and_pad(data_list, window=20):
    if not data_list: return np.array([])
    smoothed_data = [np.convolve(run, np.ones(window)/window, mode='valid') for run in data_list]
    min_len = min(len(run) for run in smoothed_data)
    return np.array([run[:min_len] for run in smoothed_data])

def plot_curve(ax, data, label, color, linestyle='-'):
    if data.size == 0: return
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    x = np.arange(len(mean))
    ax.plot(x, mean, label=label, color=color, linestyle=linestyle)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

def generate_transfer_comparison_plot(seq_dir, scratch_dir, window_size=20):
    """Generates the comparison plot and returns the Matplotlib figure and axes objects."""
    
    sf_seq_p0, sf_seq_p1 = load_data(seq_dir, "sf_ddpg")
    base_seq_p0, base_seq_p1 = load_data(seq_dir, "ddpg")
    
    sf_scratch_p0, _ = load_data(scratch_dir, "sf_ddpg")
    base_scratch_p0, _ = load_data(scratch_dir, "ddpg")
    
    sf_seq_p0_c = smooth_and_pad(sf_seq_p0, window_size)
    sf_seq_p1_c = smooth_and_pad(sf_seq_p1, window_size)
    base_seq_p0_c = smooth_and_pad(base_seq_p0, window_size)
    base_seq_p1_c = smooth_and_pad(base_seq_p1, window_size)
    
    sf_scratch_c = smooth_and_pad(sf_scratch_p0, window_size)
    base_scratch_c = smooth_and_pad(base_scratch_p0, window_size)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    plot_curve(axes[0], sf_seq_p0_c, "SF-DDPG", "blue")
    plot_curve(axes[0], base_seq_p0_c, "Standard DDPG", "orange")
    axes[0].set_title("Phase 0: Task 1 (Forward)")
    axes[0].set_xlabel("Episodes")
    axes[0].set_ylabel("Return (Smoothed)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    plot_curve(axes[1], sf_seq_p1_c, "SF-DDPG (After Task 1)", "blue")
    plot_curve(axes[1], base_seq_p1_c, "Standard DDPG (After Task 1)", "orange")
    
    plot_curve(axes[1], sf_scratch_c, "SF-DDPG (From Scratch)", "blue", linestyle='--')
    plot_curve(axes[1], base_scratch_c, "Standard DDPG (From Scratch)", "orange", linestyle='--')
    
    axes[1].set_title("Phase 1: Task 2 (Backward) - Transfer vs Scratch")
    axes[1].set_xlabel("Episodes")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    return fig, axes

if __name__ == "__main__":
    SEQ_DIR = os.path.join("artifacts", "final_eval")
    SCRATCH_DIR = os.path.join("artifacts", "final_eval_backward_only")
    
    fig, axes = generate_transfer_comparison_plot(SEQ_DIR, SCRATCH_DIR, window_size=20)
    
    fig.savefig("transfer_vs_scratch_comparison.pdf")
    plt.show()