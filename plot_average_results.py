import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_data(base_path, algo_name): # Loads Phase 0 and Phase 1 returns for all seeds for a given algorithm
    all_seeds_phase_0 = []
    all_seeds_phase_1 = []
    
    for seed_dir in os.listdir(base_path): # Look for directories in the base path
        if seed_dir.startswith("seed_"): # Ensure we only look at seed directories
            file_path = os.path.join(base_path, seed_dir, f"{algo_name}_returns.json") # Construct the file path for the returns JSON
            if os.path.exists(file_path): # Check if the file exists before trying to open it
                with open(file_path, "r") as f: # Open the JSON file and load the data
                    data = json.load(f) 
                    all_seeds_phase_0.append(data[0]) # Append the first element (Phase 0 returns) to the list
                    if len(data) > 1:
                        all_seeds_phase_1.append(data[1]) # Append the second element (Phase 1 returns) to the list if it exists
                    
    return all_seeds_phase_0, all_seeds_phase_1

def smooth_and_pad(data_list, window=20): # Applies a moving average to each run and pads them to the same length based on the shortest run (even if they should all be the same length, we want to be safe and ensure they are aligned for averaging)
    if not data_list: return np.array([]) # Return an empty array if the input list is empty
    smoothed_data = [np.convolve(run, np.ones(window)/window, mode='valid') for run in data_list] # Apply moving average to each run (convolution with a uniform kernel 1/N where N is the window size = average over the window)
    min_len = min(len(run) for run in smoothed_data) # Find the minimum length among the smoothed runs to ensure we can pad them to the same length
    return np.array([run[:min_len] for run in smoothed_data]) # Pad each smoothed run to the minimum length by slicing (this ensures all runs are the same length for averaging)

def plot_curve(ax, data, label, color, linestyle='-'):
    if data.size == 0: return
    
    n_seeds = data.shape[0] # Number of seeds is the number of rows in the data array (each row corresponds to a different seed's learning curve)
    mean = np.mean(data, axis=0) # Calculate the mean across seeds for each episode (average return at each episode across all seeds)
    std = np.std(data, axis=0) # Calculate the standard deviation across seeds for each episode (variability of returns at each episode across all seeds)
    
    # Calculate Standard Error of the Mean (SEM)
    sem = std / np.sqrt(n_seeds)
    
    # Optional: For a 95% Confidence Interval, uncomment the next line
    # margin = 1.96 * sem 
    margin = sem # Using 1 Standard Error
    
    x = np.arange(len(mean)) # x-axis is the episode number (0 to length of mean - 1)
    ax.plot(x, mean, label=label, color=color, linestyle=linestyle) # Plot the mean learning curve for the given label and color
    ax.fill_between(x, mean - margin, mean + margin, color=color, alpha=0.15) # Fill the area between (mean - margin) and (mean + margin) to create a shaded region representing the confidence interval around the mean curve

def generate_transfer_comparison_plot(seq_dir, scratch_dir, window_size=20):
    """Generates the comparison plot and returns the Matplotlib figure and axes objects."""
    
    sf_seq_p0, sf_seq_p1 = load_data(seq_dir, "sf_ddpg") # Load Phase 0 and Phase 1 returns for SF-DDPG from the sequential training directory
    base_seq_p0, base_seq_p1 = load_data(seq_dir, "ddpg") # Load Phase 0 and Phase 1 returns for Standard DDPG from the sequential training directory
    
    sf_scratch_p0, _ = load_data(scratch_dir, "sf_ddpg") # Load Phase 0 returns for SF-DDPG from the scratch training directory (we only care about Phase 0 for the scratch runs since they are trained from scratch on Task 2)
    base_scratch_p0, _ = load_data(scratch_dir, "ddpg") # Load Phase 0 returns for Standard DDPG from the scratch training directory (same reason as above)
    
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


def generate_gamma_ablation_plot(run_dir, gamma_label, window_size=20):
    """Generates the comparison plot for a single gamma ablation run."""
    
    # 1. Load data 
    sf_p0, sf_p1 = load_data(run_dir, "sf_ddpg") 
    base_p0, base_p1 = load_data(run_dir, "ddpg")

    # 2. Smooth data (Removed all scratch_c variables)
    sf_p0_c = smooth_and_pad(sf_p0, window_size)
    sf_p1_c = smooth_and_pad(sf_p1, window_size)
    base_p0_c = smooth_and_pad(base_p0, window_size)
    base_p1_c = smooth_and_pad(base_p1, window_size)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- Phase 0 (Task 1) ---
    plot_curve(axes[0], sf_p0_c, f"SF-DDPG ($\gamma$ = {gamma_label})", "blue")
    plot_curve(axes[0], base_p0_c, f"Baseline DDPG ($\gamma$ = {gamma_label})", "orange")
    axes[0].set_title("Phase 0: Task 1 (Forward)")
    axes[0].set_xlabel("Episodes")
    axes[0].set_ylabel("Return (Smoothed)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # --- Phase 1 (Task 2) ---
    plot_curve(axes[1], sf_p1_c, f"SF-DDPG ($\gamma$ = {gamma_label})", "blue")
    plot_curve(axes[1], base_p1_c, f"Baseline DDPG ($\gamma$ = {gamma_label})", "orange")
    axes[1].set_title("Phase 1: Task 2 (Backward) - Transfer Learning")
    axes[1].set_xlabel("Episodes")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    return fig, axes

if __name__ == "__main__":
    LAMBDA_Q = "0.2"
    LAMBDA_VEC = "1.0"
    STEPS_PER_PHASE = "50000"

    # SEQ_DIR = os.path.join("artifacts", "final_eval")
    # SCRATCH_DIR = os.path.join("artifacts", "final_eval_backward_only")

    SEQ_DIR = os.path.join("artifacts", "transfer_learning", "eval_transfer_0_99_lq_0_2_lvec_1_0_stepsxphase_50000_transfer_learning")
    SCRATCH_DIR = os.path.join("artifacts", "transfer_learning", "eval_transfer_0_99_lq_0_2_lvec_1_0_stepsxphase_50000_backward_only")
    
    ROOT_DIR = "artifacts" # for saving the plots

    # Create the figures directory
    figures_dir = os.path.join(ROOT_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Create the transfer comparison subfolder
    transfer_comparison_folder = os.path.join(figures_dir, "transfer_comparison")
    os.makedirs(transfer_comparison_folder, exist_ok=True)

    # Create the gamma ablation subfolder
    gamma_folder = os.path.join(figures_dir, "gamma_ablation")
    os.makedirs(gamma_folder, exist_ok=True)

    fig, axes = generate_transfer_comparison_plot(SEQ_DIR, SCRATCH_DIR, window_size=20)
    
    # Save transfer comparison plot
    save_path = os.path.join(transfer_comparison_folder, "transfer.pdf")
    fig.savefig(save_path)

    # Save gamma ablation plot

    # List of gamma values to loop through
    gamma_values = ["0.1", "0.2" , "0.3", "0.5", "0.7", "0.8"]

    for gamma in gamma_values:
        # 1. Format the folder name (e.g., "0.5" becomes "eval_gamma_0_5")
        folder_name = f"eval_gamma_{gamma.replace('.', '_')}_lq_{LAMBDA_Q.replace('.', '_')}_lvec_{LAMBDA_VEC.replace('.', '_')}_stepsxphase_{STEPS_PER_PHASE}" 
        run_dir = os.path.join("artifacts", folder_name)
        
        # 2. Generate the plot
        fig, axes = generate_gamma_ablation_plot(run_dir, gamma)

        # 3. Format the save path (e.g., "0.5" becomes "gamma_0_5.pdf")
        pdf_filename = f"gamma_{gamma.replace('.', '_')}_lq_{LAMBDA_Q.replace('.', '_')}_lvec_{LAMBDA_VEC.replace('.', '_')}_stepsxphase_{STEPS_PER_PHASE}.pdf"
        save_path = os.path.join(gamma_folder, pdf_filename)
        
        # 4. Save and clear the figure from memory
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig) 
        
        print(f"Saved: {save_path}")
    
    #plt.show()