
import os
import subprocess
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Run PSI diagnostics across all seeds for a given gamma.")
    parser.add_argument("gamma", type=str, help="The gamma value to process (e.g., 0.5)")
    parser.add_argument("phase", type=str, help="The phase value to process (default: 0)")
    args = parser.parse_args()

    gamma = args.gamma
    gamma_str = gamma.replace(".", "_")
    phase = args.phase
    phase_str = phase.replace(".", "_")
    
    print(f"Starting PSI diagnostics for Gamma = {gamma}, Phase = {phase}...")

    # Loop over seeds 1 to 10
    for i in range(1, 11):
        run_name = f"transfer_learning/eval_transfer_0_99_lq_0_2_lvec_1_0_stepsxphase_50000_transfer_learning/seed_{i}"
        
        # Check if the directory exists using os.path.join for cross-platform safety
        dir_path = os.path.join("artifacts", run_name)
        
        if os.path.isdir(dir_path):
            print(f"\n---> Processing Seed {i} ({run_name})")
            
            # Construct the command exactly as you would type it in the terminal
            command = [
                sys.executable, "diagnose_psi.py",
                "--run_name", run_name,
                "--phase", phase,
                "--gamma", gamma
            ]
            
            # Run the command
            subprocess.run(command)
        else:
            print(f"\n---> Skipping Seed {i}: Folder {dir_path} not found.")

    print(f"\nDiagnostics complete for Gamma {gamma}.")

if __name__ == "__main__":
    main()