
import os
import subprocess
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Run PSI diagnostics across all seeds for a given gamma.")
    parser.add_argument("--gamma", type=str, help="The gamma value to process (e.g., 0.5)")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["sf"],
        default="sf",
        help="PSI diagnostics supporta solamente SF.",
    )
    parser.add_argument("--mode", type=str, default="sf_action_optimization", help="The mode value to process (default: sf_action_optimization)")
    parser.add_argument("--phase", type=str, help="The phase value to process (default: 0)")
    parser.add_argument("--task", type=str, default="backward", help="The task value to process (default: backward)")


    args = parser.parse_args()

    gamma = args.gamma
    gamma_str = gamma.replace(".", "_")
    model_type = args.model_type
    mode = args.mode
    phase = args.phase
    task = args.task


    print(f"Starting PSI diagnostics for Gamma = {gamma}, Phase = {phase}, Mode = {mode}, Task = {task}...")

    # Loop over seeds 1 to 10
    for i in range(1, 11):

        run_name = f"eval_transfer_{gamma_str}_lq_0_2_lvec_1_0_stepsxphase_50000/seed_{i}"
        
        # Check if the directory exists using os.path.join for cross-platform safety
        dir_path = os.path.join("artifacts", run_name)
        
        if os.path.isdir(dir_path):
            print(f"\n---> Processing Seed {i} ({run_name})")
            
            # Construct the command exactly as you would type it in the terminal
            command = [
                sys.executable, "diagnose_psi.py",
                "--run_name", run_name,
                "--gamma", gamma,
                "--model_type", model_type,
                "--policy_mode", mode,
                "--phase", phase,
                "--eval_task", task,
            ]
            
            
            # Run the command
            subprocess.run(command, check=True)
        else:
            print(f"\n---> Skipping Seed {i}: Folder {dir_path} not found.")

    print(f"\nDiagnostics complete for Gamma {gamma}.")

if __name__ == "__main__":
    main()

#comando
# python .\run_psi_diagnostic_mod.py 0.99 --model_type sf sf_action_optimization 0 backward
