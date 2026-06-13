import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run PSI diagnostics across all seeds for a given gamma.")
    parser.add_argument("gamma", type=str, help="The gamma value to process (e.g., 0.5)")
    args = parser.parse_args()

    gamma = args.gamma
    gamma_str = gamma.replace(".", "_")
    
    print(f"Starting PSI diagnostics for Gamma = {gamma}...")

    # Loop over seeds 1 to 10
    for i in range(1, 11):
        run_name = f"eval_gamma_{gamma_str}/seed_{i}" # TODO CHANGE NAME APPROPRIATELY
        
        # Check if the directory exists using os.path.join for cross-platform safety
        dir_path = os.path.join("artifacts", run_name)
        
        if os.path.isdir(dir_path):
            print(f"\n---> Processing Seed {i} ({run_name})")
            
            # Construct the command exactly as you would type it in the terminal
            command = [
                "python", "diagnose_psi.py",
                "--run_name", run_name,
                "--phase", "0",
                "--gamma", gamma
            ]
            
            # Run the command
            subprocess.run(command)
        else:
            print(f"\n---> Skipping Seed {i}: Folder {dir_path} not found.")

    print(f"\nDiagnostics complete for Gamma {gamma}.")

if __name__ == "__main__":
    main()