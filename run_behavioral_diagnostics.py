import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run behavioral diagnostics.")
    parser.add_argument("gamma", type=str, help="Gamma value (e.g., 0.5)")
    args = parser.parse_args()

    gamma_str = args.gamma.replace(".", "_")
    
    print(f"Starting behavioral diagnostics for Gamma = {args.gamma}...")

    for i in range(1, 11):
        run_name = f"eval_gamma_{gamma_str}/seed_{i}"
        dir_path = os.path.join("artifacts", run_name)
        
        if os.path.isdir(dir_path):
            print(f"\n---> Processing Seed {i} ({run_name})")
            
            # 1. Feature Scales (Zero-shot transfer to backward task)
            subprocess.run([
                "python", "diagnose_feature_scales.py",
                "--run_name", run_name,
                "--model_type", "sf",
                "--phase", "0",
                "--task", "backward"
            ])
            
            # 2. Rollout Dynamics (Zero-shot transfer to backward task)
            subprocess.run([
                "python", "diagnose_rollout_dynamics.py",
                "--run_name", run_name,
                "--mode", "sf_actor_only",
                "--phase", "0",
                "--task", "backward"
            ])
        else:
            print(f"\n---> Skipping Seed {i}: Folder {dir_path} not found.")

if __name__ == "__main__":
    main()