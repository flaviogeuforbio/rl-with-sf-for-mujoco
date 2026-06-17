import os
import subprocess
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Run behavioral diagnostics.")
    parser.add_argument("gamma", type=str, help="Gamma value (e.g., 0.99)")
    parser.add_argument("phase", type=str, help="Phase value (e.g., 0)")
    parser.add_argument("task", type=str, default="backward", help="Task value (default: backward)")
    parser.add_argument("mode", type=str, default="sf_action_optimization", help="Mode value (default: sf_action_optimization)")

    args = parser.parse_args()

    gamma = args.gamma
    gamma_str = gamma.replace(".", "_")
    phase = args.phase
    phase_str = phase.replace(".", "_")
    task = args.task
    task_str = task.replace(".", "_")
    mode = args.mode
    mode_str = mode.replace(".", "_")

    print(f"Starting behavioral diagnostics for Gamma = {gamma}, Phase = {phase}, Task = {task}, Mode = {mode}...")

    for i in range(1, 2):
        run_name = f"transfer_learning/eval_transfer_0_99_lq_0_2_lvec_1_0_stepsxphase_50000_transfer_learning/seed_{i}"
        dir_path = os.path.join("artifacts", run_name)
        
        if os.path.isdir(dir_path):
            print(f"\n---> Processing Seed {i} ({run_name})")
            
            # 1. Feature Scales (Zero-shot transfer to backward task)
            subprocess.run([
                sys.executable, "diagnose_feature_scales.py",
                "--run_name", run_name,
                "--model_type", "sf",
                "--phase", phase,
                "--task", task
            ])
            
            # 2. Rollout Dynamics (Zero-shot transfer to backward task)
            subprocess.run([
                sys.executable, "diagnose_rollout_dynamics.py",
                "--run_name", run_name,
                "--mode", mode,
                "--phase", phase,
                "--task", task,
                # "--render",
                # "--save_timeseries",

            ])
        else:
            print(f"\n---> Skipping Seed {i}: Folder {dir_path} not found.")

if __name__ == "__main__":
    main()