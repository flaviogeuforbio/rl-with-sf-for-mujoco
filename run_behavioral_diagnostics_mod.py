import os
import subprocess
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Run behavioral diagnostics.")
    parser.add_argument("--gamma", type=str, help="Gamma value (e.g., 0.99)")
    parser.add_argument("--mode", type=str, default="sf_action_optimization", help="Mode value (default: sf_action_optimization)")
    parser.add_argument("--model_type", type=str, default="sf", help="Model type (default: sf)")
    parser.add_argument("--phase", type=str, help="Phase value (e.g., 0)")
    parser.add_argument("--task", type=str, default="backward", help="Task value (default: backward)")
    parser.add_argument("--output_name", type=str, default=None, help="Output diagnostics results file name")

    args = parser.parse_args()

    gamma = args.gamma
    gamma_str = gamma.replace(".", "_")
    phase = args.phase
    # phase_str = phase.replace(".", "_")
    task = args.task
    # task_str = task.replace(".", "_")
    mode = args.mode
    # mode_str = mode.replace(".", "_")
    model_type = args.model_type 
    # model_type_str = model_type.replace(".", "_")
    output_name = args.output_name

    print(f"Starting behavioral diagnostics for Gamma = {gamma}, Mode = {mode}, Model Type = {model_type}, Phase = {phase}, Task = {task}...")

    for i in range(1, 2):

        # run_name = f"transfer_learning/eval_transfer_{gamma_str}_lq_0_2_lvec_1_0_stepsxphase_50000_transfer_learning/seed_{i}"
        run_name = f"eval_gamma_{gamma_str}_lq_0_2_lvec_1_0_stepsxphase_50000/seed_{i}"
        dir_path = os.path.join("artifacts", run_name)
        
        if os.path.isdir(dir_path):
            print(f"\n---> Processing Seed {i} ({run_name})")
            
            # 1. Feature Scales (Zero-shot transfer to backward task) 
            # subprocess.run([
            #     sys.executable, "diagnose_feature_scales.py",
            #     "--run_name", run_name,
            #     "--gamma", gamma,
            #     "--mode", mode,
            #     "--model_type", model_type,
            #     "--phase", phase,
            #     "--task", task,
            # ])
            
            # 2. Rollout Dynamics (Zero-shot transfer to backward task)
            subprocess.run([
                sys.executable, "diagnose_rollout_dynamics.py",
                "--run_name", run_name,
                "--gamma", gamma,
                "--mode", mode,
                "--model_type", model_type,
                "--phase", phase,
                "--task", task,
                "--output_name", output_name,
                "--render",
                "--save_timeseries",

            ])
        else:
            print(f"\n---> Skipping Seed {i}: Folder {dir_path} not found.")

if __name__ == "__main__":
    main()


#comando
# python .\run_behavioral_diagnostics_mod.py --gamma 0.99 --mode sf_action_optimization sf 0 backward