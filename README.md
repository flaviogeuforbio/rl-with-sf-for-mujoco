# Successor Features under Stress

This `diagnostics` branch contains SF-DDPG and DDPG training for
`HalfCheetah-v5`, zero-shot forward-to-backward evaluation, and diagnostics at
feature, critic, action, and rollout level.

## Setup

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Activate the environment:

- Linux/macOS: `source .venv/bin/activate`
- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`

Then install the dependencies:

```bash
pip install -r requirements.txt
```

## Main commands

Train SF-DDPG and the DDPG baseline:

```bash
python train_sf_ddpg.py --run_name example_run --gamma 0.99 --steps_per_phase 50000 --lambda_q 0.2 --lambda_vec 1.0 --baseline
```

Run the zero-shot backward evaluation:

```bash
python zero_shot_eval.py --run_name example_run --phase 0 --mode all
```

Run the three main diagnostic levels:

```bash
python diagnose_feature_scales.py --run_name example_run --gamma 0.99 --model_type sf --mode sf_action_optimization --phase 0 --task backward
python diagnose_psi.py --run_name example_run --gamma 0.99 --model_type sf --policy_mode sf_action_optimization --phase 0 --eval_task backward
python diagnose_rollout_dynamics.py --run_name example_run --gamma 0.99 --model_type sf --mode sf_action_optimization --phase 0 --task backward --episodes 20 --save_timeseries
```

Models and results are saved under `artifacts/<run_name>/`; diagnostic outputs
are written to `artifacts/<run_name>/diagnostics/`.
