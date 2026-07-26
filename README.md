This branch contains the Walker2d task implementation using checkpointing, designed to be executed via Kaggle notebooks. The main script is `train_cheetah_walker.py`. The alternative script `survive_reward_train_cheetah_walker.py` implements the same task and checkpointing logic using 4D features, allowing for a comparison between the two representations, replicating the same experiment described in the report with the latter script.

## Reproducibility and Training

Due to hardware constraints and the need for multi-million step training, the models were trained using Kaggle environments. You can view the exact training environments, or fork them to run the experiments yourself, using the links below:

* **Walker2d Transfer Learning -- Initial Steps:** [View Notebook](https://www.kaggle.com/code/domenicoscarlatti/seed-1-initial-steps)
* **Walker2d Scratch Learning -- Initial Steps:** [View Notebook](https://www.kaggle.com/code/domenicoscarlatti/walker-only-seed-1-initial-steps)

* **Walker2d Transfer Learning -- Checkpointing:** [View Notebook](https://www.kaggle.com/code/domenicoscarlatti/seed-1-checkpointing)
* **Walker2d Scratch Learning -- Checkpointing:** [View Notebook](https://www.kaggle.com/code/domenicoscarlatti/walker-only-seed-1-checkpointing)

* **Checkpointing Dataset:** [View Dataset](https://kaggle.com/datasets/4e1808879e14162e1fbc4ec91f1fe3cbe8ed87e9a4a2f88f4fa6c07247bfdc61) 
*(Note: This dataset must be added as an input to the checkpointing scripts to resume training from the last saved state, and must be updated after each run with the new output).*

The completed checkpoints and the final data are available as a compressed zip folder in this Google Drive link: https://drive.google.com/file/d/1HycOL3qcKovVQRKE5T4rBcH0O-Z_I3oY/view?usp=drive_link

Static copies of these notebooks are also available in the `Notebooks/` directory of this repository for version control.

-- 
*Note: "Domenico Scarlatti" is a Kaggle alias for one of the developers who is very fond of this composer :)*