import os
import subprocess
import json

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="./conf", config_name="ialspp.yaml")
def run_ials(cfg: DictConfig):
    full_command = [
        f"./bin/{cfg.command}",
        "--train_data",
        cfg.train_data,
        "--test_train_data",
        cfg.test_train_data,
        "--test_test_data",
        cfg.test_test_data,
        "--embedding_dim",
        str(cfg.embedding_dim),
        "--stddev",
        cfg.stddev,
        "--regularization",
        cfg.regularization,
        "--regularization_exp",
        cfg.regularization_exp,
        "--unobserved_weight",
        cfg.unobserved_weight,
        "--epochs",
        cfg.epochs,
        "--block_size",
        cfg.block_size,
        "--eval_during_training",
        cfg.eval_during_training,
    ]

    result = subprocess.run(full_command, stdout=subprocess.PIPE)
    run_output = result.stdout.decode("utf-8")

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cur_dir = hydra_cfg["runtime"]["output_dir"]

    with open(os.path.join(cur_dir, "results.txt"), "w") as f:
        f.write(run_output)


if __name__ == "__main__":
    run_ials()
