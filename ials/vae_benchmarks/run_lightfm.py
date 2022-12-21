import os
import subprocess
import json
import time
from typing import Any
from itertools import chain

import pandas as pd
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k
from lightfm.data import Dataset
import hydra
from scipy import sparse
from omegaconf import DictConfig

NUM_THREADS = 12


@hydra.main(version_base=None, config_path="./conf", config_name="lightfm_warp.yaml")
def run_lightfm(cfg: DictConfig):
    result: str = ""

    dataset_name = cfg.dataset_name  # "ml-20m"
    train_ml20 = pd.read_csv(f"{dataset_name}/train.csv")
    test_tr_ml20 = pd.read_csv(f"{dataset_name}/test_tr.csv")
    test_te_ml20 = pd.read_csv(f"{dataset_name}/test_te.csv")

    dataset = Dataset()
    dataset.fit_partial(train_ml20["uid"].tolist(), train_ml20["sid"].tolist())
    dataset.fit_partial(test_tr_ml20["uid"].tolist(), test_tr_ml20["sid"].tolist())
    dataset.fit_partial(test_te_ml20["uid"].tolist(), test_te_ml20["sid"].tolist())

    # (X_ml20_train, weights) = dataset.build_interactions(
    #     ((x[1]["uid"], x[1]["sid"]) for x in train_ml20.iterrows())
    # )
    # (X_ml20_test_tr, weights) = dataset.build_interactions(
    #     ((x[1]["uid"], x[1]["sid"]) for x in test_tr_ml20.iterrows())
    # )
    (X_ml20_full_train, weights) = dataset.build_interactions(
        (
            (x[1]["uid"], x[1]["sid"])
            for x in chain(train_ml20.iterrows(), test_tr_ml20.iterrows())
        )
    )
    (X_ml20_test_te, weights) = dataset.build_interactions(
        ((x[1]["uid"], x[1]["sid"]) for x in test_te_ml20.iterrows())
    )

    # X_ml20_full_train = sparse.coo_matrix(
    #     X_ml20_train.toarray() + X_ml20_test_tr.toarray()
    # )

    model = LightFM(
        no_components=int(cfg.embedding_dim),
        loss=cfg.loss_name,
        learning_schedule=cfg.learning_schedule,
        learning_rate=float(cfg.learning_rate),
    )

    model.fit_partial(X_ml20_full_train, epochs=0, num_threads=NUM_THREADS)
    full_train_time = 0

    for epoch in range(int(cfg.epochs)):
        start_time = time.time()
        model.fit_partial(X_ml20_full_train, epochs=1, num_threads=NUM_THREADS)
        # model.fit_partial(X_ml20_train, epochs=1, num_threads=NUM_THREADS)
        full_train_time += time.time() - start_time

        if epoch % 2 == 0:
            eval_metrics = {
                "Rec20": recall_at_k(
                    model,
                    test_interactions=X_ml20_test_te,
                    train_interactions=X_ml20_full_train,
                    k=20,
                    num_threads=NUM_THREADS,
                ).mean(),
                "Rec50": recall_at_k(
                    model,
                    test_interactions=X_ml20_test_te,
                    train_interactions=X_ml20_full_train,
                    k=50,
                    num_threads=NUM_THREADS,
                ).mean(),
            }
        # eval_metrics = {
        #     "Rec20": recall_at_k(
        #         model,
        #         test_interactions=X_ml20_test_te,
        #         train_interactions=X_ml20_test_tr,
        #         k=20,
        #         # num_threads=NUM_THREADS,
        #     ).mean(),
        #     "Rec50": recall_at_k(
        #         model,
        #         test_interactions=X_ml20_test_te,
        #         train_interactions=X_ml20_test_tr,
        #         k=50,
        #         # num_threads=NUM_THREADS,
        #     ).mean(),
        # }

        print(
            f"Epoch {epoch}\t Rec20={eval_metrics['Rec20']:.4f}, Rec50={eval_metrics['Rec50']:.4f}, time={full_train_time:.4f}"
        )
        result += f"Epoch {epoch}\t Rec20={eval_metrics['Rec20']:.4f}, Rec50={eval_metrics['Rec50']:.4f}, time={full_train_time:.4f}\n"

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cur_dir = hydra_cfg["runtime"]["output_dir"]

    with open(os.path.join(cur_dir, "results.txt"), "w") as f:
        print("write to file")
        f.write(result)


if __name__ == "__main__":
    run_lightfm()
