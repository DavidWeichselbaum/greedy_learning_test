import itertools
import random

import torch
import wandb

from train import run
from utils import get_commit_hash


PROJECT_NAME = "greedy_learning_test_CIFAR10"
TEST_NAME = "3-resid-modes"
N_REPEATS = 20
FIXED_PARAMS = {
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 0.001,
    "log_steps": 500,
}
TEST_PARAMS = {
    "do_auxloss": [True, False],
    "propagate_gradients": [True, False],
    "residual_mode": [None, "regular"]
    "classifier_mode": ["dense", "dense-s1", "average"]
}
TEST_PARAMS_combinations = [
    dict(zip(TEST_PARAMS.keys(), values))
    for values in itertools.product(*TEST_PARAMS.values())
]


def run_test(config, repeat, project_name, test_name):
    group = f"{test_name}_{'+auxloss' if config['do_auxloss'] else '-auxloss'}" \
            f"_{'+gradients' if config['propagate_gradients'] else '-gradients'}" \
            f"_resid={config['residual_mode']}" \
            f"_classifier={config['classifier_mode']}"
    name = f"{group}_run-{repeat}"

    wandb.init(
        # mode="disabled",
        project=project_name,
        group=group,
        name=name,
        notes=get_commit_hash(),
        config=config,
        reinit=True,
    )
    torch.manual_seed(wandb.config["seed"])

    run()


if __name__ == "__main__":
    for repeat in range(N_REPEATS):
        for i, test_params in enumerate(TEST_PARAMS_combinations):
            config = {**FIXED_PARAMS, **test_params, "seed": random.randint(0, 10000)}
            print(f"Test {i}, repeat {repeat}: {config}")
            run_test(config, repeat, PROJECT_NAME, TEST_NAME)
