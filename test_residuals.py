import itertools
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb

from train import run


PROJECT_NAME = "greedy_learning_test_CIFAR10"
TEST_NAME = "rand-res_v3"
N_REPEATS = 20
FIXED_PARAMS = {
    "epochs": 20,
    "batch_size": 64,
    "learning_rate": 0.001,
    "log_steps": 200,
}
TEST_PARAMS = {
    "do_auxloss": [True, False],
    "propagate_gradients": [True, False],
    "use_residuals": [True, False],
}
TEST_PARAMS_combinations = [
    dict(zip(TEST_PARAMS.keys(), values))
    for values in itertools.product(*TEST_PARAMS.values())
]


def run_test(config, repeat, project_name, test_name):
    group = f"{test_name}_{'+auxloss' if config['do_auxloss'] else '-auxloss'}_" \
            f"{'+gradients' if config['propagate_gradients'] else '-gradients'}_" \
            f"{'+residuals' if config['use_residuals'] else '-residuals'}_"
    name = f"{group}_run-{repeat}"

    wandb.init(
        # mode="disabled",
        project=project_name,
        group=group,
        name=name,
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
