import itertools
import random
from time import sleep

import torch
import wandb

from train import run


PROJECT_NAME = "greedy_learning_test_CIFAR10"
TEST_NAME = "test_big_resid"
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
    "use_residuals": [True],
}
TEST_PARAMS_combinations = [
    dict(zip(TEST_PARAMS.keys(), values))
    for values in itertools.product(*TEST_PARAMS.values())
]


def run_test(config, repeat, project_name, test_name):
    group = f"{test_name}_{'+auxloss' if config['do_auxloss'] else '-auxloss'}" \
            f"_{'+gradients' if config['propagate_gradients'] else '-gradients'}" \
            f"_{'+resid' if config['use_residuals'] else '-resid'}"
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
            try:
                run_test(config, repeat, PROJECT_NAME, TEST_NAME)
            except KeyboardInterrupt:
                sleep(0.5)  # allow for killing
                print("Skipped")
