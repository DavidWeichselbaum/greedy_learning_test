import itertools
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb

from train import MNISTClassifier, train


test_name = "rand-res_v3"

n_repeats = 20

fixed_params = {
    "epochs": 20,
    "batch_size": 64,
    "learning_rate": 0.001,
    "log_steps": 200,
}
test_params = {
    "do_auxloss": [True, False],
    "propagate_gradients": [True, False],
    "use_residuals": [True, False],
}
test_params_combinations = [
    dict(zip(test_params.keys(), values))
    for values in itertools.product(*test_params.values())
]


def run(config, repeat, test_name):
    group = f"{test_name}_{'+auxloss' if config['do_auxloss'] else '-auxloss'}_" \
            f"{'+gradients' if config['propagate_gradients'] else '-gradients'}_" \
            f"{'+residuals' if config['use_residuals'] else '-residuals'}_"
    name = f"{group}_run-{repeat}"
    tags = [
        "auxloss" if config["do_auxloss"] else "no_auxloss",
        "gradients" if config["propagate_gradients"] else "no_gradients",
        "residuals" if config["use_residuals"] else "no_residuals",
    ]

    wandb.init(
        # mode="disabled",
        project="greedy_learning_test_MNIST",
        group=group,
        name=name,
        config=config,
        tags=tags,
        reinit=True,
    )
    torch.manual_seed(wandb.config["seed"])

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    val_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)

    # Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTClassifier(
        do_auxloss=wandb.config.do_auxloss,
        propagate_gradients=wandb.config.propagate_gradients,
        use_residuals=wandb.config.use_residuals,
    )
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=wandb.config.epochs)


for repeat in range(n_repeats):
    for i, test_params in enumerate(test_params_combinations):
        config = {**fixed_params, **test_params, "seed": random.randint(0, 10000)}
        print(f"Test {i}, repeat {repeat}: {config}")
        run(config, repeat, test_name)
