import itertools
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb

from train import MNISTClassifier, train


n_repeats = 5

fixed_params = {
    "epochs": 12,
    "batch_size": 64,
    "learning_rate": 0.001,
    "log_steps": 200,
}
test_params = {
    "do_auxloss": [True, False],
    "propagate_gradients": [True, False],
    "use_residuals": [True, False],
    "learned_residuals": [True, False]
}
test_params_combinations = [
    dict(zip(test_params.keys(), values))
    for values in itertools.product(*test_params.values())
]


for test_params in test_params_combinations:
    for repeat in range(n_repeats):
        config = {**fixed_params, **test_params, "seed": random.randint(0, 10000)}
        group = f"{'+auxloss' if config['do_auxloss'] else '-auxloss'}_" \
                f"{'+gradients' if config['propagate_gradients'] else '-gradients'}_" \
                f"{'+residuals' if config['use_residuals'] else '-residuals'}_" \
                f"{'+learned_residuals' if config['learned_residuals'] else '-learned_residuals'}"
        name = f"test2_{group}_run-{repeat}"
        tags = [
            "auxloss" if config["do_auxloss"] else "no_auxloss",
            "gradients" if config["propagate_gradients"] else "no_gradients",
            "residuals" if config["use_residuals"] else "no_residuals",
            "learned_residuals" if config["learned_residuals"] else "no_learned_residuals",
        ]
        print(f"Run Name: {name}")
        print(f"Config: {config}\n")

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
            learned_residuals=wandb.config.learned_residuals,
        )
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=wandb.config.epochs)
