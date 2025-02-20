import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchsummary import summary
import wandb

from model import GreedyClassifier


def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=5):
    model.train()
    # wandb.watch(model, log="all")  # Log gradients and model parameters

    train_loss, train_accuracy, train_accumulation_steps = 0, 0, 0
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            outputs = model(data)

            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss = loss / len(outputs)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            final_output = outputs[-1]  # only care about final classification performance
            train_accuracy += get_accuracy(final_output, target)
            train_accumulation_steps += 1

            total_steps = epoch * len(train_loader) + batch_idx
            if total_steps % wandb.config.log_steps == 0:
                train_loss /= train_accumulation_steps
                train_accuracy /= train_accumulation_steps
                val_loss, val_accuracy = validate(model, val_loader, criterion, device)
                wandb.log({
                    "Steps": total_steps,
                    "Epoch": epoch + 1,
                    "Batch": batch_idx,
                    "Train Loss": train_loss,
                    "Val Loss": val_loss,
                    "Train Accuracy": train_accuracy,
                    "Val Accuracy": val_accuracy,
                })
                print(f"Epoch {epoch+1}, Batch {batch_idx},"
                      f" Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f},"
                      f" Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")
                train_loss, train_accuracy, train_accumulation_steps = 0, 0, 0

    # Save model checkpoint
    # torch.save(model.state_dict(), "mnist_model.pth")
    # wandb.save("mnist_model.pth")


def get_accuracy(output, target):
    _, predicted = torch.max(output, dim=1)
    correct = (predicted == target).sum().item()
    total = target.size(0)
    accuracy = correct / total
    return accuracy


def log_validation_bars(losses, accuracies):
    output_labels = [f"Output {i+1}" for i in range(len(losses))]

    losses_data = [[label, loss] for label, loss in zip(output_labels, losses)]
    losses_table = wandb.Table(data=losses_data, columns=["Output", "Loss"])
    losses_plot = wandb.plot.bar(losses_table, "Output", "Loss", title="Validation Losses for Each Output")

    accuracies_data = [[label, acc] for label, acc in zip(output_labels, accuracies)]
    accuracies_table = wandb.Table(data=accuracies_data, columns=["Output", "Accuracy"])
    accuracies_plot = wandb.plot.bar(accuracies_table, "Output", "Accuracy", title="Validation Accuracies for Each Output")

    wandb.log({"Validation Accuracies": accuracies_plot, "Validation Losses": losses_plot})


def validate(model, val_loader, criterion, device):
    val_losses = []
    val_accuracies = []
    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)

            if not val_losses:
                val_losses = [[] for _ in range(len(outputs))]
                val_accuracies = [[] for _ in range(len(outputs))]

            for i, output in enumerate(outputs):
                loss = criterion(output, target)
                val_losses[i].append(loss.item())
                val_accuracies[i].append(get_accuracy(output, target))
    model.train()

    avg_val_losses = [sum(losses) / len(losses) for losses in val_losses]
    avg_val_accuracies = [sum(accs) / len(accs) for accs in val_accuracies]

    if len(avg_val_losses) > 1:
        log_validation_bars(avg_val_losses, avg_val_accuracies)

    return avg_val_losses[-1], avg_val_accuracies[-1]


def run():
    # Dataset
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    val_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)

    # Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GreedyClassifier(
        do_auxloss=wandb.config.do_auxloss,
        propagate_gradients=wandb.config.propagate_gradients,
        use_residuals=wandb.config.use_residuals,
    )
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    summary(model, input_size=(3, 32, 32))

    train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=wandb.config.epochs)


if __name__ == "__main__":
    wandb.init(
        mode="disabled",
        project="greedy_learning_test_CIFAR10",
        name="test",
        config={
            "epochs": 20,
            "batch_size": 64,
            "learning_rate": 0.001,
            "log_steps": 200,
            "seed": 42,
            "do_auxloss": True,
            "propagate_gradients": False,
            "use_residuals": True,
        }
    )
    torch.manual_seed(wandb.config["seed"])
    run()
