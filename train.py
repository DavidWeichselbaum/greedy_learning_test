import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)

            final_output = outputs[-1]  # only care about final classification performance
            loss = criterion(final_output, target)

            val_loss += loss.item()
            val_accuracy += get_accuracy(final_output, target)

    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader)
    model.train()  # Switch back to training mode
    return val_loss, val_accuracy


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

    train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=wandb.config.epochs)


if __name__ == "__main__":
    wandb.init(
        mode="disabled",
        project="greedy_learning_test_MNIST",
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
