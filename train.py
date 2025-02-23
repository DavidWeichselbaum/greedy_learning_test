import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchsummary import summary
import wandb
import pandas as pd

from model import GreedyClassifier
from utils import get_commit_hash


def train(model, train_loader, val_loader, optimizers, criterion, device, num_epochs):
    outputs_df = pd.DataFrame(columns=["Steps", "Output", "Val Loss", "Val Accuracy"])
    model.train()
    # wandb.watch(model, log="all")  # Log gradients and model parameters

    train_loss, train_accuracy, train_accumulation_steps = 0, 0, 0
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            outputs = model(data)

            if model.do_deep_supervision:  # global optimizer
                optimizer = optimizers
                optimizer.zero_grad()
                loss = 0
                for i, output in enumerate(outputs):
                    loss += criterion(output, target)
                loss /= len(outputs)
                loss.backward()
                optimizer.step()
            else:  # separate optimizer for each auxiliary head or linear probe
                for i, (output, optimizer) in enumerate(zip(outputs, optimizers)):
                    optimizer.zero_grad()
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

            train_loss += loss.item()  # only care about last loss
            train_accuracy += get_accuracy(output, target)  # only care about final classification performance

            total_steps = epoch * len(train_loader) + batch_idx
            train_accumulation_steps += 1
            if total_steps % wandb.config.log_steps == 0:
                train_loss /= train_accumulation_steps
                train_accuracy /= train_accumulation_steps

                val_loss, val_accuracy, outputs_df = validate(model, val_loader, criterion, device, total_steps, outputs_df)
                if not model.propagate_gradients:
                    compare_gradients(model, val_loader, device, criterion)

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


def log_outputs_table(total_steps, losses, accuracies, outputs_df):
    new_rows = []
    for output_idx, (loss, accuracy) in enumerate(zip(losses, accuracies)):
        new_row = {
            "Steps": total_steps,
            "Output": output_idx,
            "Val Loss": loss,
            "Val Accuracy": accuracy
        }
        new_rows.append(new_row)
    new_df = pd.DataFrame(new_rows)
    outputs_df = pd.concat([outputs_df, new_df], ignore_index=True)
    wandb.log({"Classifier Head Metrics": wandb.Table(dataframe=outputs_df), "Steps": total_steps})
    return outputs_df


def log_outputs_plot(total_steps, losses, accuracies):
    output_labels = [f"Output {i+1}" for i in range(len(losses))]

    losses_data = [[label, loss] for label, loss in zip(output_labels, losses)]
    losses_table = wandb.Table(data=losses_data, columns=["Output", "Loss"])
    losses_plot = wandb.plot.bar(losses_table, "Output", "Loss", title="Validation Losses for Each Output")

    accuracies_data = [[label, acc] for label, acc in zip(output_labels, accuracies)]
    accuracies_table = wandb.Table(data=accuracies_data, columns=["Output", "Accuracy"])
    accuracies_plot = wandb.plot.bar(accuracies_table, "Output", "Accuracy", title="Validation Accuracies for Each Output")

    wandb.log(
        {"Validation Accuracies": accuracies_plot, "Validation Losses": losses_plot},
        step=total_steps)


def validate(model, val_loader, criterion, device, total_steps, outputs_df):
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
        outputs_df = log_outputs_table(total_steps, avg_val_losses, avg_val_accuracies, outputs_df)
        log_outputs_plot(total_steps, avg_val_losses, avg_val_accuracies)

    return avg_val_losses[-1], avg_val_accuracies[-1], outputs_df


def compare_gradients(model, val_loader, device, criterion):
    model.eval()
    layer_types = {name: type(module).__name__ for name, module in model.named_modules()}

    for data, target in val_loader:
        data, target = data.to(device), target.to(device)

        grads_backprop = get_backprop_grads(model, data, target, criterion)
        grads_no_backprop = get_no_backprop_grads(model, data, target, criterion)

        for param_name in grads_backprop.keys():
            grad_backprop = grads_backprop[param_name]
            grad_no_backprop = grads_no_backprop[param_name]

            cos_sim = torch.nn.functional.cosine_similarity(grad_no_backprop.view(-1), grad_backprop.view(-1), dim=0)
            norm_ratio = grad_no_backprop.norm() / grad_backprop.norm()

            layer_name = '.'.join(param_name.split('.')[:-1])  # Extract the layer name without "weight" or "bias"
            layer_type = layer_types.get(layer_name, 'Unknown')

            print(f"{param_name:>20}    {layer_type:>20}    cos sim: {cos_sim.item():.4f}    norm ratio: {norm_ratio.item():.4f}")

        return  # only do one batch
    model.train()


def get_backprop_grads(model, data, target, criterion):
    model.propagate_gradients = True
    outputs = model(data)
    model.propagate_gradients = False

    final_output = outputs[-1]
    loss = criterion(final_output, target)

    model.zero_grad()
    loss.backward()

    grads_backprop = {}
    for param_name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad = param.grad.clone()
            grads_backprop[param_name] = grad
    return grads_backprop


def get_no_backprop_grads(model, data, target, criterion):
    outputs = model(data)

    model.zero_grad()
    for i, output in enumerate(outputs):
        loss = criterion(output, target)
        loss.backward()

    grads_no_backprop = {}
    for param_name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad = param.grad.clone()
            grads_no_backprop[param_name] = grad
    return grads_no_backprop


def init_optimizers(model):
    if model.do_deep_supervision:
        return init_global_optimizer(model)
    else:
        return init_separate_optimizers(model)


def init_separate_optimizers(model):
    optimizers = []
    for i, (layer, classifier) in enumerate(zip(model.layers, model.classifiers)):
        print(f"Layer {i+1}")
        if not model.do_auxloss and i == len(model.layers) - 1:  # final classifier. if no auxloss, last optimizers gets all layers and last classifier
            parameters = list(model.layers.parameters()) + list(model.classifiers[-1].parameters())
            print(f"Parameters final classifier: {sum(p.numel() for p in parameters)}")
        elif model.do_linear_probes:  # linear probes
            parameters = list(classifier.parameters())
            print(f"Parameters linear probe: {sum(p.numel() for p in parameters)}")
        else:  #  auxiliary classifier. Gets all parameter of single layer
            parameters = list(layer.parameters()) + list(classifier.parameters())
            print(f"Parameters auxiliary classifier: {sum(p.numel() for p in parameters)}")

        optimizer = optim.Adam(parameters, lr=wandb.config.learning_rate)
        optimizers.append(optimizer)
    return optimizers


def init_global_optimizer(model):
    return optim.Adam(model.parameters(), lr=wandb.config.learning_rate)


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
        residual_mode=wandb.config.residual_mode,
    )
    model.to(device)
    summary(model, input_size=(3, 32, 32))

    optimizers = init_optimizers(model)
    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, val_loader, optimizers, criterion, device, wandb.config.epochs)


if __name__ == "__main__":
    config={
        "epochs": 20,
        "batch_size": 64,
        "learning_rate": 0.001,
        "log_steps": 200,
        "seed": 42,
        "do_auxloss": True,
        "propagate_gradients": False,
        "residual_mode": None,
    }
    name = f"test_{'+auxloss' if config['do_auxloss'] else '-auxloss'}" \
           f"_{'+gradients' if config['propagate_gradients'] else '-gradients'}" \
           f"_resid={config['residual_mode']}"
    wandb.init(
        mode="disabled",
        project="greedy_learning_test_CIFAR10",
        group=name,
        name=name,
        notes=get_commit_hash(),
        config=config,
    )
    torch.manual_seed(wandb.config["seed"])
    run()
