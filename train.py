import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb


wandb.init(
    project="greedy_learning_test_MNIST",
    # mode="disabled",
    config={
        "epochs": 20,
        "batch_size": 64,
        "learning_rate": 0.001,
        "log_steps": 200,
        "do_auxloss": True,
        "propagate_gradients": False,
    }
)


class MNISTClassifier(nn.Module):
    def __init__(self, do_auxloss=False, propagate_gradients=True):
        super(MNISTClassifier, self).__init__()
        self.do_auxloss = do_auxloss
        self.propagate_gradients = propagate_gradients
        self.use_residuals = True
        self.learned_residuals = True

        self.layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)),   # 32 x 14 x 14
            nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)),  # 64 x 7 x 7
            nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))  # 128 x 3 x 3
        ])

        self.classifiers = nn.ModuleList([
            nn.Linear(32 * 14 * 14, 10),
            nn.Linear(64 * 7 * 7, 10),
            nn.Linear(128 * 3 * 3, 10)
        ])

        self.residual_weights = [nn.Parameter(torch.randn(n)) for n in range(len(self.layers))]

    def forward(self, x):
        outputs = []
        residuals = []
        for i, layer in enumerate(self.layers):
            if not self.propagate_gradients and i > 0:
                x = x.detach()  # Detach if propagate_gradients is False

            if self.use_residuals:
                # print(i)
                # print('x', x.shape)
                # for residual in residuals:
                #     print('res', residual.shape)

                x_original = x
                if residuals:
                    if self.learned_residuals:
                        weighted_residuals = []
                        for residual, residual_weight in zip(residuals, self.residual_weights[i]):
                            weighted_residual = residual * residual_weight
                            weighted_residuals.append(weighted_residual)
                        weighted_residuals = torch.stack(weighted_residuals, dim=0)
                        residual = torch.mean(weighted_residuals, dim=0)
                    else:
                        residual = residuals[-1]  # pseudo-classic residuals
                    x = x + residual

                residuals.append(x_original)

            x = layer(x)

            if self.do_auxloss:  # use all classifier layers
                x_reshaped = x.view(x.shape[0], -1)
                output = self.classifiers[i](x_reshaped)
                outputs.append(output)

            if self.use_residuals and i < len(self.layers) - 1:  # last layer needs to prepare no residuals
                propagated_residuals = []
                for residual in residuals:  # propagate all residuals
                    pooling = layer[-1]  # assumes pooling is last in layer
                    conv = layer[0]  # assumes conv is first layer

                    conv_in = conv.in_channels
                    conv_out = conv.out_channels
                    n_repeat = conv_out // conv_in

                    propagated_residual = pooling(residual)
                    propagated_residual = propagated_residual.repeat(1, n_repeat, 1, 1)
                    propagated_residuals.append(propagated_residual)
                residuals = propagated_residuals

        if not self.do_auxloss:  # only use last classifier layer
            x_reshaped = x.view(x.shape[0], -1)
            output = self.classifiers[-1](x_reshaped)
            outputs.append(output)

        return outputs


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
                train_accuracy /=  train_accumulation_steps
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
    propagate_gradients=wandb.config.propagate_gradients)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
criterion = nn.CrossEntropyLoss()

train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=wandb.config.epochs)
