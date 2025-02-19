import torch
import torch.nn as nn
import torch.optim as optim


class GreedyClassifier(nn.Module):
    def __init__(self, do_auxloss=False, propagate_gradients=True, use_residuals=False):
        super(GreedyClassifier, self).__init__()
        self.do_auxloss = do_auxloss
        self.propagate_gradients = propagate_gradients
        self.use_residuals = use_residuals

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2)),   # 32 x 16 x 16
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2)),  # 64 x 8 x 8
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2))  # 128 x 4 x 4
        ])

        self.classifiers = nn.ModuleList([
            nn.Linear(32 * 16 * 16, 10),
            nn.Linear(64 * 8 * 8, 10),
            nn.Linear(128 * 4 * 4, 10)
        ])

        self.residual_residual_projections = nn.ParameterList([
            nn.Parameter(torch.randn(3, 32), requires_grad=False),  # non trainable, random projections
            nn.Parameter(torch.randn(32, 64), requires_grad=False),
            nn.Parameter(torch.randn(64, 128), requires_grad=False),
        ])
        self.activation_residual_projections = nn.ParameterList([
            nn.Parameter(torch.randn(32, 32), requires_grad=False),  # non trainable, random projections
            nn.Parameter(torch.randn(64, 64), requires_grad=False),
            nn.Parameter(torch.randn(128, 128), requires_grad=False),
        ])

        self.residual_batchnorms = nn.ModuleList([
            nn.BatchNorm2d(32),
            nn.BatchNorm2d(64),
        ])

    def forward(self, x):
        outputs = []

        for i, layer in enumerate(self.layers):
            if not self.propagate_gradients and i > 0:
                x = x.detach()  # Detach if propagate_gradients is False

            if self.use_residuals:
                if i == 0:
                    residual = x.clone()
                else:
                    x = x + residual

            x = layer(x)

            if self.do_auxloss:  # use all classifier layers
                x_reshaped = x.view(x.shape[0], -1)
                output = self.classifiers[i](x_reshaped)
                outputs.append(output)

            if self.use_residuals and i < len(self.layers) - 1:  # last layer needs to prepare no residuals
                pooling = layer[-1]
                residual_residual_projection = self.residual_residual_projections[i]
                activation_residual_projection = self.activation_residual_projections[i]
                bnorm = self.residual_batchnorms[i]

                projected_residual = torch.einsum("bdwh,dc->bcwh", residual, residual_residual_projection)
                projected_residual = pooling(projected_residual)
                projected_activation = torch.einsum("bdwh,dc->bcwh", x, activation_residual_projection)
                residual = projected_residual + projected_activation
                residual = bnorm(residual)

        if not self.do_auxloss:  # only use last classifier layer
            x_reshaped = x.view(x.shape[0], -1)
            output = self.classifiers[-1](x_reshaped)
            outputs.append(output)

        return outputs
