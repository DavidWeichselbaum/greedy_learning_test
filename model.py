import torch
import torch.nn as nn


class GreedyClassifier(nn.Module):
    def __init__(self, do_auxloss=False, propagate_gradients=True, use_residuals=False):
        super(GreedyClassifier, self).__init__()
        self.do_auxloss = do_auxloss
        self.propagate_gradients = propagate_gradients
        self.use_residuals = use_residuals

        self.layers = nn.ModuleList([
            # Input: 3 x 32 x 32
            nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2)
            ),   # 32 x 16 x 16
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64), nn.ReLU()
            ),  # 64 x 16 x 16
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2)
            ),  # 64 x 8 x 8
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128), nn.ReLU()
            ),  # 128 x 8 x 8
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2)
            ),  # 128 x 4 x 4
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2)
            ),  # 256 x 2 x 2
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(32 * 16 * 16, 10),
            nn.Linear(64 * 16 * 16, 10),
            nn.Linear(64 * 8 * 8, 10),
            nn.Linear(128 * 8 * 8, 10),
            nn.Linear(128 * 4 * 4, 10),
            nn.Linear(256 * 2 * 2, 10),
        ])

        self.residual_residual_projections = nn.ParameterList([
            nn.Parameter(torch.randn(3, 32), requires_grad=False),    # From input (3) to 32 channels
            nn.Parameter(torch.randn(32, 64), requires_grad=False),   # From 32 to 64 channels
            nn.Parameter(torch.randn(64, 64), requires_grad=False),   # From 64 to 64 channels (no increase)
            nn.Parameter(torch.randn(64, 128), requires_grad=False),  # From 64 to 128 channels
            nn.Parameter(torch.randn(128, 128), requires_grad=False)  # From 128 to 256 channels
        ])

        self.activation_residual_projections = nn.ParameterList([
            nn.Parameter(torch.randn(32, 32), requires_grad=False),    # Activation in Layer 1
            nn.Parameter(torch.randn(64, 64), requires_grad=False),    # Activation in Layer 2 and 3
            nn.Parameter(torch.randn(64, 64), requires_grad=False),    # Activation in Layer 3 (no change)
            nn.Parameter(torch.randn(128, 128), requires_grad=False),  # Activation in Layer 4 and 5
            nn.Parameter(torch.randn(128, 128), requires_grad=False)   # Activation in Layer 6
        ])

        self.residual_batchnorms = nn.ModuleList([
            nn.BatchNorm2d(32),   # After Layer 1
            nn.BatchNorm2d(64),   # After Layer 2
            nn.BatchNorm2d(64),   # After Layer 3
            nn.BatchNorm2d(128),  # After Layer 4
            nn.BatchNorm2d(128)   # After Layer 5
        ])

    def forward(self, x):
        outputs = []

        for i, layer in enumerate(self.layers):
            if not self.propagate_gradients and i > 0:
                x = x.detach()  # Don't propagate gradients through layers

            if self.use_residuals:
                if i == 0:
                    residual = x.clone()
                else:
                    x = x + residual

            x = layer(x)

            x_reshaped = x.view(x.shape[0], -1)
            if not self.do_auxloss and i < len(self.layers) - 1:  #  Last classifier is needed in any case
                x_reshaped = x_reshaped.detach()  # Don't propagate gradients from auxiliary classifiers (use as linear probes)
            output = self.classifiers[i](x_reshaped)
            outputs.append(output)

            if self.use_residuals and i < len(self.layers) - 1:  # last layer needs to prepare no residuals
                pooling = None
                if len(layer) == 4:
                    pooling = layer[-1]
                residual_residual_projection = self.residual_residual_projections[i]
                activation_residual_projection = self.activation_residual_projections[i]
                bnorm = self.residual_batchnorms[i]

                projected_residual = torch.einsum("bdwh,dc->bcwh", residual, residual_residual_projection)
                if pooling:
                    projected_residual = pooling(projected_residual)
                projected_activation = torch.einsum("bdwh,dc->bcwh", x, activation_residual_projection)
                residual = projected_residual + projected_activation
                residual = bnorm(residual)

        return outputs
