from copy import deepcopy

import torch
import torch.nn as nn


class GreedyClassifier(nn.Module):
    def __init__(self, do_auxloss=False, propagate_gradients=True, residual_mode=None, classifier_mode="dense-s1"):
        super(GreedyClassifier, self).__init__()
        self.do_auxloss = do_auxloss
        self.propagate_gradients = propagate_gradients
        assert residual_mode in [None, "regular"]
        self.residual_mode = residual_mode
        assert classifier_mode in ["dense", "dense-s1", "average"]
        self.classifier_mode = classifier_mode

        self.do_linear_probes = not self.do_auxloss
        self.do_deep_supervision = self.do_auxloss and self.propagate_gradients

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
        if self.classifier_mode == "dense":
            self.classifiers = nn.ModuleList([
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(32 * 16 * 16, 10),
                ),
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(64 * 16 * 16, 10),
                ),
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(64 * 8 * 8, 10),
                ),
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128 * 8 * 8, 10),
                ),
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128 * 4 * 4, 10),
                ),
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256 * 2 * 2, 10),
                ),
            ])
        elif self.classifier_mode == "dense-s1":
            self.classifiers = nn.ModuleList([
                nn.Sequential(
                    deepcopy(self.layers[1]),
                    nn.Flatten(),
                    nn.Linear(64 * 16 * 16, 10),
                ),
                nn.Sequential(
                    deepcopy(self.layers[2]),
                    nn.Flatten(),
                    nn.Linear(64 * 8 * 8, 10),
                ),
                nn.Sequential(
                    deepcopy(self.layers[3]),
                    nn.Flatten(),
                    nn.Linear(128 * 8 * 8, 10),
                ),
                nn.Sequential(
                    deepcopy(self.layers[4]),
                    nn.Flatten(),
                    nn.Linear(128 * 4 * 4, 10),
                ),
                nn.Sequential(
                    deepcopy(self.layers[5]),
                    nn.Flatten(),
                    nn.Linear(256 * 2 * 2, 10),
                ),
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256 * 2 * 2, 10),
                ),
            ])
        elif self.classifier_mode == "average":  # Use global average pooling before linear classifiers
            self.classifiers = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(32, 10),
                ),
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(64, 10),
                ),
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(64, 10),
                ),
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(128, 10),
                ),
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(128, 10),
                ),
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(256, 10),
                ),
            ])

        if self.residual_mode == "regular":
            self.residual_downsample_layers = nn.ModuleList([
                # Input: 3 x 32 x 32
                nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32), nn.MaxPool2d(2, 2)
                ),   # 32 x 16 x 16
                nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64), nn.Identity()
                ),  # 64 x 16 x 16
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64), nn.MaxPool2d(2, 2)
                ),  # 64 x 8 x 8
                nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128), nn.Identity()
                ),  # 128 x 8 x 8
                nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128), nn.MaxPool2d(2, 2)
                ),  # 128 x 4 x 4
                nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256), nn.MaxPool2d(2, 2)
                ),  # 256 x 2 x 2
                ])

    def forward(self, x):
        outputs = []
        residual = None

        for i, layer in enumerate(self.layers):
            if not self.propagate_gradients and i > 0:
                x = x.detach()  # Don't propagate gradients through layers
                if residual is not None:
                    residual = residual.detach()

            if self.residual_mode:
                layer_input = x
                if i == 0:
                    residual = x.clone()
                else:
                    x = x + residual

            x = layer(x)

            x_classifier = x.clone()
            if self.do_linear_probes and i < len(self.layers) - 1:  #  Last classifier is needed in any case
                x_classifier = x_classifier.detach()  # Don't propagate gradients from auxiliary classifiers (use as linear probes)

            classifier = self.classifiers[i]
            output = classifier(x_classifier)
            outputs.append(output)

            if self.residual_mode and i < len(self.layers) - 1:  # last layer needs to prepare no residuals
                if self.residual_mode == "regular":
                    residual_downsample_layer = self.residual_downsample_layers[i]
                    residual = residual_downsample_layer(layer_input)

        return outputs
