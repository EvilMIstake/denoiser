import pathlib

import torch
from torch import nn


class DnCNN(nn.Module):
    def __init__(self,
                 num_layers=17,
                 num_features=64,
                 parameters_path: pathlib.Path | None = None):
        assert num_layers > 2
        super().__init__()

        num_channels = 3
        kernel_size = 3
        padding = 1

        layers = [
            nn.Sequential(
                nn.Conv2d(
                    num_channels,
                    num_features,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False
                ),
                nn.ReLU(inplace=True)
            )
        ]
        for i in range(num_layers - 2):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        num_features,
                        num_features,
                        kernel_size=kernel_size,
                        padding=padding,
                        bias=False
                    ),
                    nn.BatchNorm2d(
                        num_features,
                        eps=1e-4,
                        momentum=0.95
                    ),
                    nn.ReLU(inplace=True)
                )
            )
        layers.append(
            nn.Conv2d(
                num_features,
                num_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False
            )
        )
        self.layers = nn.Sequential(*layers)

        if parameters_path is None:
            self._initialize_weights()
        else:
            self.load_state_dict(
                torch.load(
                    parameters_path,
                    weights_only=False
                )
            )

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs) -> torch.Tensor:
        y = inputs
        residual = self.layers(y)
        return y - residual
