import pathlib

import torch
from torch import nn


class DnCNN(nn.Module):
    def __init__(self,
                 num_layers: int = 17,
                 num_channels: int = 3,
                 num_features: int = 64,
                 kernel_size: int = 3,
                 padding: int = 1,
                 parameters_path: pathlib.Path | None = None,
                 bias_input: bool = True,
                 bias_mid: bool = False,
                 bias_output: bool = False,
                 residual: bool = True):
        assert num_layers > 2
        super().__init__()

        layers = [
            nn.Sequential(
                nn.Conv2d(
                    num_channels,
                    num_features,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=bias_input
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
                        bias=bias_mid
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
                bias=bias_output
            )
        )
        self.layers = nn.Sequential(*layers)

        if parameters_path is None:
            self._initialize_weights()
        else:
            model = torch.load(
                parameters_path,
                weights_only=False
            )
            self.load_state_dict(model)

        self.__output = (
            (lambda inp: inp - self.layers(inp))
            if residual else
            (lambda inp: self.layers(inp))
        )

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.__output(inputs)
