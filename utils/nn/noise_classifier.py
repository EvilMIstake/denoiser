import pathlib

import torch
import torch.nn as nn
import torchvision


def get_noise_classifier(num_classes: int,
                         parameters_path: pathlib.Path | None = None) -> torchvision.models.ResNet:
    resnet = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    )
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, num_classes)

    if parameters_path is not None:
        model = torch.load(
            parameters_path,
            weights_only=False
        )
        resnet.load_state_dict(model)

    return resnet
