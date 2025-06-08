import pathlib

import numpy as np
import torch
import torchvision
import torchvision.models
from matplotlib import pyplot as plt
from torchvision.transforms import transforms


class ToDeviceLoader:
    def __init__(self, data, device):
        self.data = data
        self.device = device

    def __iter__(self):
        for batch in self.data:
            yield to_device(batch, self.device)

    def __len__(self):
        return len(self.data)


def get_resnet() -> torchvision.models.ResNet:
    res_net = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
    )

    for param in res_net.parameters():
        param.requires_grad = False

    return res_net


def get_resnet_preprocess() -> torchvision.transforms.Compose:
    res_net_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )

    return res_net_transforms


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_all_paths(root: pathlib.Path) -> list[pathlib.Path]:
    return [p for p in root.glob("**/*") if p.is_file()]


def imshow(inp, title=None) -> None:
    """Imshow for Tensor"""

    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(15, 12))
    plt.imshow(inp)

    if title is not None:
        plt.title(title)

    plt.pause(10)


def classification_accuracy(predicted: torch.Tensor, actual: torch.Tensor) -> float:
    _, predictions = torch.max(predicted, dim=1)
    acc = torch.sum(predictions == actual).item() / len(predictions)
    return acc


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
