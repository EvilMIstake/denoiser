import pathlib

import torch
import torchvision.models
import tqdm

from utils import (
    utils,
    __SRC__
)


def _evaluate(model: torch.nn.Module, d_loader: utils.ToDeviceLoader) -> float:
    acc_numerator = .0
    acc_denominator = .0

    model.eval()

    with torch.no_grad():
        with tqdm.tqdm(d_loader, total=len(d_loader)) as tqdm_:
            tqdm_.set_description("Batch[0] | Total accuracy[0]")

            for i, batch in enumerate(tqdm_):
                images, labels = batch
                out = model(images)
                accuracy = utils.classification_accuracy(out, labels)
                acc_numerator += accuracy * len(batch)
                acc_denominator += len(batch)
                tqdm_.set_description(f"Batch[{i}] | Total accuracy[{acc_numerator / acc_denominator:.3f}]")

    total_accuracy = acc_numerator / acc_denominator

    return total_accuracy


def classification(path_with_labels: pathlib.Path,
                   transform: torchvision.transforms.Compose,
                   num_workers: int = 8,
                   batch_size: int = 256) -> float:
    res_net = utils.get_resnet()
    device = utils.get_device()
    res_net.to(device)
    res_net.eval()

    dataset = torchvision.datasets.ImageFolder(
        path_with_labels,
        transform=transform
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    to_device_data_loader = utils.ToDeviceLoader(
        data_loader,
        device
    )

    accuracy = _evaluate(res_net, to_device_data_loader)

    return accuracy


if __name__ == "__main__":
    pwl = __SRC__ / "imagenet-mini-impulse/"
    b_size = 256
    workers = 8

    acc = classification(
        pwl,
        utils.get_resnet_preprocess(),
        workers,
        b_size
    )
