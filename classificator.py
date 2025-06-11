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


def classification(model: torch.nn.Module,
                   dataset: torchvision.datasets.ImageFolder,
                   num_workers: int = 8,
                   batch_size: int = 256) -> float:
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    to_device_data_loader = utils.ToDeviceLoader(
        data_loader,
        utils.get_device()
    )

    accuracy = _evaluate(model, to_device_data_loader)

    return accuracy


if __name__ == "__main__":
    pwl = __SRC__ / "imagenet-mini-add/"
    b_size = 256
    workers = 8

    model_ = utils.get_resnet()
    for param in model_.parameters():
        param.requires_grad = False
    model_.to(utils.get_device())
    model_.eval()

    dataset_ = torchvision.datasets.ImageFolder(
        pwl,
        transform=utils.get_resnet_preprocess()
    )

    acc = classification(
        model_,
        dataset_,
        workers,
        b_size
    )
