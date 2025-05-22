import torch
import torchvision.models

from utils import (
    utils,
    __SRC__
)


if __name__ == "__main__":
    with open(__SRC__ / "image_net_classes.txt", "r") as f_stream:
        image_net_classes = f_stream.read().split("\n")

    batch_size = 256
    workers = 8

    res_net = utils.get_resnet()
    device = utils.get_device()
    res_net.to(device)
    res_net.eval()

    classification_path = __SRC__ / "imagenet-mini-poisson/"
    dataset = torchvision.datasets.ImageFolder(
        classification_path,
        transform=utils.get_resnet_preprocess()
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True,
        pin_memory=True
    )
    to_device_data_loader = utils.ToDeviceLoader(
        data_loader,
        device
    )

    acc_numerator = .0
    acc_denominator = .0

    with torch.no_grad():
        for i, batch in enumerate(to_device_data_loader):
            images, labels = batch
            out = res_net(images)
            accuracy = utils.classification_accuracy(out, labels)
            print(f"Batch[{i}]: {accuracy:.3f}")
            acc_numerator += accuracy * len(batch)
            acc_denominator += len(batch)

    total_accuracy = acc_numerator / acc_denominator
    print(f"Total accuracy: {total_accuracy:.3f}")
