import datetime
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchvision.datasets
from torch import nn
from tqdm import tqdm

from utils import (
    utils,
    __SRC__,
    __MODEL_SRC__
)
from utils.nn import config
from utils.nn.noise_classifier import get_noise_classifier

_SEED = 1359140914
_DEVICE = utils.get_device()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(_SEED)
np.random.seed(_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(_SEED)
    torch.cuda.manual_seed_all(_SEED)


@dataclass
class Measurements:
    train_acc: list[float]
    train_loss: list[float]
    val_acc: list[float]
    val_loss: list[float]

    def to_pandas(self) -> pd.DataFrame:
        data_frame = pd.DataFrame(
            {
                "TRAIN_ACC": self.train_acc,
                "TRAIN_LOSS": self.train_loss,
                "VAL_ACC": self.val_acc,
                "VAL_LOSS": self.val_loss
            }
        )
        return data_frame


def _train(train_dataloader: utils.ToDeviceLoader,
           val_dataloader: utils.ToDeviceLoader,
           num_classes: int) -> None:
    cnn = get_noise_classifier(num_classes)
    cnn = utils.to_device(cnn, _DEVICE)

    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer = optim.Adam(
        cnn.parameters(),
        lr=config.Config.learning_rate,
        weight_decay=config.Config.weight_decay
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            config.Config.milestone * i
            for i in range(1, config.Config.num_epochs // config.Config.milestone)
        ],
        gamma=config.Config.gamma
    )

    states_path = pathlib.Path("model_states")
    current_date = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    model_name = f"NoiseClassification_{current_date}"
    current_states_path = states_path / "NClass" / model_name
    current_states_path.mkdir(parents=True, exist_ok=True)
    measurements = Measurements(
        train_acc=[],
        train_loss=[],
        val_acc=[],
        val_loss=[]
    )
    measurements_path = current_states_path / "measurements.csv"

    n_test, n_val = len(train_dataloader), len(val_dataloader)

    for epoch in range(config.Config.num_epochs):
        # Train step
        cnn.train()
        total_acc_train = .0
        total_loss_train = .0

        with tqdm(total=len(train_dataloader) * config.Config.batch_size) as tqdm_:
            tqdm_.set_description(f"Train Epoch {epoch + 1}/{config.Config.num_epochs}")

            for noised, labels in train_dataloader:
                optimizer.zero_grad()
                outputs = cnn(noised)
                prediction = torch.argmax(outputs, -1)
                loss_train = criterion(outputs, labels)
                loss_train.backward()
                optimizer.step()

                # Add batch ACC/LOSS values
                total_acc_train += int(torch.sum(prediction == labels.data)) / len(noised)
                # Add mean batch loss
                total_loss_train += loss_train.item() / len(noised)

                tqdm_.update(len(noised))
                tqdm_.set_postfix_str(
                    f"LOSS: {total_loss_train / n_test:.4f} | "
                    f"ACC: {total_acc_train / n_test: .4f}"
                )

        # Validation step
        cnn.eval()
        total_acc_val = .0
        total_loss_val = .0

        with tqdm(total=len(val_dataloader) * config.Config.batch_size) as tqdm_:
            tqdm_.set_description(f"Val Epoch {epoch + 1}/{config.Config.num_epochs}")

            with torch.no_grad():
                for noised, labels in val_dataloader:
                    outputs = cnn(noised)
                    prediction = torch.argmax(outputs, -1)
                    loss_val = criterion(outputs, labels)

                    # Add batch ACC/LOSS values
                    total_acc_val += int(torch.sum(prediction == labels.data)) / len(noised)
                    # Add mean batch loss
                    total_loss_val += loss_val.item() / len(noised)

                    tqdm_.update(len(noised))
                    tqdm_.set_postfix_str(
                        f"LOSS: {total_loss_val / n_val:.4f} | "
                        f"ACC: {total_acc_val / n_val: .4f}"
                    )

        scheduler.step()

        # Save metrics
        total_acc_train /= n_test
        total_loss_train /= n_test
        total_acc_val /= n_val
        total_loss_val /= n_val

        # Extract model
        state_path = current_states_path / f"{epoch}_epoch.pth"
        torch.save(cnn.state_dict(), state_path)

        measurements.train_loss.append(total_loss_train)
        measurements.train_acc.append(total_acc_train)
        measurements.val_acc.append(total_acc_val)
        measurements.val_loss.append(total_loss_val)

        measurements_table = measurements.to_pandas()
        measurements_table.to_csv(
            measurements_path,
            sep="\t",
            index=False
        )


def train(noised_image_path: pathlib.Path,
          num_classes: int) -> None:
    train_noised_img_path = noised_image_path / "train"
    val_noised_img_path = noised_image_path / "val"
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )

    # Prepare train data
    train_dataset = torchvision.datasets.ImageFolder(
        train_noised_img_path,
        transform=transform
    )
    train_dl = utils.ToDeviceLoader(
        torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.Config.batch_size,
            num_workers=config.Config.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        ),
        _DEVICE
    )

    # Prepare val data
    val_dataset = torchvision.datasets.ImageFolder(
        val_noised_img_path,
        transform=transform
    )
    val_dl = utils.ToDeviceLoader(
        torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.Config.batch_size,
            num_workers=config.Config.num_workers,
            pin_memory=True,
            drop_last=True
        ),
        _DEVICE
    )

    # Training
    _train(
        train_dl,
        val_dl,
        num_classes
    )


def _test(test_dataloader: utils.ToDeviceLoader,
          model_path: pathlib.Path,
          num_classes: int) -> None:
    cnn = get_noise_classifier(num_classes, model_path)
    cnn = utils.to_device(cnn, _DEVICE)

    for param in cnn.parameters():
        param.requires_grad = False

    cnn.eval()

    n_test = len(test_dataloader)
    total_acc_train = .0

    with tqdm(total=len(test_dataloader) * config.Config.batch_size) as tqdm_:
        for noised, labels in test_dataloader:
            prediction = torch.argmax(cnn(noised), -1)

            # Add batch ACC/LOSS values
            total_acc_train += int(torch.sum(prediction == labels.data)) / len(noised)

            tqdm_.update(len(noised))
            tqdm_.set_postfix_str(f"ACC: {total_acc_train / n_test: .4f}")


def test(noised_image_path: pathlib.Path,
         model_path: pathlib.Path,
         num_classes: int) -> None:
    test_noised_img_path = noised_image_path / "train"
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )

    # Prepare test data
    test_dataset = torchvision.datasets.ImageFolder(
        test_noised_img_path,
        transform=transform
    )
    test_dl = utils.ToDeviceLoader(
        torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.Config.batch_size,
            num_workers=config.Config.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        ),
        _DEVICE
    )

    _test(test_dl, model_path, num_classes)


if __name__ == '__main__':
    # Training
    noised_img_path = __SRC__ / "BSDS500-noised_classes"
    model_path_ = __MODEL_SRC__ / "model_classifier.pth"
    nc = 6

    # train(noised_img_path, nc)
    test(noised_img_path, model_path_, nc)
