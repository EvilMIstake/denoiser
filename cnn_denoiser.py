import datetime
import pathlib
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.optim as optim
import numpy as np

import cv2 as cv
from tqdm import tqdm
import pandas as pd

from utils.nn.dncnn import DnCNN
from utils.nn import utils as nn_utils
from utils import (
    utils,
    metrics,
    __SRC__,
    __MODEL_STATES__
)


_SEED = 0
_DEVICE = utils.get_device()
torch.manual_seed(_SEED)


@dataclass
class Measurements:
    train_psnr: list[float]
    train_loss: list[float]

    def to_pandas(self) -> pd.DataFrame:
        data_frame = pd.DataFrame(
            {
                "TRAIN_PSNR": self.train_psnr,
                "TRAIN_LOSS": self.train_loss
            }
        )
        return data_frame


def train(train_noised: pathlib.Path,
          train_cleaned: pathlib.Path,
          num_layers: int,
          patch_size: int,
          batch_size: int,
          workers: int,
          num_epochs: int,
          learning_rate: float,
          postfix: str,
          resize_size: int = 128,
          weight_decay: float = 1e-4,
          parameters_path: pathlib.Path | None = None) -> None:
    cnn = DnCNN(
        num_layers=num_layers,
        parameters_path=parameters_path
    )
    cnn.to(_DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    dataset = nn_utils.DnCnnDataset(
        noised_data_path=train_noised,
        cleaned_data_path=train_cleaned,
        resize_size=resize_size,
        patch_size=patch_size
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    to_device_loader = utils.ToDeviceLoader(
        data_loader,
        _DEVICE
    )

    states_path = pathlib.Path("model_states")
    current_date = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    model_name = f"Model_{postfix}_{num_layers}l_{num_epochs}e_{patch_size}p_{current_date}"
    current_states_path = states_path / "DnCNN" / model_name
    current_states_path.mkdir(parents=True, exist_ok=True)
    measurements = Measurements(
        train_psnr=[],
        train_loss=[]
    )

    for epoch in range(num_epochs):
        with tqdm(total=(len(dataset) - len(dataset) % batch_size)) as tqdm_:
            tqdm_.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            total_psnr_train = .0
            total_loss_train = .0

            for data in to_device_loader:
                noised, real = data
                n = len(noised)

                optimizer.zero_grad()
                prediction = cnn(noised)
                loss_train = criterion(prediction, real)
                loss_train.backward()
                optimizer.step()

                numpy_predicted_batch, numpy_real_batch = (
                    np.uint8(prediction.detach().cpu().numpy() * 255.),
                    np.uint8(real.detach().cpu().numpy() * 255.)
                )
                total_psnr_train += metrics.peak_signal_to_noise_ratio(
                    numpy_predicted_batch[0],
                    numpy_real_batch[1]
                ) * n
                total_loss_train += loss_train.item() * n

                tqdm_.update(n)
                tqdm_.set_postfix_str(f"{total_loss_train / len(dataset):.4f}")

        scheduler.step()
        total_psnr_train /= len(dataset)
        total_loss_train /= len(dataset)

        if (epoch + 1) % 5 == 0:
            state_path = current_states_path / f"{epoch}_epoch.pth"
            torch.save(cnn.state_dict(), state_path)

        measurements.train_loss.append(total_loss_train)
        measurements.train_psnr.append(total_psnr_train)

    measurements_path = current_states_path / "measurements.csv"
    measurements_table = measurements.to_pandas()
    measurements_table.to_csv(
        measurements_path,
        sep="\t",
        index=False
    )


def test(parameters_path: pathlib.Path,
         test_path: pathlib.Path,
         real_path: pathlib.Path,
         num_layers: int,
         patch_size: int) -> None:
    def from_patches(img_tensor: torch.Tensor, y: int, x: int) -> np.ndarray:
        img_tensor = torch.clamp(img_tensor, 0, 1)

        num_patches, *_ = img_tensor.shape
        row_length = num_patches // math.ceil(x / patch_size)

        numpy_patches = np.transpose(
            np.uint8(img_tensor.detach().numpy() * 255.),
            axes=(0, 2, 3, 1)
        )
        row_patches = np.split(numpy_patches, row_length)
        rows = [np.hstack(row) for row in row_patches]
        img_numpy = np.vstack(rows)[:y, :x, :]

        return img_numpy

    cnn = DnCNN(
        num_layers=num_layers,
        parameters_path=parameters_path
    )
    cnn.eval()

    dataset = nn_utils.DnCnnDatasetTest(
        noised_data_path=test_path,
        cleaned_data_path=real_path,
        patch_size=patch_size
    )

    with torch.no_grad():
        data = dataset[123]
        noised_img_tensor, cleaned_img_numpy = data

        # denoised_img_tensor = cnn(noised_img_tensor[None, :, :, :])[0]
        denoised_img_tensor = cnn(noised_img_tensor)
        y, x, *_ = cleaned_img_numpy.shape

        noised_img_numpy = from_patches(noised_img_tensor, y, x)
        denoised_img_numpy = from_patches(denoised_img_tensor, y, x)
        images = np.vstack((cleaned_img_numpy, noised_img_numpy, denoised_img_numpy))
        # # noinspection PyUnresolvedReferences
        cv.imwrite("temp.png", images)


if __name__ == "__main__":
    # Num layers
    nl = 17
    # Patch size
    ps = 50
    # Num epochs
    ne = 50
    # Batch size
    bs = 128
    # Workers
    ws = 2
    # Learning rate
    lr = 1e-4
    # Weight decay
    wd = 0
    # Postfix
    px = "noised-add-impulse"

    model_parameters = __MODEL_STATES__ / "DnCNN" / "Model_noised-add-impulse_17l_50e_50p_2025-04-23T181245" / "39_epoch.pth"
    # model_parameters = None

    dataset_name = f"imagenet-mini-shrink"
    noised_img_path = __SRC__ / f"{dataset_name}-{px}" / "train"
    real_img_path = __SRC__ / dataset_name / "train"

    # TRAINING
    # train(
    #     noised_img_path, real_img_path,
    #     nl, ps, bs, ws, ne, lr, px,
    #     weight_decay=wd,
    #     parameters_path=model_parameters
    # )

    # TESTING
    test(
        model_parameters,
        noised_img_path,
        real_img_path,
        num_layers=nl,
        patch_size=ps
    )
