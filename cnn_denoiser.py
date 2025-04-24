import datetime
import pathlib
from dataclasses import dataclass

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
    val_psnr: list[float]
    val_loss: list[float]

    def to_pandas(self) -> pd.DataFrame:
        data_frame = pd.DataFrame(
            {
                "TRAIN_PSNR": self.train_psnr,
                "TRAIN_LOSS": self.train_loss,
                "VAL_PSNR": self.val_psnr,
                "VAL_LOSS": self.val_loss
            }
        )
        return data_frame


def _train(train_dataloader: utils.ToDeviceLoader,
           val_dataloader: utils.ToDeviceLoader,
           postfix: str,
           parameters_path: pathlib.Path | None = None) -> None:
    cnn = DnCNN(
        num_layers=nn_utils.Config.num_layers,
        parameters_path=parameters_path
    )
    cnn = utils.to_device(cnn, _DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        cnn.parameters(),
        lr=nn_utils.Config.learning_rate,
        weight_decay=nn_utils.Config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=nn_utils.Config.num_epochs
    )

    states_path = pathlib.Path("model_states")
    current_date = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    model_name = f"Model_{postfix}_{nn_utils.Config.num_layers}l_{current_date}"
    current_states_path = states_path / "DnCNN" / model_name
    current_states_path.mkdir(parents=True, exist_ok=True)
    measurements = Measurements(
        train_psnr=[],
        train_loss=[],
        val_psnr=[],
        val_loss=[]
    )
    n_test, n_val = len(train_dataloader), len(val_dataloader)

    for epoch in range(nn_utils.Config.num_epochs):
        # Train step
        cnn.train()
        total_psnr_train = .0
        total_loss_train = .0

        with tqdm(total=len(train_dataloader) * nn_utils.Config.batch_size) as tqdm_:
            tqdm_.set_description(f"Train Epoch {epoch + 1}/{nn_utils.Config.num_epochs}")

            for noised, real in train_dataloader:
                optimizer.zero_grad()
                prediction = cnn(noised)
                loss_train = criterion(prediction, real)
                loss_train.backward()
                optimizer.step()

                numpy_predicted_batch = np.uint8(noised.cpu().numpy() * 255.)
                numpy_real_batch = np.uint8(real.cpu().numpy() * 255.)

                # Add batch PSNR/LOSS values
                total_psnr_train += metrics.peak_signal_to_noise_ratio(
                    numpy_predicted_batch,
                    numpy_real_batch
                )
                total_loss_train += loss_train.item()

                tqdm_.update(len(noised))
                tqdm_.set_postfix_str(
                    f"LOSS: {total_loss_train / n_test:.4f} | "
                    f"PSNR: {total_psnr_train / n_test: .2f}"
                )

        # Validation step
        cnn.eval()
        total_psnr_val = .0
        total_loss_val = .0

        with tqdm(total=len(val_dataloader) * nn_utils.Config.batch_size) as tqdm_:
            tqdm_.set_description(f"Val Epoch {epoch + 1}/{nn_utils.Config.num_epochs}")

            with torch.no_grad():
                for noised, real in val_dataloader:
                    prediction = cnn(noised)
                    loss_val = criterion(prediction, real)

                    numpy_predicted_batch = np.uint8(noised.cpu().numpy() * 255.)
                    numpy_real_batch = np.uint8(real.cpu().numpy() * 255.)

                    # Add batch PSNR/LOSS values
                    total_psnr_val += metrics.peak_signal_to_noise_ratio(
                        numpy_predicted_batch,
                        numpy_real_batch
                    )
                    total_loss_val += loss_val.item()

                    tqdm_.update(len(noised))
                    tqdm_.set_postfix_str(
                        f"LOSS: {total_loss_val/ n_val:.4f} | "
                        f"PSNR: {total_psnr_val / n_val: .2f}"
                    )

        scheduler.step()

        # Extract model
        if (epoch + 1) % nn_utils.Config.save_step == 0:
            state_path = current_states_path / f"{epoch}_epoch.pth"
            torch.save(cnn.state_dict(), state_path)

        # Save metrics
        total_psnr_train /= n_test
        total_loss_train /= n_test
        total_psnr_val /= n_val
        total_loss_val /= n_val

        measurements.train_loss.append(total_loss_train)
        measurements.train_psnr.append(total_psnr_train)
        measurements.val_psnr.append(total_psnr_val)
        measurements.val_loss.append(total_loss_val)

    measurements_path = current_states_path / "measurements.csv"
    measurements_table = measurements.to_pandas()
    measurements_table.to_csv(
        measurements_path,
        sep="\t",
        index=False
    )


def _test(parameters_path: pathlib.Path,
          dataset: nn_utils.DnCnnDatasetTest) -> None:
    cnn = DnCNN(
        num_layers=nn_utils.Config.num_layers,
        parameters_path=parameters_path
    )
    cnn = utils.to_device(cnn, _DEVICE)

    cnn.eval()
    n = 1400

    with torch.no_grad():
        data = dataset[n]
        noised_img_tensor, pos, cleaned_img_tensor = data
        noised_img_tensor = utils.to_device(noised_img_tensor, _DEVICE)

        denoised_img_tensor = cnn(noised_img_tensor)
        cleaned_img_numpy = dataset.to_image(cleaned_img_tensor)
        noised_img_numpy = dataset.from_patches(
            noised_img_tensor,
            pos,
            cleaned_img_tensor.shape
        )
        denoised_img_numpy = dataset.from_patches(
            denoised_img_tensor,
            pos,
            cleaned_img_tensor.shape
        )

        images = np.vstack((cleaned_img_numpy, noised_img_numpy, denoised_img_numpy))

        # noinspection PyUnresolvedReferences
        images = cv.cvtColor(images, cv.COLOR_RGB2BGR)
        # noinspection PyUnresolvedReferences
        cv.imwrite("test.png", images)


def train(tni_img_path: pathlib.Path,
          tri_img_path: pathlib.Path,
          vni_img_path: pathlib.Path,
          vri_img_path: pathlib.Path,
          postfix: str) -> None:
    print("Preparing data...")

    # Prepare train data
    train_dataset = nn_utils.DnCnnDataset(
        noised_data_path=tni_img_path,
        cleaned_data_path=tri_img_path,
    )
    train_dl = utils.ToDeviceLoader(
        torch.utils.data.DataLoader(
            nn_utils.rearrange_dataset(train_dataset),
            batch_size=nn_utils.Config.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        ),
        _DEVICE
    )

    # Prepare val data
    val_dataset = nn_utils.DnCnnDataset(
        noised_data_path=vni_img_path,
        cleaned_data_path=vri_img_path,
    )
    val_dl = utils.ToDeviceLoader(
        torch.utils.data.DataLoader(
            nn_utils.rearrange_dataset(val_dataset),
            batch_size=nn_utils.Config.batch_size,
            pin_memory=True,
            drop_last=True
        ),
        _DEVICE
    )

    # _TRAINING
    _train(
        train_dl,
        val_dl,
        postfix
    )


def test(test_path: pathlib.Path,
         real_path: pathlib.Path) -> None:
    mp = __MODEL_STATES__ / "DnCNN" / "Model_noised-add-impulse_17l_2025-04-24T112600" / "2450_epoch.pth"
    dataset = nn_utils.DnCnnDatasetTest(
        noised_data_path=test_path,
        cleaned_data_path=real_path
    )
    _test(
        mp,
        dataset
    )


if __name__ == "__main__":
    # Postfix
    px = "noised-add-impulse"

    dataset_name = f"imagenet-mini-shrink"
    noised_img_path = __SRC__ / f"{dataset_name}-{px}"
    real_img_path = __SRC__ / dataset_name

    train_noised_img_path = noised_img_path / "train"
    train_real_img_path = real_img_path / "train"
    val_noised_img_path = noised_img_path / "val"
    val_real_img_path = real_img_path / "val"

    # TRAINING
    train(
        train_noised_img_path,
        train_real_img_path,
        val_noised_img_path,
        val_real_img_path,
        px
    )

    # TESTING
    # test(
    #     noised_img_path,
    #     real_img_path
    # )
