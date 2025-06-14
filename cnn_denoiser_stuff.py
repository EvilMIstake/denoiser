import datetime
import pathlib
from dataclasses import dataclass

import cv2 as cv
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm

from utils import (
    utils,
    metrics,
    __SRC__
)
from utils.nn import (
    dataset,
    config,
    readers
)
from utils.nn.dncnn import DnCNN

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
           model_path: pathlib.Path | None = None) -> None:
    def get_loss(predicted: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        loss = criterion(predicted, prior)
        loss.div_(2.)
        return loss

    cnn = DnCNN(
        num_layers=config.Config.num_layers,
        parameters_path=model_path,
        residual=config.Config.residual,
    )
    cnn = utils.to_device(cnn, _DEVICE)

    criterion = nn.MSELoss(reduction="sum")
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
    model_name = f"Model_{postfix}_{config.Config.num_layers}l_{current_date}"
    current_states_path = states_path / "DnCNN" / model_name
    current_states_path.mkdir(parents=True, exist_ok=True)
    measurements = Measurements(
        train_psnr=[],
        train_loss=[],
        val_psnr=[],
        val_loss=[]
    )
    measurements_path = current_states_path / "measurements.csv"

    n_test, n_val = len(train_dataloader), len(val_dataloader)

    for epoch in range(config.Config.num_epochs):
        # Train step
        cnn.train()
        total_psnr_train = .0
        total_loss_train = .0

        with tqdm(total=len(train_dataloader) * config.Config.batch_size) as tqdm_:
            tqdm_.set_description(f"Train Epoch {epoch + 1}/{config.Config.num_epochs}")

            for noised, real in train_dataloader:
                optimizer.zero_grad()
                prediction = cnn(noised)
                loss_train = get_loss(prediction, real)
                loss_train.backward()
                optimizer.step()

                # Evaluate results
                numpy_predicted_batch = np.uint8(prediction.detach().cpu().numpy() * 255.)
                numpy_real_batch = np.uint8(real.cpu().numpy() * 255.)

                # Add batch PSNR/LOSS values
                total_psnr_train += metrics.peak_signal_to_noise_ratio(
                    numpy_predicted_batch,
                    numpy_real_batch
                )
                # Add mean batch loss
                total_loss_train += loss_train.item() / len(noised)

                tqdm_.update(len(noised))
                tqdm_.set_postfix_str(
                    f"LOSS: {total_loss_train / n_test:.4f} | "
                    f"PSNR: {total_psnr_train / n_test: .2f}"
                )

        # Validation step
        cnn.eval()
        total_psnr_val = .0
        total_loss_val = .0

        with tqdm(total=len(val_dataloader) * config.Config.batch_size) as tqdm_:
            tqdm_.set_description(f"Val Epoch {epoch + 1}/{config.Config.num_epochs}")

            with torch.no_grad():
                img_noised = []
                img_denoised = []
                img_real = []

                for noised, real in val_dataloader:
                    prediction = cnn(noised)
                    loss_val = get_loss(prediction, real)

                    numpy_noised_batch = np.uint8(noised.cpu().numpy() * 255.)
                    numpy_predicted_batch = np.uint8(prediction.detach().cpu().numpy() * 255.)
                    numpy_real_batch = np.uint8(real.cpu().numpy() * 255.)

                    # Add batch PSNR/LOSS values
                    total_psnr_val += metrics.peak_signal_to_noise_ratio(
                        numpy_predicted_batch,
                        numpy_real_batch
                    )
                    # Add mean batch loss
                    total_loss_val += loss_val.item() / len(noised)

                    # Save validation results
                    img_real.extend(
                        (
                            numpy_real_batch[i].transpose(1, 2, 0)
                            for i in range(0, len(noised), config.Config.validation_img_extraction_step)
                        )
                    )
                    img_denoised.extend(
                        (
                            numpy_predicted_batch[i].transpose(1, 2, 0)
                            for i in range(0, len(noised), config.Config.validation_img_extraction_step)
                        )
                    )
                    img_noised.extend(
                        (
                            numpy_noised_batch[i].transpose(1, 2, 0)
                            for i in range(0, len(noised), config.Config.validation_img_extraction_step)
                        )
                    )

                    tqdm_.update(len(noised))
                    tqdm_.set_postfix_str(
                        f"LOSS: {total_loss_val / n_val:.4f} | "
                        f"PSNR: {total_psnr_val / n_val: .2f}"
                    )

            img_real_stack = np.vstack(img_real)
            img_noised_stack = np.vstack(img_noised)
            img_denoised_stack = np.vstack(img_denoised)
            img_noise_stack = img_noised_stack - img_denoised_stack
            img = np.hstack((img_real_stack, img_noised_stack, img_denoised_stack, img_noise_stack))

            # noinspection PyUnresolvedReferences
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            # noinspection PyUnresolvedReferences
            cv.imwrite("test.png", img)

        scheduler.step()

        # Save metrics
        total_psnr_train /= n_test
        total_loss_train /= n_test
        total_psnr_val /= n_val
        total_loss_val /= n_val

        # Extract model
        if total_psnr_val > config.Config.save_threshold:
            state_path = current_states_path / f"{epoch}_epoch.pth"
            torch.save(cnn.state_dict(), state_path)

        measurements.train_loss.append(total_loss_train)
        measurements.train_psnr.append(total_psnr_train)
        measurements.val_psnr.append(total_psnr_val)
        measurements.val_loss.append(total_loss_val)

        measurements_table = measurements.to_pandas()
        measurements_table.to_csv(
            measurements_path,
            sep="\t",
            index=False
        )


def _test(model_path: pathlib.Path,
          dataset_: dataset.DnCnnDatasetTest) -> None:
    cnn = DnCNN(
        num_layers=config.Config.num_layers,
        parameters_path=model_path,
        residual=config.Config.residual
    )
    cnn = utils.to_device(cnn, _DEVICE)

    cnn.eval()
    n = 10

    with torch.no_grad():
        data = dataset_[n]
        noised_img_tensor, cleaned_img_tensor = data
        noised_img_tensor = utils.to_device(noised_img_tensor, _DEVICE)
        t = noised_img_tensor.view(1, -1, *noised_img_tensor.shape[1:3])

        denoised_img_tensor = cnn(t)
        cleaned_img_numpy = dataset_.to_image(cleaned_img_tensor)
        noised_img_numpy = dataset_.to_image(noised_img_tensor)
        denoised_img_numpy = dataset_.to_image(denoised_img_tensor, clip=True)
        images = np.vstack((cleaned_img_numpy, noised_img_numpy, denoised_img_numpy))

        psnr_d = metrics.peak_signal_to_noise_ratio(cleaned_img_numpy, denoised_img_numpy)
        ssim_d = metrics.structure_similarity(cleaned_img_numpy, denoised_img_numpy)

        psnr_n = metrics.peak_signal_to_noise_ratio(cleaned_img_numpy, noised_img_numpy)
        ssim_n = metrics.structure_similarity(cleaned_img_numpy, noised_img_numpy)

        print(f"{psnr_d=:.2f}")
        print(f"{ssim_d=:.2f}")
        print(f"{psnr_n=:.2f}")
        print(f"{ssim_n=:.2f}")

        # noinspection PyUnresolvedReferences
        images = cv.cvtColor(images, cv.COLOR_RGB2BGR)
        # noinspection PyUnresolvedReferences
        cv.imwrite("test.png", images)


def train(noised_image_path: pathlib.Path,
          real_image_path: pathlib.Path,
          postfix: str,
          model_path: pathlib.Path | None) -> None:
    train_noised_img_path = noised_image_path / "train"
    train_real_img_path = real_image_path / "train"
    val_noised_img_path = noised_image_path / "val"
    val_real_img_path = real_image_path / "val"

    # Prepare train data
    train_dataset = dataset.DnCnnDataset(
        noised_data_path=train_noised_img_path,
        cleaned_data_path=train_real_img_path,
        reader=readers.CVReader()
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
    val_dataset = dataset.DnCnnDataset(
        noised_data_path=val_noised_img_path,
        cleaned_data_path=val_real_img_path,
        reader=readers.CVReader()
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
        postfix,
        model_path
    )


def test(test_path: pathlib.Path,
         real_path: pathlib.Path,
         model_path: pathlib.Path) -> None:
    dataset_ = dataset.DnCnnDatasetTest(
        noised_data_path=test_path,
        cleaned_data_path=real_path,
        reader=readers.CVReader()
    )
    _test(
        model_path,
        dataset_
    )


if __name__ == "__main__":
    # Postfix
    px = "blur"
    dataset_name = f"BSDS500"

    # Training
    noised_img_path = __SRC__ / f"{dataset_name}-{px}-pfr"
    real_img_path = __SRC__ / f"{dataset_name}-pfr"
    # parameters_path = __MODEL_STATES__ / "DnCNN" / "Model_poisson_20l_2025-05-09T200955" / "46_epoch.pth"
    parameters_path = None

    train(
        noised_img_path,
        real_img_path,
        px,
        parameters_path
    )

    # TESTING
    # noised_img_path = __SRC__ / f"{dataset_name}-{px}".rstrip("-") / "test"
    # real_img_path = __SRC__ / dataset_name / "test"
    # # parameters_path = __MODEL_STATES__ / "DnCNN" / "Model_periodic_20l_2025-05-06T114529" / "46_epoch.pth"
    # parameters_path = __MODEL_STATES__ / "DnCNN" / "Model_add_20l_2025-05-02T183920" / "34_epoch.pth"
    #
    # test(
    #     noised_img_path,
    #     real_img_path,
    #     parameters_path
    # )
