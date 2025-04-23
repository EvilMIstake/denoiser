import numpy as np
import cv2 as cv

from utils import __SRC__
from utils.nn import utils as nn_utils


if __name__ == "__main__":
    noised_img_path = __SRC__ / "imagenet-mini-shrink-noised" / "train"
    real_img_path = __SRC__ / "imagenet-mini-shrink" / "train"

    dataset = nn_utils.DnCnnDataset(
        noised_data_path=noised_img_path,
        cleaned_data_path=real_img_path,
        resize_size=100,
        patch_size=50
    )

    noised, clean = dataset[100]

    noised_numpy = np.uint8(noised.detach().numpy() * 255.)
    clean_numpy = np.uint8(clean.detach().numpy() * 255.)

    res = np.hstack((noised_numpy, clean_numpy)).transpose((1, 2, 0))
    cv.imwrite("noise_example.png", res)
