import numpy as np
import cv2 as cv

from utils import __SRC__
from utils.nn import utils as nn_utils
from utils import utils


if __name__ == "__main__":
    noised_img_path = __SRC__ / "imagenet-mini-shrink-noised" / "val"
    real_img_path = __SRC__ / "imagenet-mini-shrink" / "val"
    n = 170

    dataset = nn_utils.DnCnnDatasetTest(
        noised_data_path=noised_img_path,
        cleaned_data_path=real_img_path,
    )

    noised, pos, clean = dataset[n]
    noised = utils.to_device(noised, utils.get_device())

    img_clean = dataset.to_image(clean)
    noised_full = dataset.from_patches(noised, pos, clean.shape)

    images = np.vstack((img_clean, noised_full))

    # noinspection PyUnresolvedReferences
    images = cv.cvtColor(images, cv.COLOR_RGB2BGR)
    # noinspection PyUnresolvedReferences
    cv.imwrite("test.png", images)
