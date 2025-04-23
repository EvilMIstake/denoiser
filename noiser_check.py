import numpy as np
import cv2 as cv

from utils import __SRC__
from utils.utils import get_device
from utils.nn import utils as nn_utils


if __name__ == "__main__":
    noised_img_path = __SRC__ / "imagenet-mini-shrink-noised" / "train"
    real_img_path = __SRC__ / "imagenet-mini-shrink" / "train"

    dataset = nn_utils.DnCnnDatasetTest(
        noised_data_path=noised_img_path,
        cleaned_data_path=real_img_path,
    )

    noised, pos, clean = dataset[170]
    for p in noised:
        p.to(get_device())

    img_clean = dataset.to_image(clean)
    noised_full = dataset.from_patches(noised, pos, clean.shape)

    images = np.vstack((img_clean, noised_full))

    # noinspection PyUnresolvedReferences
    images = cv.cvtColor(images, cv.COLOR_RGB2BGR)
    # noinspection PyUnresolvedReferences
    cv.imwrite("temp.png", images)
