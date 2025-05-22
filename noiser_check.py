import numpy as np
import cv2 as cv

from utils import __SRC__, utils
from utils.nn import dataset, readers


if __name__ == "__main__":
    noised_img_path = __SRC__ / "imagenet-mini-noised" / "val"
    real_img_path = __SRC__ / "imagenet-mini" / "val"
    n = 12

    dataset_ = dataset.DnCnnDatasetTest(
        noised_data_path=noised_img_path,
        cleaned_data_path=real_img_path,
        reader=readers.CVReader()
    )

    noised, clean = dataset_[n]
    noised = utils.to_device(noised, utils.get_device())

    img_clean_numpy = dataset_.to_image(clean)
    img_noised_numpy = dataset_.to_image(noised)

    images = np.vstack((img_clean_numpy, img_noised_numpy))

    # noinspection PyUnresolvedReferences
    images = cv.cvtColor(images, cv.COLOR_RGB2BGR)
    # noinspection PyUnresolvedReferences
    cv.imwrite("test.png", images)
