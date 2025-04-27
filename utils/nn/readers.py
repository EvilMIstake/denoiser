import pathlib

import numpy as np
import cv2 as cv

from utils.nn import dataset
from utils import readers


class CVReader(dataset.IReader):
    def read_image(self, path: pathlib.Path) -> np.ndarray:
        img = readers.cv_reader(path)
        # noinspection PyUnresolvedReferences
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return img
