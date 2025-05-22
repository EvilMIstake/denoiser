import pathlib

import cv2 as cv
import numpy as np

from utils import readers
from utils.nn import dataset


class CVReader(dataset.IReader):
    def read_image(self, path: pathlib.Path) -> np.ndarray:
        img = readers.cv_reader(path)
        # noinspection PyUnresolvedReferences
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return img
