import pathlib

import numpy as np

from utils.noise.dataset_noising import IReader
from utils import readers


class MatReader(IReader):
    def read_image(self, path: pathlib.Path) -> np.ndarray:
        res = readers.mat_reader(path)
        img = res["placeholder"]
        return img


class CVReader(IReader):
    def read_image(self, path: pathlib.Path) -> np.ndarray:
        img = readers.cv_reader(path)
        return img
