import pathlib

import numpy as np

from utils import readers
from utils.noise.dataset_noising import IReader


class MatReader(IReader):
    def read_image(self, path: pathlib.Path) -> np.ndarray:
        res = readers.mat_reader(path)
        img = res["placeholder"]
        return img


class CVReader(IReader):
    def read_image(self, path: pathlib.Path) -> np.ndarray:
        img = readers.cv_reader(path)
        return img
