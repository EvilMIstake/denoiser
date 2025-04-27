import pathlib

import numpy as np
import scipy as sc
import cv2 as cv
from PIL import Image


def mat_reader(path: pathlib.Path) -> dict:
    """Read Matlab images data"""
    res = sc.io.loadmat(str(path))
    return res


def cv_reader(path: pathlib.Path) -> np.ndarray:
    """Read CV images in BGR format"""
    # noinspection PyUnresolvedReferences
    res = cv.imread(str(path), cv.IMREAD_COLOR)
    return res


def pil_reader(path: pathlib.Path) -> Image.Image:
    """Read PIL images in RGB format"""
    res = Image.open(path).convert("RGB")
    return res
