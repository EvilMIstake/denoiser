from abc import ABC, abstractmethod
from random import random, randint

import numpy as np

from utils.noise.noise import noise


class IRandomNoiser(ABC):
    @abstractmethod
    def noised_image(self) -> np.ndarray:
        ...


class GaussianNoiser(IRandomNoiser):
    def __init__(self, image: np.ndarray):
        self.__image = image
        self.__noise_level = random() * 50.

    def noised_image(self) -> np.ndarray:
        noised_img = noise.add_gaussian_noise(
            self.__image,
            self.__noise_level
        )
        return noised_img


class UniformNoiser(IRandomNoiser):
    def __init__(self, image: np.ndarray):
        self.__image = image
        self.__noise_level = random() * 50.

    def noised_image(self) -> np.ndarray:
        noised_img = noise.add_uniform_noise(
            self.__image,
            self.__noise_level
        )
        return noised_img


class SaltNPaperNoiser(IRandomNoiser):
    def __init__(self, image: np.ndarray):
        self.__image = image
        self.__prob = random() * 0.1 + 0.15

    def noised_image(self) -> np.ndarray:
        noised_img = noise.add_salt_pepper_noise(
            self.__image,
            self.__prob
        )
        return noised_img


class MotionBlurNoiser(IRandomNoiser):
    def __init__(self, image: np.ndarray):
        self.__image = image
        self.__size = (randint(3, 15) << 1) + 1
        self.__vertical = bool(randint(0, 1))

    def noised_image(self) -> np.ndarray:
        noised_img = noise.add_motion_blur(
            self.__image,
            self.__size,
            self.__vertical
        )
        return noised_img


class DeFocusBlurNoiser(IRandomNoiser):
    def __init__(self, image: np.ndarray):
        self.__image = image
        self.__size = (randint(3, 15) << 1) + 1

    def noised_image(self) -> np.ndarray:
        noised_img = noise.add_defocus_blur(
            self.__image,
            self.__size
        )
        return noised_img


class PeriodicNoiser(IRandomNoiser):
    def __init__(self, image: np.ndarray):
        self.__image = image
        self.__amplitude = randint(25, 70)
        self.__period = randint(5, 15)
        self.__offset = randint(5, 10)
        self.__vertical = bool(randint(0, 1))

    def noised_image(self) -> np.ndarray:
        noised_img = noise.add_periodic_noise(
            self.__image,
            self.__amplitude,
            self.__period,
            self.__offset,
            self.__vertical
        )
        return noised_img


class PlaceHolder(IRandomNoiser):
    def __init__(self, image: np.ndarray):
        self.__image = image

    def noised_image(self) -> np.ndarray:
        return self.__image


def get_noiser(image: np.ndarray, idx: int) -> IRandomNoiser:
    match idx:
        case 0:
            return GaussianNoiser(image)
        case 1:
            return UniformNoiser(image)
        case 2:
            return SaltNPaperNoiser(image)
        case 3:
            return MotionBlurNoiser(image)
        case 4:
            return DeFocusBlurNoiser(image)
        case 5:
            return PeriodicNoiser(image)
        case _:
            return PlaceHolder(image)
