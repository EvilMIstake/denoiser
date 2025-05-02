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
        self.__noise_level = 50

    def noised_image(self) -> np.ndarray:
        noise_level = random() * self.__noise_level
        noised_img = noise.add_gaussian_noise(
            self.__image,
            noise_level
        )
        return noised_img


class UniformNoiser(IRandomNoiser):
    def __init__(self, image: np.ndarray):
        self.__image = image
        self.__noise_level = 50

    def noised_image(self) -> np.ndarray:
        noise_level = random() * self.__noise_level
        noised_img = noise.add_uniform_noise(
            self.__image,
            noise_level
        )
        return noised_img


class SaltNPaperNoiser(IRandomNoiser):
    def __init__(self, image: np.ndarray):
        self.__image = image
        self.__prob_a, self.__prob_b = 0.1, 0.15

    def noised_image(self) -> np.ndarray:
        prob = random() * self.__prob_a + self.__prob_b
        noised_img = noise.add_salt_pepper_noise(
            self.__image,
            prob
        )
        return noised_img


class MotionBlurNoiser(IRandomNoiser):
    def __init__(self, image: np.ndarray):
        self.__image = image
        self.__h_min, self.__h_max = 3, 15

    def noised_image(self) -> np.ndarray:
        size = (randint(self.__h_min, self.__h_max) << 1) + 1
        vertical = bool(randint(0, 1))

        noised_img = noise.add_motion_blur(
            self.__image,
            size,
            vertical
        )
        return noised_img


class DeFocusBlurNoiser(IRandomNoiser):
    def __init__(self, image: np.ndarray):
        self.__image = image
        self.__h_min, self.__h_max = 3, 15

    def noised_image(self) -> np.ndarray:
        size = (randint(self.__h_min, self.__h_max) << 1) + 1

        noised_img = noise.add_defocus_blur(
            self.__image,
            size
        )
        return noised_img


class PeriodicNoiser(IRandomNoiser):
    def __init__(self, image: np.ndarray):
        self.__image = image
        self.__a_min, self.__a_max = 25, 70
        self.__p_min, self.__p_max = 5, 15
        self.__o_min, self.__o_max = 5, 10

    def noised_image(self) -> np.ndarray:
        amplitude = randint(self.__a_min, self.__a_max)
        period = randint(self.__p_min, self.__p_max)
        offset = randint(self.__o_min, self.__o_max)
        vertical = bool(randint(0, 1))

        noised_img = noise.add_periodic_noise(
            self.__image,
            amplitude,
            period,
            offset,
            vertical
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
