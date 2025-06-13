from abc import ABC, abstractmethod
from random import random, randint

import numpy as np

from utils.noise.noise import noise


class IRandomNoiser(ABC):
    @abstractmethod
    def noised_image(self) -> np.ndarray:
        ...


class GaussianNoiser(IRandomNoiser):
    def __init__(self,
                 image: np.ndarray,
                 max_level: int = 10,
                 min_level: int = 50):
        self.__image = image
        self.__noise_level = max_level
        self.__start = min_level
        self.__range = max_level - min_level

    def noised_image(self) -> np.ndarray:
        noise_level = random() * self.__range + self.__start
        noised_img = noise.add_gaussian_noise(
            self.__image,
            noise_level
        )
        return noised_img


class UniformNoiser(IRandomNoiser):
    def __init__(self,
                 image: np.ndarray,
                 max_level: int = 10,
                 min_level: int = 50):
        self.__image = image
        self.__noise_level = max_level
        self.__start = min_level
        self.__range = max_level - min_level

    def noised_image(self) -> np.ndarray:
        noise_level = random() * self.__range + self.__start
        noised_img = noise.add_uniform_noise(
            self.__image,
            noise_level
        )
        return noised_img


class SaltNPaperNoiser(IRandomNoiser):
    def __init__(self,
                 image: np.ndarray,
                 prob_a: int = 0.1,
                 prob_b: int = 0.15):
        self.__image = image
        self.__prob_a, self.__prob_b = prob_a, prob_b

    def noised_image(self) -> np.ndarray:
        prob = random() * self.__prob_a + self.__prob_b
        noised_img = noise.add_salt_pepper_noise(
            self.__image,
            prob
        )
        return noised_img


class MotionBlurNoiser(IRandomNoiser):
    def __init__(self,
                 image: np.ndarray,
                 h_min: int = 3,
                 h_max: int = 7):
        self.__image = image
        self.__h_min, self.__h_max = h_min, h_max

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
    def __init__(self,
                 image: np.ndarray,
                 h_min: int = 3,
                 h_max: int = 7):
        self.__image = image
        self.__h_min, self.__h_max = h_min, h_max

    def noised_image(self) -> np.ndarray:
        size = (randint(self.__h_min, self.__h_max) << 1) + 1

        noised_img = noise.add_defocus_blur(
            self.__image,
            size
        )
        return noised_img


class PeriodicNoiser(IRandomNoiser):
    def __init__(self,
                 image: np.ndarray,
                 a_min: int = 25,
                 a_max: int = 70,
                 p_min: int = 5,
                 p_max: int = 15,
                 o_min: int = 5,
                 o_max: int = 10):
        self.__image = image
        self.__a_min, self.__a_max = a_min, a_max
        self.__p_min, self.__p_max = p_min, p_max
        self.__o_min, self.__o_max = o_min, o_max

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


class PoissonNoiser(IRandomNoiser):
    def __init__(self,
                 image: np.ndarray,
                 peak_min: int = 20,
                 peak_max: int = 60):
        self.__image = image
        self.__peak_min, self.__peak_max = peak_min, peak_max

    def noised_image(self) -> np.ndarray:
        peak = randint(self.__peak_min, self.__peak_max)
        noised_img = noise.add_poisson_noise(self.__image, peak)
        return noised_img


class PlaceHolder(IRandomNoiser):
    def __init__(self, image: np.ndarray):
        self.__image = image

    def noised_image(self) -> np.ndarray:
        return self.__image


def get_noiser(image: np.ndarray, idx: int, *args, **kwargs) -> IRandomNoiser:
    match idx:
        case 0:
            return GaussianNoiser(image, *args, **kwargs)
        case 1:
            return UniformNoiser(image, *args, **kwargs)
        case 2:
            return SaltNPaperNoiser(image, *args, **kwargs)
        case 3:
            return MotionBlurNoiser(image, *args, **kwargs)
        case 4:
            return DeFocusBlurNoiser(image, *args, **kwargs)
        case 5:
            return PeriodicNoiser(image, *args, **kwargs)
        case 6:
            return PoissonNoiser(image, *args, **kwargs)
        case _:
            return PlaceHolder(image)
