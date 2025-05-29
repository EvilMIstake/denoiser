import cv2 as cv
import numpy as np

from utils.noise.noise import noise_samples


def add_gaussian_noise(image: np.ndarray,
                       noise_level: float) -> np.ndarray[np.uint8]:
    gauss = noise_samples.gaussian_noise(image.shape, noise_level)
    noisy_image = np.clip(image + gauss, 0., 255.).astype(np.uint8)
    return noisy_image


def add_uniform_noise(image: np.ndarray,
                      noise_level: float) -> np.ndarray[np.uint8]:
    uniform = noise_samples.uniform_noise(image.shape, noise_level)
    noisy_image = np.clip(image + uniform, 0., 255.).astype(np.uint8)
    return noisy_image


def _add_salt_paper_noise(image: np.ndarray,
                          value: int,
                          prob: float = 0.05) -> np.ndarray[np.uint8]:
    noisy_image = image.copy()
    sample_y, sample_x = noise_samples.salt_paper_sample(image.shape, prob)
    noisy_image[sample_y, sample_x, :] = value
    return noisy_image


def add_salt_noise(image: np.ndarray,
                   prob: float = 0.05) -> np.ndarray[np.uint8]:
    salty_image = _add_salt_paper_noise(image, 255, prob)
    return salty_image


def add_pepper_noise(image: np.ndarray,
                     prob: float = 0.05) -> np.ndarray[np.uint8]:
    peppered_image = _add_salt_paper_noise(image, 0, prob)
    return peppered_image


def add_salt_pepper_noise(image: np.ndarray,
                          prob: float = 0.05) -> np.ndarray[np.uint8]:
    prob *= 0.5
    salty_image = add_salt_noise(image, prob)
    peppered_salty_image = add_pepper_noise(salty_image, prob)
    return peppered_salty_image


def add_motion_blur(image: np.ndarray,
                    size: int,
                    vertical: bool = True) -> np.ndarray[np.uint8]:
    motion_kernel = noise_samples.motion_blur_kernel(size, vertical)

    # noinspection PyUnresolvedReferences
    blured_image = cv.filter2D(image, -1, motion_kernel)
    return blured_image


def add_defocus_blur(image: np.ndarray,
                     size: int) -> np.ndarray:
    disk_kernel = noise_samples.defocus_blur_kernel(size)

    # noinspection PyUnresolvedReferences
    blured_image = cv.filter2D(image, -1, disk_kernel)
    return blured_image


def add_periodic_noise(image: np.ndarray,
                       amplitude: int,
                       period: int,
                       offset: int,
                       vertical: bool = True) -> np.ndarray:
    periodic_noise = noise_samples.periodic_noise(
        image.shape,
        amplitude,
        period,
        offset,
        vertical
    )

    # noinspection PyUnresolvedReferences
    blured_image = cv.add(image, periodic_noise)
    return blured_image


def add_poisson_noise(image: np.ndarray, peak: int) -> np.ndarray:
    noisy_image = np.uint8(np.random.poisson(image / 255.0 * peak) / peak * 255)
    return noisy_image
