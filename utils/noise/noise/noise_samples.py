import math

import numpy as np


def gaussian_noise(shape: tuple[int, int, ...], noise_level: float) -> np.ndarray[np.uint8]:
    gauss = (np.random.normal(size=shape) * noise_level).astype("uint8")
    return gauss


def uniform_noise(shape: tuple[int, int, ...], noise_level: float) -> np.ndarray[np.uint8]:
    uniform = (np.random.uniform(-1, 1, size=shape) * noise_level).astype("uint8")
    return uniform


def salt_paper_sample(shape: tuple[int, int, ...], prob: float = 0.05) -> tuple[np.ndarray, ...]:
    y, x, *_ = shape
    noise_samples_count = int(math.ceil(prob * x * y))

    # Uniform distribution
    samples_cords = tuple(
        np.random.randint(0, i - 1, noise_samples_count)
        for i in shape[:2]
    )

    return samples_cords


def motion_blur_kernel(size: int, vertical: bool = True) -> np.ndarray:
    assert size & 1, "Kernel size must be odd"

    kernel = np.zeros((size, size))
    kernel[:, size >> 1] = np.ones(size)

    if vertical:
        kernel = kernel.T

    # Kernel normalisation
    kernel /= size

    return kernel


def defocus_blur_kernel(size: int) -> np.ndarray:
    """Not realistic defocus blur disk-kernel"""
    assert size & 1, "Kernel size must be odd"

    cxy = size >> 1
    kernel = np.zeros((size, size))
    x, y = map(
        lambda grid: grid - cxy,
        np.ogrid[:size, :size]
    )
    circle_x, circle_y = np.where(x * x + y * y <= cxy * cxy)
    kernel[circle_x, circle_y] = 1

    # Kernel normalisation
    kernel /= np.count_nonzero(kernel)

    return kernel


def periodic_noise(shape: tuple[int, int, ...],
                   amplitude: int,
                   period: int,
                   offset: int,
                   vertical: bool = True) -> np.ndarray:
    sx, sy, c, *_ = shape
    dir_x, dir_y = -1, 1
    reps_x, reps_y = 1, sy

    if vertical:
        sx, sy = sy, sx
        dir_x, dir_y = dir_y, dir_x
        reps_x, reps_y = sy, 1

    x = np.arange(0, sx).reshape(dir_x, dir_y)
    cos_vector = np.uint8(amplitude * (1. + np.cos(2. * np.pi * x / period)) * 0.5 + offset)
    p_noise = np.tile(cos_vector, (reps_x, reps_y))
    noise = np.repeat(np.expand_dims(p_noise, axis=2), c, axis=2)

    return noise
