import numpy as np
import cv2 as cv


def _check_shapes(img_a: np.ndarray, img_b: np.ndarray) -> None:
    assert img_a.shape == img_b.shape, "Img shapes must be consistent"


def mse(img_prior: np.ndarray, img_predicted: np.ndarray) -> np.float32:
    """MSE calculation"""

    _check_shapes(img_prior, img_predicted)
    img_prior = img_prior.astype(np.float32)
    img_predicted = img_predicted.astype(np.float32)

    diff = img_prior - img_predicted
    diff_sq = diff * diff
    metric = np.sum(diff_sq) / img_prior.size

    return metric


def rmsd(img_prior: np.ndarray, img_predicted: np.ndarray) -> np.float32:
    """RMSD calculation"""

    metric = np.sqrt(mse(img_prior, img_predicted))

    return metric


def peak_signal_to_noise_ratio(img_prior: np.ndarray, img_predicted: np.ndarray) -> np.float32:
    """PSNR calculation

    Пиковое отношение сигнал/шум, обозначающее отношение
    между максимально возможной мощностью искажающего сигнала
    и мощностью искажающего шума. Поскольку многие сигналы
    имеют очень широкий динамический диапазон, PSNR обычно
    выражается как логарифмическая величина (с использованием
    шкалы децибел)
    """

    max_i = 255.
    mse_ = mse(img_prior, img_predicted)
    metric = 20. * np.log10(max_i) - 10. * np.log10(mse_)

    return metric


def structure_similarity(img_prior: np.ndarray, img_predicted: np.ndarray) -> np.float32:
    """SSIM calculation

    Метрика схожести двух изображений. Разницей, в сравнении
    с другими метриками (PSNR, MSE), заключается в том, что
    они оценивают абсолютные ошибки. SSIM-индекс, в свою
    очередь, является моделью, рассматривающей ухудшение
    изображения как воспринимаемое изменение структурной
    информации. Структурная информация – идея о том, что
    пиксели имеют сильные взаимозависимости, особенно
    когда они пространственно близки. Эти зависимости
    содержат важную информацию о структуре объектов в
    визуальной сцене. Иными словами, данная метрика
    более точно отражает человеческое восприятие изображений
    """

    _check_shapes(img_prior, img_predicted)
    is_grayscale = img_prior.ndim == 2
    is_colorized = img_prior.ndim == 3 and img_prior.shape[2] == 3

    assert is_grayscale or is_colorized

    if is_colorized:
        # noinspection PyUnresolvedReferences
        img_prior = cv.cvtColor(img_prior, cv.COLOR_BGR2GRAY)
        # noinspection PyUnresolvedReferences
        img_predicted = cv.cvtColor(img_predicted, cv.COLOR_BGR2GRAY)

    img_prior = img_prior.astype(np.float32)
    img_predicted = img_predicted.astype(np.float32)
    # noinspection PyUnresolvedReferences
    kernel = cv.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    # noinspection PyUnresolvedReferences
    mu_pri = cv.filter2D(img_prior, -1, window)
    # noinspection PyUnresolvedReferences
    mu_pre = cv.filter2D(img_predicted, -1, window)
    mu_pri_sq = mu_pri * mu_pri
    mu_pre_sq = mu_pre * mu_pre
    mu_pri_mu_pre = mu_pri * mu_pre

    # noinspection PyUnresolvedReferences
    sigma_pri_sq = cv.filter2D(img_prior * img_prior, -1, window) - mu_pri_sq
    # noinspection PyUnresolvedReferences
    sigma_pre_sq = cv.filter2D(img_predicted * img_predicted, -1, window) - mu_pre_sq
    # noinspection PyUnresolvedReferences
    sigma_pri_pre = cv.filter2D(img_predicted * img_prior, -1, window) - mu_pri_mu_pre

    c1 = (0.01 * 255.) ** 2
    c2 = (0.03 * 255.) ** 2

    ssim_map = (2. * mu_pri_mu_pre + c1) * (2. * sigma_pri_pre + c2) / \
               ((mu_pre_sq + mu_pri_sq + c1) * (sigma_pri_sq + sigma_pre_sq + c2))
    metric = np.mean(ssim_map)

    # noinspection PyTypeChecker
    return metric
