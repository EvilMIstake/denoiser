import pathlib

import numpy as np
from PIL import Image
import cv2 as cv

import torch
from torchvision.transforms import transforms, functional
from torch.utils.data import Dataset

from utils.utils import get_all_paths


class _DatasetMixins:
    @staticmethod
    def _get_all_paths(path: pathlib.Path) -> list[pathlib.Path]:
        return get_all_paths(path)

    @staticmethod
    def _read_pil_image(path: pathlib.Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    @staticmethod
    def _read_cv_image(path: pathlib.Path) -> np.ndarray:
        # noinspection PyUnresolvedReferences
        return cv.imread(str(path))


class DnCnnDataset(Dataset, _DatasetMixins):
    def __init__(self,
                 noised_data_path: pathlib.Path,
                 cleaned_data_path: pathlib.Path,
                 resize_size: int = 256,
                 patch_size: int = 50):
        super(Dataset, self).__init__()
        self.__noised_data_paths = self._get_all_paths(noised_data_path)
        self.__cleaned_data_paths = self._get_all_paths(cleaned_data_path)

        assert len(self.__cleaned_data_paths) == len(self.__noised_data_paths), \
            "Datasets must be consistent"

        self.__patch_size = patch_size
        self.__resize_size = resize_size

    def __len__(self) -> int:
        return len(self.__noised_data_paths)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_n = self._read_pil_image(self.__noised_data_paths[item])
        img_c = self._read_pil_image(self.__cleaned_data_paths[item])

        # Apply resize transform
        resize = transforms.Resize(
            self.__resize_size,
            interpolation=transforms.InterpolationMode.BICUBIC
        )
        img_np = resize(img_n)
        img_cp = resize(img_c)

        # Random patch crop
        i, j, h, w = transforms.RandomCrop.get_params(
            img_np,
            (self.__patch_size, self.__patch_size)
        )
        img_np = functional.crop(img_np, i, j, h, w)
        img_cp = functional.crop(img_cp, i, j, h, w)

        # noinspection PyTypeChecker
        img_np_tensor = functional.to_tensor(img_np)
        # noinspection PyTypeChecker
        img_cp_tensor = functional.to_tensor(img_cp)

        return img_np_tensor, img_cp_tensor


class DnCnnDatasetTest(Dataset, _DatasetMixins):
    def __init__(self,
                 noised_data_path: pathlib.Path,
                 cleaned_data_path: pathlib.Path,
                 patch_size: int):
        super(Dataset, self).__init__()
        self.__noised_data_path = self._get_all_paths(noised_data_path)
        self.__cleaned_data_path = self._get_all_paths(cleaned_data_path)
        self.__patch_size = patch_size

    def __len__(self) -> int:
        return len(self.__noised_data_path)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, np.ndarray]:
        img_n = self._read_cv_image(self.__noised_data_path[item])
        img_c = self._read_cv_image(self.__cleaned_data_path[item])
        y, x, _ = img_n.shape

        padding_y = y % self.__patch_size
        padding_x = x % self.__patch_size

        if padding_y:
            padding_y = self.__patch_size - padding_y
        if padding_x:
            padding_x = self.__patch_size - padding_x

        # noinspection PyUnresolvedReferences
        img_n_padded = cv.copyMakeBorder(
            img_n,
            0,
            padding_y,
            0,
            padding_x,
            cv.BORDER_CONSTANT
        )

        to_tensor_transform = transforms.ToTensor()
        patches = [
            to_tensor_transform(
                img_n_padded[
                    sy:sy + self.__patch_size,
                    sx:sx + self.__patch_size,
                    :
                ]
            )
            for sy in range(0, y, self.__patch_size)
            for sx in range(0, x, self.__patch_size)
        ]

        img_np_tensor = torch.stack(patches)

        return img_np_tensor, img_c
