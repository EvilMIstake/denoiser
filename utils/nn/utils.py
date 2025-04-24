import pathlib

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset

from utils.utils import get_all_paths, get_device


class _ConfigMeta(type):
    def __str__(cls) -> str:
        return f"{cls.__name__}({cls.__str__()})"


class Config(metaclass=_ConfigMeta):
    r = 25
    patch_size = (r << 1) + 1
    image_width = patch_size * 5
    stride = (3 * patch_size) >> 2

    batch_size: int = 256
    num_layers: int = 17
    num_epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.
    save_step: int = 5

    @classmethod
    def __str__(cls) -> str:
        return f"\n\t" \
               f"{cls.patch_size=}, \n\t" \
               f"{cls.image_width=}, \n\t" \
               f"{cls.stride=}, \n\t" \
               f"{cls.batch_size=}, \n\t" \
               f"{cls.num_layers=}, \n\t" \
               f"{cls.num_epochs=}, \n\t" \
               f"{cls.learning_rate=}, \n\t" \
               f"{cls.weight_decay=}\n"


class _DatasetMixins:
    @staticmethod
    def _get_all_paths(path: pathlib.Path) -> list[pathlib.Path]:
        return get_all_paths(path)

    @staticmethod
    def _read_image(path: pathlib.Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    @staticmethod
    def to_image(image: torch.Tensor) -> np.ndarray:
        img = np.uint8(image.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255.)
        return img

    @staticmethod
    def from_patches(patches: torch.Tensor,
                     positions: list[tuple[int, int]],
                     shape: tuple[int, int, int, int]) -> np.ndarray:
        i, j = positions[-1]
        h, w = i + Config.patch_size, j + Config.patch_size

        device = get_device()
        image = torch.zeros((1, 3, h, w), device=device)
        patch_count_mask = torch.zeros((1, 1, h, w), device=device)

        for (i, j), p in zip(positions, patches):
            image[:, :, i:i + Config.patch_size, j:j + Config.patch_size] += p
            patch_count_mask[:, :, i:i + Config.patch_size, j:j + Config.patch_size] += 1

        # Preventing division by zero
        patch_count_mask = torch.where(
            patch_count_mask == 0,
            torch.ones_like(patch_count_mask),
            patch_count_mask
        )
        image /= patch_count_mask

        _, _, h, w = shape
        image_numpy = _DatasetMixins.to_image(image)[:h, :w, :]

        return image_numpy


class _DummyDataset(Dataset):
    def __init__(self, data: list[tuple[torch.Tensor, torch.Tensor]]):
        super(Dataset, self).__init__()
        self.__data = data

    def __len__(self) -> int:
        return len(self.__data)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.__data[item]


class DnCnnDataset(Dataset, _DatasetMixins):
    def __init__(self,
                 noised_data_path: pathlib.Path,
                 cleaned_data_path: pathlib.Path):
        super(Dataset, self).__init__()
        self.__noised_data_paths = self._get_all_paths(noised_data_path)
        self.__cleaned_data_paths = self._get_all_paths(cleaned_data_path)

        assert len(self.__cleaned_data_paths) == len(self.__noised_data_paths), \
            "Datasets must be consistent"

        self.__transforms = transforms.Compose(
            [
                transforms.Resize((Config.image_width, Config.image_width)),
                transforms.ToTensor(),
            ]
        )

    @staticmethod
    def _get_patches(image: torch.Tensor) -> list[torch.Tensor]:
        _, h, w = image.shape

        # Cutting overlapping patches without using edges
        patches = [
            image[
                :,
                i:i + Config.patch_size,
                j:j + Config.patch_size
            ]
            for i in range(0, h - Config.patch_size + 1, Config.stride)
            for j in range(0, w - Config.patch_size + 1, Config.stride)
        ]

        return patches

    def __len__(self) -> int:
        return len(self.__noised_data_paths)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Read images
        img_n_raw = self._read_image(self.__noised_data_paths[item])
        img_c_raw = self._read_image(self.__cleaned_data_paths[item])

        # Apply transforms
        img_n = self.__transforms(img_n_raw)
        img_c = self.__transforms(img_c_raw)

        # Extract patches
        img_n_patches_list = self._get_patches(img_n)
        img_c_patches_list = self._get_patches(img_c)

        # Patches concat
        img_n_patches_tensor = torch.stack(img_n_patches_list)
        img_c_patches_tensor = torch.stack(img_c_patches_list)

        return img_n_patches_tensor, img_c_patches_tensor


class DnCnnDatasetTest(Dataset, _DatasetMixins):
    def __init__(self,
                 noised_data_path: pathlib.Path,
                 cleaned_data_path: pathlib.Path):
        super(Dataset, self).__init__()
        self.__noised_data_paths = self._get_all_paths(noised_data_path)
        self.__cleaned_data_paths = self._get_all_paths(cleaned_data_path)

        assert len(self.__cleaned_data_paths) == len(self.__noised_data_paths), \
            "Datasets must be consistent"

        self.__transform = transforms.ToTensor()

    @staticmethod
    def _get_patches(image: torch.Tensor) -> tuple[list[torch.Tensor], list[tuple[int, int]]]:
        c, h, w = image.shape

        # Cutting overlapping patches without using edges (right/bottom)
        positions: list[tuple[int, int]] = [
            (i, j)
            for i in range(0, h - Config.patch_size + 1, Config.stride)
            for j in range(0, w - Config.patch_size + 1, Config.stride)
        ]
        patches = [
            image[
                :,
                i:min(i + Config.patch_size, h),
                j:min(j + Config.patch_size, w)
            ]
            for i, j in positions
        ]

        return patches, positions

    def __len__(self) -> int:
        return len(self.__noised_data_paths)

    def __getitem__(self, item: int) -> \
            tuple[
                torch.Tensor,
                list[tuple[int, int]],
                torch.Tensor
            ]:
        # Read images
        img_n_raw = self._read_image(self.__noised_data_paths[item])
        img_c_raw = self._read_image(self.__cleaned_data_paths[item])
        x, y = img_n_raw.size

        # Convert images to tensor
        img_n = self.__transform(img_n_raw)
        img_c = self.__transform(img_c_raw)

        # Extract patches from noised image
        pad_x, pad_y = (
            Config.patch_size - x % Config.stride,
            Config.patch_size - y % Config.stride
        )
        img_n = torch.nn.functional.pad(
            img_n,
            (0, pad_x, 0, pad_y),
            mode="constant"
        )

        img_n_patches_list, img_n_patches_positions = self._get_patches(img_n)
        img_n_patches_tensor = torch.stack(img_n_patches_list)
        img_c = img_c.unsqueeze(0)

        return img_n_patches_tensor, img_n_patches_positions, img_c


def rearrange_dataset(dataset: DnCnnDataset) -> Dataset:
    noise_patches = []
    clean_patches = []

    for n_p, c_p in dataset:
        noise_patches.extend(n_p)
        clean_patches.extend(c_p)

    new_pairs = list(zip(noise_patches, clean_patches))
    new_dataset = _DummyDataset(new_pairs)

    return new_dataset
