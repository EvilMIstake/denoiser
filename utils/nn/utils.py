import pathlib

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset

from utils.utils import get_all_paths, get_device, to_device


class Config:
    r = 25
    patch_size = (r << 1) + 1
    image_width = patch_size << 3
    stride = (3 * patch_size) >> 2

    batch_size: int = 8
    num_layers: int = 17
    num_workers: int = 2
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4


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
    def from_patches(patches: list[torch.Tensor],
                     positions: list[tuple[int, int]],
                     shape: tuple[int, int, int, int]) -> np.ndarray:
        _, c, h, w = shape
        device = get_device()
        image = torch.zeros(shape, device=device)
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
        image_numpy = _DatasetMixins.to_image(image)

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
        device = get_device()
        img_n = to_device(self.__transforms(img_n_raw), device)
        img_c = to_device(self.__transforms(img_c_raw), device)

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
        image = image.unsqueeze(0)

        # Cutting overlapping patches without using edges (right/bottom)
        positions: list[tuple[int, int]] = [
            (i, j)
            for i in range(0, h, Config.stride)
            for j in range(0, w, Config.stride)
        ]
        patches = [
            image[
                :,
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
                list[torch.Tensor],
                list[tuple[int, int]],
                torch.Tensor
            ]:
        # Read images
        img_n_raw = self._read_image(self.__noised_data_paths[item])
        img_c_raw = self._read_image(self.__cleaned_data_paths[item])

        # Convert images to tensor
        device = get_device()
        img_n = to_device(self.__transform(img_n_raw), device)
        img_c = to_device(self.__transform(img_c_raw), device)

        # Extract patches from noised image
        img_n_patches_tensor, img_n_patches_positions = self._get_patches(img_n)
        img_c = img_c.unsqueeze(0)

        return img_n_patches_tensor, img_n_patches_positions, img_c


def rearrange_dataset(dataset: DnCnnDataset, shuffle: bool) -> torch.utils.data.DataLoader:
    noise_patches = []
    clean_patches = []

    for n_p, c_p in dataset:
        noise_patches.extend(n_p)
        clean_patches.extend(c_p)

    new_pairs = list(zip(noise_patches, clean_patches))
    data_loader = torch.utils.data.DataLoader(
        _DummyDataset(new_pairs),
        batch_size=Config.batch_size,
        shuffle=shuffle
    )

    return data_loader
