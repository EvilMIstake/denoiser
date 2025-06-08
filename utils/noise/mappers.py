import random
from typing import Iterable

import cv2 as cv
import numpy as np

from utils.noise import data, dataset_mapping
from utils.noise.noise import noiser as noiser_


class Noiser(dataset_mapping.IMapper):
    def __init__(self, left: int, right: int):
        """
        :param left: start of the range of noise classes
        :param right: end of the range of noise classes
        """

        self.__left = left
        self.__right = right

    def __call__(self, pipeline_data: data.LoadData) -> Iterable[data.LoadData]:
        idx = random.randint(self.__left, self.__right)
        noiser_entity = noiser_.get_noiser(pipeline_data.data, idx)
        pipeline_data.data = noiser_entity.noised_image()
        return pipeline_data,


class Cropper(dataset_mapping.IMapper):
    def __init__(self,
                 size: int,
                 patch_size: int,
                 stride: int):
        """
        :param size: size of transformed image
        :param patch_size: size of image chunk
        :param stride: step length
        """

        self.__size = size
        self.__patch_size = patch_size
        self.__stride = stride

        patches_size = size // stride
        self.__patches_shape = (
            patches_size,
            patches_size,
            patch_size,
            patch_size,
            3
        )

    def __call__(self, pipeline_data: data.LoadData) -> Iterable[data.LoadData]:
        image = pipeline_data.data
        img_path = pipeline_data.name
        root_img, img_name, img_ext = img_path.parent, img_path.stem, img_path.suffix

        # noinspection PyUnresolvedReferences
        image_resized = cv.resize(
            image,
            (self.__size, self.__size),
            interpolation=cv.INTER_CUBIC
        )

        strides = image_resized.strides
        patches = np.lib.stride_tricks.as_strided(
            np.ravel(image_resized),
            shape=self.__patches_shape,
            strides=(
                self.__stride * strides[0],
                self.__stride * strides[1],
                *strides
            )
        )

        processed_data = tuple(
            data.LoadData(
                data=patch,
                name=root_img / f"{img_name}{i * len(patch_col) + j}{img_ext}"
            )
            for i, patch_col in enumerate(patches)
            for j, patch in enumerate(patch_col)
        )

        return processed_data


class Flipper(dataset_mapping.IMapper):
    def __call__(self, pipeline_data: data.LoadData) -> Iterable[data.LoadData]:
        image = pipeline_data.data
        img_path = pipeline_data.name
        root_img, img_name, img_ext = img_path.parent, img_path.stem, img_path.suffix

        image_flip_horizontal = np.fliplr(image)
        image_flip_vertical = np.flipud(image)

        processed_data = (
            data.LoadData(
                data=image,
                name=root_img / f"{img_name}0{img_ext}"
            ),
            data.LoadData(
                data=image_flip_horizontal,
                name=root_img / f"{img_name}1{img_ext}"
            ),
            data.LoadData(
                data=image_flip_vertical,
                name=root_img / f"{img_name}2{img_ext}"
            )
        )

        return processed_data


class Rotator(dataset_mapping.IMapper):
    def __call__(self, pipeline_data: data.LoadData) -> Iterable[data.LoadData]:
        image = pipeline_data.data
        img_path = pipeline_data.name
        root_img, img_name, img_ext = img_path.parent, img_path.stem, img_path.suffix

        image_flip_rot0 = np.rot90(image, k=1)
        image_flip_rot1 = np.rot90(image, k=2)
        image_flip_rot2 = np.rot90(image, k=3)

        processed_data = (
            data.LoadData(
                data=image,
                name=root_img / f"{img_name}0{img_ext}"
            ),
            data.LoadData(
                data=image_flip_rot0,
                name=root_img / f"{img_name}1{img_ext}"
            ),
            data.LoadData(
                data=image_flip_rot1,
                name=root_img / f"{img_name}2{img_ext}"
            ),
            data.LoadData(
                data=image_flip_rot2,
                name=root_img / f"{img_name}3{img_ext}"
            )
        )

        return processed_data
