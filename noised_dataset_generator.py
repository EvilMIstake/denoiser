from utils import __SRC__
from utils.nn import config
from utils.noise import (
    dataset_mapping,
    mappers,
    readers
)

if __name__ == "__main__":
    # Noiser stuff
    left, right = -1, -1
    noiser = mappers.Noiser(left, right)

    # Cropper stuff
    cropper = mappers.Cropper(
        config.Config.image_width,
        config.Config.patch_size,
        config.Config.stride
    )

    # Flipper stuff
    flipper = mappers.Flipper()

    # Rotator stuff
    rotator = mappers.Rotator()

    mode = "val"
    i_path = __SRC__ / "BSDS500-poisson-pfr" / mode
    e_path = __SRC__ / "BSDS500-noised_classes" / mode / "5"

    num_workers = 4
    mapper = noiser
    reader = readers.CVReader()
    part = slice(0, 1000000, 10)

    dataset_mapping.dataset_process(
        i_path,
        e_path,
        num_workers=num_workers,
        part=part,
        mapper=mapper,
        reader=reader
    )
