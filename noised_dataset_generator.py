from utils.noise import (
    dataset_noising,
    mappers,
    readers
)
from utils.nn import config
from utils import __SRC__


if __name__ == "__main__":
    # Noiser stuff
    left, right = 0, 1
    noiser = mappers.Noiser(left, right)

    # Cropper stuff
    cropper = mappers.Cropper(
        config.Config.image_width,
        config.Config.patch_size,
        config.Config.stride
    )

    # Flipper stuff
    flipper = mappers.Flipper()

    mode = ""
    i_path = __SRC__ / "BSDS500-periodic-p" / mode
    e_path = __SRC__ / "BSDS500-periodic-pf" / mode

    num_workers = 4
    mapper = flipper
    reader = readers.CVReader()
    part = None

    dataset_noising.dataset_process(
        i_path,
        e_path,
        num_workers=num_workers,
        part=part,
        mapper=mapper,
        reader=reader
    )
