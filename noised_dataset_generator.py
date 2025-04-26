from utils.noise import dataset_noising, mappers
from utils.nn import utils as nn_utils
from utils import __SRC__


if __name__ == "__main__":
    # Noiser stuff
    left, right = 1, 1
    noiser = mappers.Noiser(left, right)

    # Cropper stuff
    cropper = mappers.Cropper(
        nn_utils.Config.image_width,
        nn_utils.Config.patch_size,
        nn_utils.Config.stride
    )

    mode = "train"
    i_path = __SRC__ / "imagenet-mini-shrink-uniform" / mode
    e_path = __SRC__ / "imagenet-mini-shrink-uniform-p" / mode

    num_workers = 8
    mapper = cropper
    part = None

    dataset_noising.dataset_process(
        i_path,
        e_path,
        num_workers=num_workers,
        part=part,
        mapper=mapper
    )
