from utils.noise import dataset_noising
from utils import __SRC__


if __name__ == "__main__":
    mode = "val"
    i_path = __SRC__ / "imagenet-mini-shrink" / mode
    e_path = __SRC__ / "imagenet-mini-shrink-add" / mode
    left = 0
    right = 2

    dataset_noising.dataset_process(
        i_path,
        e_path,
        left,
        right,
        num_workers=4
    )
