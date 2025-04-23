from utils.noise import dataset_noising
from utils import __SRC__


if __name__ == "__main__":
    mode = "val"
    i_path = __SRC__ / "imagenet-mini" / mode
    e_path = __SRC__ / "imagenet-mini-noised" / mode
    left = 0
    right = 5

    dataset_noising.dataset_process(
        i_path,
        e_path,
        left,
        right
    )
