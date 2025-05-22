class _ConfigMeta(type):
    def __str__(cls) -> str:
        return f"{cls.__name__}({cls.__str__()})"


# Dummy
class Config(metaclass=_ConfigMeta):
    r = 25
    patch_size = (r << 1) + 1
    image_width = patch_size << 3
    stride = (3 * patch_size) >> 2

    # 1 for small, 4 for large datasets
    num_workers: int = 2
    # ~Optimum
    batch_size: int = 400
    num_layers: int = 20
    num_epochs: int = 50
    learning_rate: float = 1e-3
    gamma: float = 0.25
    milestone: int = 15
    weight_decay: float = 0.
    save_threshold: int = 23.
    validation_img_extraction_step: int = 500
    residual: bool = True

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
