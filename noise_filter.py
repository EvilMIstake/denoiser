import enum
import pathlib

import cv2 as cv
import torch
from torchvision import transforms

from utils import (
    utils,
    metrics,
    __SRC__,
    __MODEL_SRC__
)
from utils.nn import dataset, noise_classifier
from utils.nn.dncnn import DnCNN


class NoiseEnum(enum.Enum):
    ADDICTIVE: int = enum.auto()
    BLUR: int = enum.auto()
    IMPULSE: int = enum.auto()
    PERIODIC: int = enum.auto()
    POISSON: int = enum.auto()
    UNKNOWN: int = enum.auto()
    NONE: int = enum.auto()


class Denoiser:
    def __init__(self):
        add_model_pth = __MODEL_SRC__ / "model_add_second.pth"
        blur_model_pth = __MODEL_SRC__ / "model_blur.pth"
        impulse_model_pth = __MODEL_SRC__ / "model_impulse.pth"
        periodic_model_pth = __MODEL_SRC__ / "model_periodic.pth"
        poisson_model_pth = __MODEL_SRC__ / "model_poisson.pth"
        classifier_model_pth = __MODEL_SRC__ / "model_classifier.pth"

        self._add_model = self.get_model(add_model_pth, 20, True)
        self._blur_model = self.get_model(blur_model_pth, 20)
        self._impulse_model = self.get_model(impulse_model_pth, 20)
        self._periodic_model = self.get_model(periodic_model_pth, 20)
        self._poisson_model = self.get_model(poisson_model_pth, 20)
        self._classifier = self.get_classificator(classifier_model_pth, 6)

    @staticmethod
    def denoise(model: DnCNN, tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            tensor = tensor.to(utils.get_device())
            result = model(tensor)

        return result

    @staticmethod
    def get_model(parameters_path: pathlib.Path,
                  num_layers: int,
                  residual: bool = False) -> DnCNN:
        model = DnCNN(
            num_layers=num_layers,
            parameters_path=parameters_path,
            residual=residual
        )
        model = model.to(utils.get_device())

        for param in model.parameters():
            param.requires_grad = False

        model.eval()

        return model

    @staticmethod
    def get_classificator(parameters_path: pathlib.Path,
                          num_classes: int) -> torch.nn.Module:
        model = noise_classifier.get_noise_classifier(num_classes, parameters_path)
        model = model.to(utils.get_device())

        for param in model.parameters():
            param.requires_grad = False

        model.eval()

        return model

    def denoise_add(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.denoise(self._add_model, tensor)

    def denoise_blur(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.denoise(self._blur_model, tensor)

    def denoise_impulse(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.denoise(self._impulse_model, tensor)

    def denoise_periodic(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.denoise(self._periodic_model, tensor)

    def denoise_poisson(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.denoise(self._poisson_model, tensor)

    def denoise_blind(self, tensor: torch.Tensor) -> torch.Tensor:
        """Attention: works with one image in tensor"""

        noise_class, *_ = torch.argmax(self._classifier(tensor), -1).cpu().int()
        de_noised_image = tensor

        match noise_class:
            case 1:
                de_noised_image = self.denoise_add(tensor)
            case 2:
                de_noised_image = self.denoise_impulse(tensor)
            case 3:
                de_noised_image = self.denoise_blur(tensor)
            case 4:
                de_noised_image = self.denoise_periodic(tensor)
            case 5:
                de_noised_image = self.denoise_poisson(tensor)

        return de_noised_image

    def __call__(self, type_: NoiseEnum, tensor: torch.Tensor) -> torch.Tensor:
        res_tensor = tensor

        match type_:
            case NoiseEnum.ADDICTIVE:
                res_tensor = self.denoise_add(tensor)
            case NoiseEnum.BLUR:
                res_tensor = self.denoise_blur(tensor)
            case NoiseEnum.IMPULSE:
                res_tensor = self.denoise_impulse(tensor)
            case NoiseEnum.PERIODIC:
                res_tensor = self.denoise_periodic(tensor)
            case NoiseEnum.POISSON:
                res_tensor = self.denoise_poisson(tensor)
            case NoiseEnum.UNKNOWN:
                res_tensor = self.denoise_blind(tensor)

        return res_tensor


if __name__ == "__main__":
    denoiser = Denoiser()

    img_name = "8068.jpg"
    img_path = __SRC__ / "BSDS500-impulse" / "test" / img_name
    img_real_path = __SRC__ / "BSDS500" / "test" / img_name

    # noinspection PyUnresolvedReferences
    img_n = cv.imread(str(img_path))
    # noinspection PyUnresolvedReferences
    img_n = cv.cvtColor(img_n, cv.COLOR_BGR2RGB)

    # noinspection PyUnresolvedReferences
    img = cv.imread(str(img_real_path))
    # noinspection PyUnresolvedReferences
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    transform = transforms.ToTensor()
    img_tensor = transform(img_n)[None, :, :, :]
    img_d_tensor = denoiser.denoise_impulse(img_tensor)

    img_d_numpy = dataset.DnCnnDataset.to_image(img_d_tensor, clip=True)

    ssim_d = metrics.structure_similarity(img, img_d_numpy)
    ssim_n = metrics.structure_similarity(img, img_n)

    print(f"{ssim_n=:.3f} | {ssim_d=:.3f}")

    # noinspection PyUnresolvedReferences
    img_d_numpy = cv.cvtColor(img_d_numpy, cv.COLOR_RGB2BGR)
    # noinspection PyUnresolvedReferences
    cv.imwrite("test.png", img_d_numpy)
