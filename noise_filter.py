import torch
from torchvision import transforms

import cv2 as cv

from utils.nn.dncnn import DnCNN
from utils.nn import dataset
from utils import (
    utils,
    metrics,
    __SRC__,
    __MODEL_STATES__
)


if __name__ == "__main__":
    # img_path = __SRC__ / "BSDS500" / "val" / "21077.jpg"
    img_path = __SRC__ / "real_noise" / "t.png"
    # img_path = __SRC__ / "BSDS500-periodic" / "val" / "3096.jpg"
    model_path = __MODEL_STATES__ / "DnCNN" / "Model_add_20l_2025-05-02T183920" / "34_epoch.pth"

    # noinspection PyUnresolvedReferences
    img = cv.imread(str(img_path))
    # noinspection PyUnresolvedReferences
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    device = utils.get_device()
    transform = transforms.ToTensor()
    img_tensor = transform(img)[None, :, :, :]
    img_tensor = img_tensor.to(device)

    model = DnCNN(num_layers=20, parameters_path=model_path, residual=True)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        img_d_tensor, *_ = model(img_tensor)

    img_d = dataset.DnCnnDataset.to_image(img_d_tensor, clip=True)

    # noinspection PyUnresolvedReferences
    images = cv.cvtColor(img_d, cv.COLOR_RGB2BGR)
    # noinspection PyUnresolvedReferences
    cv.imwrite("test.png", images)
