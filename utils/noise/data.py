import pathlib
import dataclasses

import numpy as np


@dataclasses.dataclass
class LoadData:
    data: np.ndarray
    name: pathlib.Path
