import os
import time
import queue
import random
import pathlib
import threading
import dataclasses
from typing import Generator
from concurrent import futures

import tqdm
import cv2 as cv
import numpy as np

from utils.noise.utils import noiser
from utils.utils import get_all_paths


@dataclasses.dataclass
class LoadData:
    data: np.ndarray
    name: pathlib.Path


def generator(file_names: list[pathlib.Path],
              generator_queue: queue.Queue[LoadData | None]) -> None:
    for f_name in tqdm.tqdm(file_names, postfix=f"[PID] {os.getpid()}"):
        # noinspection PyUnresolvedReferences
        img = cv.imread(str(f_name))
        img_name = pathlib.Path(*f_name.parts[-2:])
        data_container = LoadData(img, img_name)
        generator_queue.put_nowait(data_container)

    generator_queue.put_nowait(None)


def processor(generator_queue: queue.Queue[LoadData | None],
              receiver_queue: queue.Queue[LoadData | None],
              left: int,
              right: int) -> None:
    """
    :param generator_queue
    :param receiver_queue
    :param left: start of the range of noise classes
    :param right: end of the range of noise classes
    """

    assert left <= right

    while True:
        data_container = generator_queue.get(block=True)

        if data_container is None:
            break

        idx = random.randint(left, right)
        noiser_ = noiser.get_noiser(data_container.data, idx)
        data_container.data = noiser_.noised_image()
        receiver_queue.put_nowait(data_container)

    receiver_queue.put_nowait(None)


def receiver(export_path: pathlib.Path,
             receiver_queue: queue.Queue[LoadData | None]) -> None:
    while True:
        load_data = receiver_queue.get(block=True)

        if load_data is None:
            break

        img_name, img = load_data.name, load_data.data
        img_path = export_path / img_name

        dir_path = img_path.parent
        if not dir_path.exists():
            dir_path.mkdir(parents=True)

        # noinspection PyUnresolvedReferences
        cv.imwrite(
            str(img_path),
            load_data.data
        )


def worker(file_names: list[pathlib.Path],
           to: pathlib.Path,
           left: int,
           right: int) -> None:
    random.seed(os.getpid() * int(time.time()) % 31_415_926_535)

    g_queue = queue.Queue()
    r_queue = queue.Queue()

    r_thread = threading.Thread(
        target=receiver,
        args=(
            to,
            r_queue
        )
    )
    p_thread = threading.Thread(
        target=processor,
        args=(
            g_queue,
            r_queue,
            left,
            right
        )
    )
    r_thread.start()
    p_thread.start()

    generator(
        file_names,
        g_queue
    )

    r_thread.join()
    p_thread.join()


def dataset_process(from_: pathlib.Path,
                    to: pathlib.Path,
                    left: int,
                    right: int,
                    part: slice | None = None,
                    num_workers: int = 8) -> None:
    def chunks_generator() -> Generator[list[pathlib.Path], None, None]:
        nonlocal num_workers
        assert num_workers > 0, "Num workers must be non negative"

        file_names = get_all_paths(from_)

        if part is not None:
            file_names = file_names[part]

        n = len(file_names)

        assert n > 0, "Files not found"

        num_workers = min(n, num_workers)
        chunk_size = n // num_workers

        for i in range(num_workers - 1):
            yield file_names[i * chunk_size:(i + 1) * chunk_size]

        yield file_names[(num_workers - 1) * chunk_size:]

    t = time.time()

    with futures.ProcessPoolExecutor() as exc:
        futures_ = [
            exc.submit(
                worker,
                chunk,
                to,
                left,
                right
            )
            for chunk in chunks_generator()
        ]

        for future in futures_:
            future.result()

    t = time.time() - t
    print(f"Total time: {t:.2f}s")
