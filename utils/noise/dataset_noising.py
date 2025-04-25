import os
import time
import queue
import random
import pathlib
import threading
from typing import (
    Generator,
    Iterable
)
from concurrent import futures
from abc import ABC, abstractmethod

import tqdm
import cv2 as cv

from utils.utils import get_all_paths
from utils.noise import data


class IMapper(ABC):
    @abstractmethod
    def __call__(self, pipeline_data: data.LoadData) -> Iterable[data.LoadData]:
        ...


def generator(file_names: list[pathlib.Path],
              generator_queue: queue.Queue[data.LoadData | None]) -> None:
    for f_name in file_names:
        # noinspection PyUnresolvedReferences
        img = cv.imread(str(f_name))
        img_name = pathlib.Path(*f_name.parts[-2:])
        data_container = data.LoadData(img, img_name)
        generator_queue.put_nowait(data_container)

    generator_queue.put_nowait(None)


def processor(generator_queue: queue.Queue[data.LoadData | None],
              receiver_queue: queue.Queue[data.LoadData | None],
              mapper: IMapper) -> None:
    while True:
        data_container = generator_queue.get(block=True)

        if data_container is None:
            break

        mapped_data = mapper(data_container)
        for md in mapped_data:
            receiver_queue.put_nowait(md)

    receiver_queue.put_nowait(None)


def receiver(export_path: pathlib.Path,
             receiver_queue: queue.Queue[data.LoadData | None]) -> None:
    with tqdm.tqdm(total=0, postfix=f"[PID] {os.getpid()}") as tqdm_:
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

            tqdm_.update(1)


def worker(file_names: list[pathlib.Path],
           to: pathlib.Path,
           mapper: IMapper) -> None:
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
            mapper
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
                    mapper: IMapper,
                    part: slice | None = None,
                    num_workers: int = 8) -> None:
    def chunks_generator() -> Generator[list[pathlib.Path], None, None]:
        nonlocal num_workers
        assert num_workers > 0, "Num workers must be positive integer"

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
                mapper
            )
            for chunk in chunks_generator()
        ]

        for future in futures_:
            future.result()

    t = time.time() - t
    print(f"Total time: {t:.2f}s")
