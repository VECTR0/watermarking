import time
import numpy as np

# from src.dto import ImageType # TODO fix type to ImageType


def log(text: str) -> None:
    LOG = False
    if LOG:
        print(text)  # noqa: T201


def measure_time(func: callable, *args, **kwargs):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        log(f"Method '{func.__name__}' took {execution_time:.4f} seconds to execute.")
        return result, execution_time

    return wrapper


def generate_dummy_image() -> np.array:  # TODO fix type to ImageType
    # TODO: IF smaller than 256x256 remove or skip for Riva Gan
    return np.ones((256, 256, 3), dtype=np.float32) * 100
