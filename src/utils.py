import time

import numpy as np

from src.config import logger
from src.types import ImageType


def measure_time(func: callable) -> callable:
    def wrapper(*args: any, **kwargs: any) -> tuple[any, float]:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.log(
            f"Method '{func.__name__}' took {execution_time:.4f} seconds to execute."
        )
        return result, execution_time

    return wrapper


def generate_dummy_image() -> ImageType:
    return np.ones((256, 256, 3), dtype=np.float32) * 100
