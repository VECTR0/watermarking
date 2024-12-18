import time

import cv2
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


def watermark_to_grayscale_image(watermark: bytes, shape, scale) -> ImageType:
    assert isinstance(watermark, bytes)
    if len(shape) == 3:
        shape = shape[:2]
    bits = np.unpackbits(np.frombuffer(watermark, dtype=np.uint8))
    matrix_w, matrix_h = shape[1] // scale, shape[0] // scale
    matrix_bits_len = matrix_w * matrix_h
    bits_to_fill_matrix = []
    for i in range(0, matrix_bits_len):
        bits_to_fill_matrix.append(255 * bits[i % len(bits)])
    bits_matrix = np.array(bits_to_fill_matrix).reshape(matrix_h, matrix_w)
    bits_image = np.kron(bits_matrix, np.ones((scale, scale)))
    blank_image = np.zeros(shape, dtype=np.uint8)
    blank_image[: matrix_h * scale, : matrix_w * scale] = bits_image
    return blank_image


def grayscale_image_to_watermark(image: ImageType, scale, length) -> bytes:
    shape = image.shape
    matrix = image[: shape[0] // scale * scale, : shape[1] // scale * scale]
    matrix = cv2.resize(
        matrix, (shape[1] // scale, shape[0] // scale), interpolation=cv2.INTER_LINEAR
    )
    bits = matrix.flatten()
    avg = np.zeros(length * 8, dtype=np.int32)
    avg_num = np.zeros(length * 8, dtype=np.int32)
    for i in range(len(bits)):
        avg[i % (length * 8)] += int(bits[i])
        avg_num[i % (length * 8)] += 1
    avg = avg / avg_num
    avg = (avg > 127).astype(np.uint8)
    avg = np.packbits(avg)
    return avg.tobytes()
