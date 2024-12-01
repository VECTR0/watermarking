import time

import numpy as np

from src.utils import generate_dummy_image, measure_time


def test_measure_time_on_function() -> None:
    time_sleep = 0.1

    @measure_time
    def simple_addition(a: int, b: int) -> int:
        time.sleep(0.1)
        return a + b

    result, exec_time = simple_addition(3, 7)

    # Validate results
    assert result == 3 + 7
    assert time_sleep < exec_time < 2 * time_sleep


def test_measure_time_on_void_function() -> None:
    time_sleep = 0.05

    @measure_time
    def simple_function() -> None:
        time.sleep(0.05)

    result, exec_time = simple_function()

    # Validate results
    assert result is None
    assert time_sleep < exec_time < 2 * time_sleep


def test_generate_dummy_image() -> None:
    img = generate_dummy_image()

    # Validate the shape and type of the image
    assert isinstance(img, np.ndarray)
    assert img.shape == (256, 256, 3)
    assert img.dtype == np.float32
    assert np.all(img == 100)  # All pixels should be 100
