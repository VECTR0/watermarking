import numpy as np

from src.dto import Dto
from src.watermarker import Watermarker


class BasicWatermarker(Watermarker):
    def __init__(self) -> None:
        super().__init__()

    def watermark(self, dto: Dto) -> Dto:
        print("Applying Basic Watermark...")

        new_dto = dto.copy()
        new_dto.watermarked_image = np.ones((64, 64, 3), dtype=np.float32) * 100
        return dto
