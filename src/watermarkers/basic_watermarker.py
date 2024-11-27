import numpy as np

from src.dto import Dto
from src.watermarker import Watermarker
from src.watermarker import DecodingResults, EncodingResults


class BasicWatermarker(Watermarker):
    def __init__(self) -> None:
        super().__init__()

    def encode(self, dto: Dto) -> EncodingResults:
        print("Applying Basic Watermark...")
        return (np.ones((64, 64, 3), dtype=np.float32) * 100, 1.32)

    def decode(self, image: np.array) -> DecodingResults:
        print("Decoding Basic Watermark...")
        return ("Basic Watermark", 0.32)

    def get_name(self) -> str:
        return self.__class__.__name__
