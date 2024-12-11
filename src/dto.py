import copy

import cv2
from pydantic import BaseModel

from src.metric import ImageMetricsModel
from src.types import AttackingAnalysisResults, ImageType


class DtoLog(BaseModel):
    filepath: str
    watermarked_analysis_results: ImageMetricsModel
    watermark_method: str
    attack_method: str
    encoding_time: float
    decoding_time: float


class Dto:
    # TODO: The use of Optional None is somewhat inconvenient. Maybe enforce non-None?

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.source_image: ImageType = self.load_image()

        self.watermarked_image: ImageType | None = None
        self.watermark: str | None = None
        self.watermarked_analysis_results: ImageMetricsModel | None = None

        self.decoded_watermark: str | None = None

        self.attacked_image: ImageType | None = None
        self.attacked_analysis_results: list[AttackingAnalysisResults] | None = None

    def load_image(self) -> ImageType:
        return cv2.imread(self.filepath)

    def copy(self) -> "Dto":
        return copy.deepcopy(self)
