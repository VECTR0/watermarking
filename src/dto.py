import copy

import numpy as np

from src.utils import generate_dummy_image

AnalysisResults = dict[str, float | str]
ImageType = np.ndarray


class Dto:
    def __init__(self, filepath: str) -> None:
        # TODO: Optional None troche niewygodny - może wymusić w niektórych metodach aby nie był None?
        self.filepath = filepath
        self.source_image: ImageType = self.load_image()

        self.watermark_method: str | None = None
        self.watermarked_image: ImageType | None = None
        self.watermark: str | None = None
        self.watermarking_time: float | None = None

        self.decoded_watermark: str | None = None
        self.decoding_time: float | None = None
        self.watermarked_analysis_results: list[AnalysisResults] = []

        self.attack_method: str | None = None
        self.attacked_image: ImageType | None = None
        self.attacked_analysis_results: list[AnalysisResults] = []

    def load_image(self) -> ImageType:
        print(f"Loading image from {self.filepath}")
        # return f"Image({self.filepath})"
        # TODO ASAP load imafe from file
        return generate_dummy_image()

    def save(self) -> None:
        print(f"Saving analysis results for {self.filepath}...")

    def copy(self) -> "Dto":
        return copy.deepcopy(self)
