import copy

import numpy as np

AnalysisResults = dict[str, float | str]
ImageType = np.ndarray


class Dto:
    def __init__(self, filepath: str) -> None:
        # TODO: Optional None troche niewygodny - może wymusić w niektórych metodach aby nie był None?
        self.filepath = filepath
        self.loaded_image = self.load_image()

        self.watermarked_image: ImageType | None = None
        self.watermark: str | None = None
        self.watermarking_time: float | None = None

        self.decoded_watermark: str | None = None
        self.decoding_time: float | None = None
        self.watermarked_analysis_results: list[AnalysisResults] = []

        self.attacked_image: ImageType | None = None
        self.attacked_analysis_results: list[AnalysisResults] = []

    def load_image(self) -> str:
        print(f"Loading image from {self.filepath}")
        return f"Image({self.filepath})"

    def save(self) -> None:
        print(f"Saving analysis results for {self.filepath}...")

    def add_watermarked_results(self, results: AnalysisResults) -> None:
        self.watermarked_analysis_results.append(results)

    def add_attacked_results(self, results: AnalysisResults) -> None:
        self.attacked_analysis_results.append(results)

    def copy(self) -> "Dto":
        return copy.deepcopy(self)
