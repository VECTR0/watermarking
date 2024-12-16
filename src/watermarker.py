from abc import ABC, abstractmethod
from typing import ClassVar

from src.dto import Dto, ImageType

EncodingResults = tuple[ImageType, float]
DecodingResults = tuple[str | None, float]


class Watermarker(ABC):
    _all_watermarkers: ClassVar[list["Watermarker"]] = []

    def __init__(self) -> None:
        Watermarker._all_watermarkers.append(self)

    @staticmethod
    def get_all() -> list["Watermarker"]:
        return Watermarker._all_watermarkers

    # TODO: super decorator
    # TODO: move dto.watermark str to arg here
    @abstractmethod
    def encode(self, image: "ImageType", watermark: str) -> EncodingResults:
        pass

    @abstractmethod
    def decode(self, image: "ImageType") -> DecodingResults:
        pass

    def get_name(self) -> str:
        return self.__class__.__name__
