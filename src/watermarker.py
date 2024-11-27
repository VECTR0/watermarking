from src.dto import Dto, ImageType

EncodingResults = tuple[ImageType, float]
DecodingResults = tuple[str | None, float]


class Watermarker:
    _all_watermarkers: list["Watermarker"] = []

    def __init__(self) -> None:
        Watermarker._all_watermarkers.append(self)

    @staticmethod
    def get_all() -> list["Watermarker"]:
        return Watermarker._all_watermarkers

    # TODO: super decorator
    # TODO: move dto.watermark str to arg here
    def encode(self, dto: Dto) -> EncodingResults:
        raise NotImplementedError

    def decode(self, image: ImageType) -> DecodingResults:
        raise NotImplementedError

    def get_name(self) -> str:
        return self.__class__.__name__
