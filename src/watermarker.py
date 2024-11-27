from src.dto import Dto


class Watermarker:
    _all_watermarkers: list["Watermarker"] = []

    def __init__(self) -> None:
        Watermarker._all_watermarkers.append(self)

    @staticmethod
    def get_all() -> list["Watermarker"]:
        return Watermarker._all_watermarkers

    def watermark(self, dto: Dto) -> Dto:
        raise NotImplementedError
