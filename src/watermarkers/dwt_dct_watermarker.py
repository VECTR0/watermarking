from src.dto import Dto, ImageType
from src.watermarker import Watermarker
from src.watermarker_invisible import WatermarkInvisibleMethod


class BasicWatermarker(Watermarker):
    def __init__(self) -> None:
        super().__init__()

    # TODO: Może lepiej modyfikować dto zamist zwracć za każdym razem ?
    def watermark(self, dto: Dto) -> Dto:
        image, watermark = dto.watermarked_image, dto.watermark
        assert isinstance(image, ImageType), "sorry bro"
        assert isinstance(watermark, str), ":-()"

        watermarked_image = WatermarkInvisibleMethod.DWT_DCT.encode(image, watermark)

        new_dto = dto.copy()
        new_dto.watermarked_image = watermarked_image

        return new_dto
