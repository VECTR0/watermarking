from src.analyser import measure_time
from src.dto import Dto, ImageType
from src.watermarker import Watermarker
from src.watermarker_invisible import WatermarkInvisibleMethod


class DwtDctWatermarker(Watermarker):
    def __init__(self) -> None:
        super().__init__()

    def watermark(self, dto: Dto) -> Dto:
        image, watermark = dto.watermarked_image, dto.watermark
        assert isinstance(image, ImageType), "sorry bro"
        assert isinstance(watermark, str), ":-()"

        watermarked_image, time_taken = measure_time(
            WatermarkInvisibleMethod.DWT_DCT.encode(image, watermark)
        )

        new_dto = dto.copy()
        new_dto.watermarked_image = watermarked_image
        new_dto.watermarking_time = time_taken

        return new_dto
