from src.utils import measure_time
from src.dto import Dto, ImageType
from src.watermarker import Watermarker, DecodingResults, EncodingResults
from src.watermarker_invisible import WatermarkInvisibleMethod


class DwtDctSvdWatermarker(Watermarker):
    def __init__(self) -> None:
        super().__init__()

    def encode(self, dto: Dto) -> EncodingResults:
        image, watermark = dto.source_image, dto.watermark
        assert isinstance(image, ImageType), "sorry bro"
        assert isinstance(watermark, str), ":-()"

        watermarked_image, time_taken = measure_time(
            WatermarkInvisibleMethod.DWT_DCT_SVD.encode
        )(image, watermark)

        new_dto = dto.copy()
        new_dto.watermarked_image = watermarked_image
        new_dto.encoding_time = time_taken

        return watermarked_image, time_taken

    def decode(self, image: ImageType) -> DecodingResults:
        decoded, time_taken = measure_time(WatermarkInvisibleMethod.DWT_DCT_SVD.decode)(
            image
        )

        return decoded, time_taken

    def get_name(self) -> str:
        return self.__class__.__name__
