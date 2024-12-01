from src.dto import Dto, ImageType
from src.utils import measure_time
from src.watermarker import DecodingResults, EncodingResults, Watermarker
from src.watermarker_invisible import WatermarkInvisibleMethod


class DwtDctWatermarker(Watermarker):
    def __init__(self) -> None:
        super().__init__()

    def encode(self, dto: Dto) -> EncodingResults:
        image, watermark = super().validate_encode_input(dto)

        watermarked_image, time_taken = measure_time(
            WatermarkInvisibleMethod.DWT_DCT.encode
        )(image, watermark)

        new_dto = dto.copy()
        new_dto.watermarked_image = watermarked_image
        new_dto.encoding_time = time_taken

        return watermarked_image, time_taken

    def decode(self, image: ImageType) -> DecodingResults:
        decoded, time_taken = measure_time(WatermarkInvisibleMethod.DWT_DCT.decode)(
            image
        )

        return decoded, time_taken
