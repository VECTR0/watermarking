from src.dto import Dto, ImageType
from src.utils import measure_time
from src.watermarker import DecodingResults, EncodingResults, Watermarker
from src.watermarker_invisible import WatermarkInvisibleMethod


class RivaGanWatermarker(Watermarker):
    # TODO: Currently only 32-bit watermarks are supported.
    #       Update the implementation to support watermarks of different bit depths.

    def __init__(self) -> None:
        super().__init__()

    def encode(self, dto: Dto) -> EncodingResults:
        image, watermark = super().validate_encode_input(dto)

        watermarked_image, time_taken = measure_time(
            WatermarkInvisibleMethod.RIVA_GAN.encode
        )(image, watermark)

        new_dto = dto.copy()
        new_dto.watermarked_image = watermarked_image
        new_dto.encoding_time = time_taken

        return watermarked_image, time_taken

    def decode(self, image: ImageType) -> DecodingResults:
        decoded, time_taken = measure_time(WatermarkInvisibleMethod.RIVA_GAN.decode)(
            image
        )

        return decoded, time_taken
