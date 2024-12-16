from src.types import ImageType
from src.utils import measure_time
from src.watermarker import DecodingResults, EncodingResults, Watermarker
from src.watermarker_invisible import WatermarkInvisibleMethod


class RivaGanWatermarker(Watermarker):
    def __init__(self) -> None:
        super().__init__()

    def encode(self, image: ImageType, watermark: str) -> EncodingResults:
        assert len(watermark) == 4, "Watermark string must be 32 bits"
        watermarked_image, time_taken = measure_time(
            WatermarkInvisibleMethod.RIVA_GAN.encode
        )(image, watermark)
        return watermarked_image, time_taken

    def decode(self, image: ImageType) -> DecodingResults:
        decoded, time_taken = measure_time(WatermarkInvisibleMethod.RIVA_GAN.decode)(
            image
        )
        return decoded, time_taken
