from src.types import ImageType
from src.utils import measure_time
from src.watermarker import DecodingResults, EncodingResults, Watermarker
from src.watermarker_invisible import WatermarkInvisibleMethod


class DwtDctWatermarker(Watermarker):
    def __init__(self) -> None:
        super().__init__()

    def encode(self, image: ImageType, watermark: str) -> EncodingResults:
        watermarked_image, time_taken = measure_time(
            WatermarkInvisibleMethod.DWT_DCT.encode
        )(image, watermark)
        return watermarked_image, time_taken

    def decode(self, image: ImageType) -> DecodingResults:
        decoded, time_taken = measure_time(WatermarkInvisibleMethod.DWT_DCT.decode)(
            image
        )
        return decoded, time_taken
