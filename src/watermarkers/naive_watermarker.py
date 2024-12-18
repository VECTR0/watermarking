from src.types import ImageType
from src.utils import (
    grayscale_image_to_watermark,
    measure_time,
    watermark_to_grayscale_image,
)
from src.watermarker import DecodingResults, EncodingResults, Watermarker


class NaiveWatermarker(Watermarker):
    def __init__(self, scale, watermark_length) -> None:
        super().__init__()
        self.scale = scale
        self.watermark_length = watermark_length

    def __encode(self, image: ImageType, watermark: str) -> EncodingResults:
        print("NaiveWatermarker.__encode")
        watermark_bytes = watermark.encode("utf-8")
        greyscale_watermark = watermark_to_grayscale_image(
            watermark_bytes, image.shape, self.scale
        )
        watermarked_image = image + greyscale_watermark[:, :, None]
        return watermarked_image

    def encode(self, image: ImageType, watermark: str) -> EncodingResults:
        watermarked_image, time_taken = measure_time(self.__encode)(image, watermark)
        return watermarked_image, time_taken

    def __decode(self, image: ImageType, source: ImageType) -> str:
        diff = image - source
        diff_avg = diff.mean(axis=2)
        watermark_bytes = grayscale_image_to_watermark(
            diff_avg, self.scale, self.watermark_length
        )
        watermark = watermark_bytes.decode("utf-8")
        return watermark

    def decode(
        self, image: ImageType, *, source: ImageType | None = None
    ) -> DecodingResults:
        decoded, time_taken = measure_time(self.__decode)(image, source)
        return decoded, time_taken

    def get_name(self) -> str:
        return f"{super().get_name()} scale={self.scale} length={self.watermark_length}"
