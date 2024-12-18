import numpy as np
from src.types import ImageType
from src.utils import (
    grayscale_image_to_watermark,
    measure_time,
    watermark_to_grayscale_image,
)
from src.watermarker import DecodingResults, EncodingResults, Watermarker


class NaiveWatermarker(Watermarker):
    def __init__(self, amount: float, scale: int, watermark_length: int) -> None:
        super().__init__()
        self.amount = amount
        self.scale = scale
        self.watermark_length = watermark_length

    def __encode(self, image: ImageType, watermark: str) -> EncodingResults:
        watermark_bytes = watermark.encode("utf-8")
        greyscale_watermark = watermark_to_grayscale_image(
            watermark_bytes, image.shape, self.scale
        )
        watermarked_image = image.astype(int) + (
            greyscale_watermark[:, :, None].astype(int) * self.amount
        )
        watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)
        return watermarked_image

    def encode(self, image: ImageType, watermark: str) -> EncodingResults:
        watermarked_image, time_taken = measure_time(self.__encode)(image, watermark)
        return watermarked_image, time_taken

    def __decode(self, image: ImageType, source: ImageType) -> str:
        source_avg = source.mean(axis=(0, 1))
        image_avg = image.mean(axis=(0, 1))
        avg_diff = source_avg - image_avg
        image_avg = np.clip(
            image_avg.astype(int) + avg_diff.astype(int), 0, 255
        ).astype(np.uint8)
        diff = image.astype(int) - source.astype(int)
        diff_avg = diff.mean(axis=2)
        diff_avg = diff_avg.astype(float) / self.amount
        diff_avg = (diff_avg - diff_avg.min()) / (diff_avg.max() - diff_avg.min()) * 255
        diff_avg = np.clip(diff_avg, 0, 255).astype(np.uint8)
        watermark_bytes = grayscale_image_to_watermark(
            diff_avg, self.scale, self.watermark_length
        )
        try:
            watermark = watermark_bytes.decode("utf-8")
        except UnicodeDecodeError:
            watermark = None
        return watermark

    def decode(
        self, image: ImageType, *, source: ImageType | None = None
    ) -> DecodingResults:
        decoded, time_taken = measure_time(self.__decode)(image, source)
        return decoded, time_taken

    def get_name(self) -> str:
        return f"{super().get_name()} amount={self.amount} scale={self.scale} length={self.watermark_length}"
