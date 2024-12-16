import numpy as np

from src.dto import ImageType
from src.utils import measure_time
from src.watermarker import DecodingResults, EncodingResults, Watermarker


class LSB:
    @staticmethod
    def encode(image: ImageType, watermark: str) -> EncodingResults:
        """
        Encodes a watermark into an image using LSB (Least Significant Bit).
        Modifies the least significant bit of each pixel to encode the watermark.
        """
        # Convert watermark to binary
        watermark_bin = "".join(format(ord(c), "08b") for c in watermark)
        watermark_bin = watermark_bin.ljust(image.shape[0] * image.shape[1], "0")

        # Encode watermark in the least significant bit of the image pixels
        idx = 0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if idx < len(watermark_bin):
                    # Modify the least significant bit of each pixel
                    image[i, j] = (image[i, j] & 0xFE) | int(watermark_bin[idx])
                    idx += 1
        return image

    @staticmethod
    def decode(image: ImageType) -> str:
        """
        Decodes the watermark from an image using LSB (Least Significant Bit).
        Extracts the watermark by reading the least significant bits of the pixels.
        """

        watermark_bin = [
            str(image[i, j] & 1)
            for i in range(image.shape[0])
            for j in range(image.shape[1])
        ]

        # Convert binary to characters
        watermark_bin = "".join(watermark_bin)
        watermark = "".join(
            chr(int(watermark_bin[i : i + 8], 2))
            for i in range(0, len(watermark_bin), 8)
        )

        return watermark.strip()


class LSBWatermarker(Watermarker):
    def __init__(self) -> None:
        super().__init__()

    def encode(self, image: ImageType, watermark: str) -> EncodingResults:
        watermarked_image, time_taken = measure_time(LSB.encode)(image, watermark)
        return watermarked_image, time_taken

    def decode(self, image: ImageType) -> DecodingResults:
        decoded, time_taken = measure_time(LSB.decode)(image)
        return decoded, time_taken
