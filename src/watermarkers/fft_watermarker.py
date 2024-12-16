import numpy as np

from src.dto import ImageType
from src.utils import measure_time
from src.watermarker import DecodingResults, EncodingResults, Watermarker


class FFT:
    @staticmethod
    def encode(image: ImageType, watermark: str) -> EncodingResults:
        """
        Encodes a watermark into an image using FFT (Fast Fourier Transform).
        Embeds watermark in the frequency domain by modifying the FFT coefficients.
        """
        # Convert the watermark to a binary string and pad it
        watermark_bin = "".join(format(ord(c), "08b") for c in watermark)
        watermark_bin = watermark_bin.ljust(image.shape[0] * image.shape[1], "0")

        # Perform FFT on the image (2D FFT)
        f_transform = np.fft.fft2(image)
        magnitude = np.abs(f_transform)
        phase = np.angle(f_transform)

        # Embed the watermark in the magnitude
        idx = 0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if idx < len(watermark_bin):
                    if watermark_bin[idx] == "1":
                        magnitude[i, j] += 0.1  # Modify magnitude
                    else:
                        magnitude[i, j] -= 0.1  # Modify magnitude
                idx += 1

        # Recombine modified magnitude and original phase
        f_transform = magnitude * np.exp(1j * phase)

        # Perform inverse FFT to get the watermarked image
        watermarked_image = np.abs(np.fft.ifft2(f_transform))
        return np.uint8(watermarked_image)

    @staticmethod
    def decode(image: ImageType) -> str:
        """
        Decodes the watermark from the image using FFT (Fast Fourier Transform).
        Extracts watermark from the magnitude of the FFT coefficients.
        """
        # Perform FFT on the image (2D FFT)
        f_transform = np.fft.fft2(image)
        magnitude = np.abs(f_transform)

        # Extract the watermark from the magnitude
        watermark_bin = []
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if magnitude[i, j] > 1:  # Threshold based on magnitude change
                    watermark_bin.append("1")
                else:
                    watermark_bin.append("0")

        # Convert binary to characters
        watermark_bin = "".join(watermark_bin)
        watermark = "".join(
            chr(int(watermark_bin[i : i + 8], 2))
            for i in range(0, len(watermark_bin), 8)
        )

        return watermark.strip()


class FFTWatermarker(Watermarker):
    def __init__(self) -> None:
        super().__init__()

    def encode(self, image: ImageType, watermark: str) -> EncodingResults:
        watermarked_image, time_taken = measure_time(FFT.encode)(image, watermark)
        return watermarked_image, time_taken

    def decode(self, image: ImageType) -> DecodingResults:
        decoded, time_taken = measure_time(FFT.decode)(image)
        return decoded, time_taken
