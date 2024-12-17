import random
import cv2
import numpy as np

from src.types import ImageType
from src.utils import measure_time
from src.watermarker import DecodingResults, EncodingResults, Watermarker


class FFT:
    alpha = 0.7

    @staticmethod
    def encode(image: ImageType, watermark: str) -> ImageType:
        watermark_bin = "".join(format(ord(char), "08b") for char in watermark)
        wm_length = len(watermark_bin)

        # Perform FFT on the image
        img_f = np.fft.fft2(image)
        magnitude = np.abs(img_f)
        phase = np.angle(img_f)

        height, width, _ = magnitude.shape
        random.seed(height + width)  # Ensure reproducibility
        indices = list(range(height * width))
        random.shuffle(indices)

        # Embed the watermark in the magnitude of the frequency components
        for idx in range(min(wm_length, len(indices))):
            flat_idx = indices[idx]
            i, j = divmod(flat_idx, width)
            bit = int(watermark_bin[idx])
            magnitude[i, j] += FFT.alpha * bit  # Adjust magnitude based on binary bit

        # Reconstruct the frequency domain with modified magnitude and original phase
        watermarked_f = magnitude * np.exp(1j * phase)
        watermarked_image = np.fft.ifft2(watermarked_f)

        ret = np.real(watermarked_image)
        ret = np.clip(ret, 0, 255)
        ret = ret.astype(np.uint8)
        return ret

    @staticmethod
    def decode(image: ImageType, source: ImageType) -> str:

        # ori = cv2.imread(ori_path)
        # img = cv2.imread(img_path)
        ori_f = np.fft.fft2(source)
        img_f = np.fft.fft2(image)
        height, width = source.shape[0], source.shape[1]
        watermark = (ori_f - img_f) / FFT.alpha
        watermark = np.real(watermark)
        res = np.zeros(watermark.shape)
        random.seed(height + width)
        x = range(height / 2)
        y = range(width)
        random.shuffle(x)
        random.shuffle(y)
        for i in range(height / 2):
            for j in range(width):
                res[x[i]][y[j]] = watermark[i][j]
        binary_string = "".join(map(str, res))
        text = "".join(
            chr(int(binary_string[i : i + 8], 2))
            for i in range(0, len(binary_string), 8)
        )
        return text
        # cv2.imwrite(res_path, res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        return
        """
        Decode the watermark from a watermarked image using the original source image.

        Args:
            image (ImageType): The watermarked image.
            source (ImageType): The original source image.

        Returns:
            str: The extracted watermark.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        img_f_watermarked = np.fft.fft2(image)
        img_f_source = np.fft.fft2(source)

        magnitude_watermarked = np.abs(img_f_watermarked)
        magnitude_source = np.abs(img_f_source)

        height, width, _ = magnitude_watermarked.shape

        random.seed(height + width)  # Ensure reproducibility
        indices = list(range(height * width))
        random.shuffle(indices)

        watermark_bin = []

        # Extract the watermark bits from the difference in magnitudes
        for flat_idx in indices:
            i, j = divmod(flat_idx, width)
            diff = magnitude_watermarked[i, j] - magnitude_source[i, j]

            # Decode binary bit based on the difference threshold
            if diff > FFT.alpha / 2:
                watermark_bin.append("1")
            elif diff < -FFT.alpha / 2:
                watermark_bin.append("0")

            # Stop extracting bits if watermark length is determined
            if len(watermark_bin) % 8 == 0 and len(watermark_bin) > 0:
                try:
                    # Attempt partial decoding to ensure validity
                    watermark_str = "".join(
                        chr(int("".join(watermark_bin[i : i + 8]), 2))
                        for i in range(0, len(watermark_bin), 8)
                    )
                    if watermark_str.isascii():
                        break
                except ValueError:
                    continue

        # Convert the final binary watermark to a string
        watermark = "".join(
            chr(int("".join(watermark_bin[i : i + 8]), 2))
            for i in range(0, len(watermark_bin), 8)
        )

        return watermark

    # def decode(self, image: "ImageType", source: "ImageType") -> str:
    # """
    # Decodes the watermark from the image using FFT (Fast Fourier Transform).
    # Extracts watermark from the magnitude of the FFT coefficients.
    # """
    # # Perform FFT on the image (2D FFT)
    # f_transform = np.fft.fft2(image)
    # magnitude = np.abs(f_transform)

    # # Extract the watermark from the magnitude
    # watermark_bin = []
    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         if magnitude[i, j] > 1:  # Threshold based on magnitude change
    #             watermark_bin.append("1")
    #         else:
    #             watermark_bin.append("0")

    # # Convert binary to characters
    # watermark_bin = "".join(watermark_bin)
    # watermark = "".join(
    #     chr(int(watermark_bin[i : i + 8], 2))
    #     for i in range(0, len(watermark_bin), 8)
    # )

    # return watermark.strip()


class FFTWatermarker(Watermarker):
    def __init__(self) -> None:
        super().__init__()

    def encode(self, image: ImageType, watermark: str) -> EncodingResults:
        watermarked_image, time_taken = measure_time(FFT.encode)(image, watermark)
        return watermarked_image, time_taken

    def decode(
        self, image: ImageType, *, source: ImageType | None = None
    ) -> DecodingResults:
        assert source is not None, "Source image reqiured for FFT decode"
        decoded, time_taken = measure_time(FFT.decode)(image, source)
        return decoded, time_taken
