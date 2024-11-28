from enum import Enum

import numpy as np
from imwatermark import WatermarkDecoder, WatermarkEncoder

# for RivaGAN
WatermarkEncoder.loadModel()


class WatermarkInvisibleMethod(Enum):
    DWT_DCT = "dwtDct"
    DWT_DCT_SVD = "dwtDctSvd"
    RIVA_GAN = "rivaGan"

    def encode(self, image: np.ndarray, watermark: str) -> np.ndarray:
        """
        Encode a watermark into an image using this method.

        :param image: Input image as a numpy array (BGR format)
        :param watermark: Watermark text to embed
        :return: Watermarked image as a numpy array
        """
        if image is None or not isinstance(image, np.ndarray):
            msg = "Invalid input image. Must be a numpy array."
            raise ValueError(msg)

        if not watermark:
            msg = "Watermark text cannot be empty."
            raise ValueError(msg)

        # Initialize encoder
        encoder = WatermarkEncoder()
        encoder.set_watermark("bytes", watermark.encode("utf-8"))

        try:
            # Encode watermark using the method's value
            return encoder.encode(image, self.value)
        except Exception as e:
            raise RuntimeError(
                f"Error encoding watermark with method '{self.name}': {e}"
            )

    def decode(self, image: np.ndarray) -> str | None:
        """
        Decode a watermark from an image using this method.

        :param image: Watermarked image as a numpy array (BGR format)
        :return: Decoded watermark as a string, or None if not found
        """
        if image is None or not isinstance(image, np.ndarray):
            msg = "Invalid input image. Must be a numpy array."
            raise ValueError(msg)

        # Initialize decoder
        decoder = WatermarkDecoder("bytes", 32)  # Adjust key length as necessary

        try:
            # Decode watermark using the method's value
            watermark = decoder.decode(image, self.value)
        except Exception as e:
            msg = f"Error decoding watermark with method '{self.name}': {e}"
            raise RuntimeError(msg)

        if watermark:
            try:
                return watermark.decode("utf-8")
            except UnicodeDecodeError:
                msg = "Decoded watermark could not be converted to UTF-8."
                raise ValueError(msg)
        return None
