import time
import cv2
from imwatermark import WatermarkEncoder, WatermarkDecoder
from enum import Enum
import numpy as np
from typing import Tuple, Union
from dataset.attack import attack_functions, 


# https://github.com/ShieldMnt/invisible-watermark
class WatermarkMethod(Enum):
    DWT_DCT = "dwtDct"
    DWT_DCT_SVD = "dwtDctSvd"
    RIVA_GAN = "rivaGan"


# def encode_watermark(
#     image_path: str, watermark: str, method: WatermarkMethod, output_path: str
# ) -> None:
#     """
#     Encode watermark into an image using the specified method.

#     :param image_path: Path to the input image
#     :param watermark: Watermark text to be embedded
#     :param method: Watermark method to use (e.g., 'dwtDct', 'rivagan')
#     :param output_path: Path to save the watermarked image
#     """
#     # Read the image
#     bgr = cv2.imread(image_path)

#     # Initialize encoder
#     encoder = WatermarkEncoder()
#     encoder.set_watermark("bytes", watermark.encode("utf-8"))

#     # Encode the watermark using the chosen method
#     bgr_encoded = encoder.encode(bgr, method.value)

#     # Save the watermarked image
#     cv2.imwrite(output_path, bgr_encoded)
#     print(f"Watermarked image saved at {output_path}")


# def decode_watermark(image_path: str, method: WatermarkMethod) -> Union[str, None]:
#     """
#     Decode watermark from the image using the specified method.

#     :param image_path: Path to the watermarked image
#     :param method: Watermark method to use (e.g., 'dwtDct', 'rivagan')
#     :return: The decoded watermark as a string, or None if no watermark is found
#     """
#     # Read the watermarked image
#     bgr = cv2.imread(image_path)

#     # Initialize decoder
#     decoder = WatermarkDecoder("bytes", 32)

#     # Decode the watermark using the chosen method
#     watermark = decoder.decode(bgr, method.value)

#     # Return the decoded watermark text
#     if watermark:
#         print("w", watermark)
#         # return watermark.decode("utf-8")
#         return watermark.decode("utf-8")
#     return None


def encode_watermark(
    image: np.ndarray, watermark: str, method: WatermarkMethod
) -> np.ndarray:
    """
    Encode watermark into an image using the specified method.

    :param image: Input image as a numpy array (BGR format)
    :param watermark: Watermark text to be embedded
    :param method: Watermark method to use (e.g., 'dwtDct', 'rivagan')
    :return: Watermarked image as a numpy array
    """
    # Initialize encoder
    encoder = WatermarkEncoder()
    encoder.set_watermark("bytes", watermark.encode("utf-8"))

    # Encode the watermark using the chosen method
    bgr_encoded = encoder.encode(image, method.value)

    return bgr_encoded


def decode_watermark(image: np.ndarray, method: WatermarkMethod) -> Union[str, None]:
    """
    Decode watermark from the image using the specified method.

    :param image: Watermarked image as a numpy array (BGR format)
    :param method: Watermark method to use (e.g., 'dwtDct', 'rivagan')
    :return: The decoded watermark as a string, or None if no watermark is found
    """
    # Initialize decoder
    decoder = WatermarkDecoder("bytes", 32)

    # Decode the watermark using the chosen method
    watermark = decoder.decode(image, method.value)

    # Return the decoded watermark text
    if watermark:
        return watermark.decode("utf-8")
    return None


# Example usage
if __name__ == "__main__":
    # Define watermark and file paths
    watermark_text = "test"
    # input_image_path = "./cat.jpg"
    # output_image_path = "test_wm.png"

    def benchmark_watermarking(
        image: np.ndarray, watermark: str, method: WatermarkMethod
    ) -> Tuple[float, float]:
        start_encode = time.time()
        watermarked_image = encode_watermark(image, watermark, method)
        end_encode = time.time()

        start_decode = time.time()
        decoded_watermark = decode_watermark(watermarked_image, method)
        end_decode = time.time()

        return end_encode - start_encode, end_decode - start_decode, decoded_watermark

    def process(file_path):
        for method in WatermarkMethod:
            bgr_image = cv2.imread(file_path)

            # Encode watermark
            watermarked_image = encode_watermark(bgr_image, watermark_text, method)
            # wczytaj

            # Decode watermark
            decoded_watermark = decode_watermark(watermarked_image, method)
            # TODO: metrics?? encoded vs decoded images ; decoded watermark
            # TODO: benchmark??
            if decoded_watermark:
                print(f"Decoded watermark: {decoded_watermark}")
            else:
                print("No watermark found")

            for attack in attack_functions:
                attacked_image = apply_attack(bgr_image, attack)

                # Encode the watermark on the attacked image and try decoding
                attacked_watermarked_image = encode_watermark(
                    attacked_image, watermark_text, method
                )
                attacked_decoded_watermark = decode_watermark(
                    attacked_watermarked_image, method
                )

                # Measure similarity between watermarked image and attacked watermarked image
                similarity = calculate_image_similarity(
                    attacked_image, attacked_watermarked_image
                )

                print(f"Attack: {attack}, Similarity: {similarity:.4f}")
                if attacked_decoded_watermark:
                    print(
                        f"Decoded Watermark After {attack}: {attacked_decoded_watermark}"
                    )
                else:
                    print(f"Watermark could not be decoded after {attack}")
