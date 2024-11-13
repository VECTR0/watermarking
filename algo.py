import cv2
from imwatermark import WatermarkEncoder, WatermarkDecoder
from enum import Enum
from typing import Union


# https://github.com/ShieldMnt/invisible-watermark
class WatermarkMethod(Enum):
    DWT_DCT = "dwtDct"
    DWT_DCT_SVD = "dwtDctSvd"
    RIVA_GAN = "rivaGan"


def encode_watermark(
    image_path: str, watermark: str, method: WatermarkMethod, output_path: str
) -> None:
    """
    Encode watermark into an image using the specified method.

    :param image_path: Path to the input image
    :param watermark: Watermark text to be embedded
    :param method: Watermark method to use (e.g., 'dwtDct', 'rivagan')
    :param output_path: Path to save the watermarked image
    """
    # Read the image
    bgr = cv2.imread(image_path)

    # Initialize encoder
    encoder = WatermarkEncoder()
    encoder.set_watermark("bytes", watermark.encode("utf-8"))

    # Encode the watermark using the chosen method
    bgr_encoded = encoder.encode(bgr, method.value)

    # Save the watermarked image
    cv2.imwrite(output_path, bgr_encoded)
    print(f"Watermarked image saved at {output_path}")


def decode_watermark(image_path: str, method: WatermarkMethod) -> Union[str, None]:
    """
    Decode watermark from the image using the specified method.

    :param image_path: Path to the watermarked image
    :param method: Watermark method to use (e.g., 'dwtDct', 'rivagan')
    :return: The decoded watermark as a string, or None if no watermark is found
    """
    # Read the watermarked image
    bgr = cv2.imread(image_path)

    # Initialize decoder
    decoder = WatermarkDecoder("bytes", 32)

    # Decode the watermark using the chosen method
    watermark = decoder.decode(bgr, method.value)

    # Return the decoded watermark text
    if watermark:
        print("w", watermark)
        # return watermark.decode("utf-8")
        return watermark.decode("utf-8")
    return None


# Example usage
if __name__ == "__main__":
    # Define watermark and file paths
    watermark_text = "test"
    input_image_path = "./cat.jpg"
    output_image_path = "test_wm.png"

    for method in WatermarkMethod:
        # Encode watermark
        encode_watermark(input_image_path, watermark_text, method, output_image_path)

        # Decode watermark
        decoded_watermark = decode_watermark(output_image_path, method)
        # TODO: metrics?? encoded vs decoded images ; decoded watermark
        # TODO: benchmark??
        if decoded_watermark:
            print(f"Decoded watermark: {decoded_watermark}")
        else:
            print("No watermark found")
