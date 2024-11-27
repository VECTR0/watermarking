from enum import Enum
from typing import Callable
import cv2
import numpy as np

# Type alias for image arrays (grayscale or RGB)
ImageArray = np.ndarray


class WatermarkAttack(Enum):
    BLUR = "blur"
    SHARPEN = "sharpen"
    MEDIAN = "median"
    JPEG_COMPRESSION = "jpeg_compression"
    SALT_AND_PEPPER_NOISE = "salt_and_pepper_noise"
    GAUSSIAN_NOISE = "gaussian_noise"


# 1. Filtering (Blurring, Sharpening, Median Filtering)


def blur_filter(image: ImageArray, ksize: int = 5) -> ImageArray:
    """Apply blurring filter."""
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def sharpen_filter(image: ImageArray) -> ImageArray:
    """Apply sharpening filter."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def median_filter(image: ImageArray, ksize: int = 5) -> ImageArray:
    """Apply median filter."""
    return cv2.medianBlur(image, ksize)


# 2. Compression (JPEG Compression)


def jpeg_compression(image: ImageArray, quality: int = 50) -> ImageArray:
    """Apply JPEG compression."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_img = cv2.imencode(".jpg", image, encode_param)
    return cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)


# 3. Noise Addition (Salt-and-Pepper, Gaussian)


def add_salt_and_pepper_noise(image: ImageArray, amount: float = 0.01) -> ImageArray:
    """Add salt-and-pepper noise."""
    noisy_image = image.copy()
    num_salt = int(np.ceil(amount * image.size * 0.5))
    num_pepper = int(np.ceil(amount * image.size * 0.5))

    # Add Salt
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 255

    # Add Pepper
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 0

    return noisy_image


def add_gaussian_noise(
    image: ImageArray, mean: float = 0, var: float = 0.01
) -> ImageArray:
    """Add Gaussian noise."""
    sigma = var**0.5
    gaussian = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = cv2.addWeighted(image.astype(np.float32), 1.0, gaussian, 1.0, 0.0)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


# 4. Cropping and Resizing


def crop_image(image: ImageArray, crop_fraction: float = 0.2) -> ImageArray:
    """Crop a portion of the image."""
    h, w = image.shape[:2]
    start_h, end_h = int(h * crop_fraction), int(h * (1 - crop_fraction))
    start_w, end_w = int(w * crop_fraction), int(w * (1 - crop_fraction))
    return image[start_h:end_h, start_w:end_w]


def resize_image(image: ImageArray, scale: float = 0.5) -> ImageArray:
    """Resize image by a given scale."""
    h, w = image.shape[:2]
    return cv2.resize(
        image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR
    )


# 5. Histogram Equalization and Intensity Modification


def histogram_equalization(image: ImageArray) -> ImageArray:
    """Apply histogram equalization."""
    if len(image.shape) == 2:  # Grayscale image
        return cv2.equalizeHist(image)
    elif len(image.shape) == 3:  # Color image
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return image


def intensity_modification(
    image: ImageArray, alpha: float = 1.2, beta: int = 30
) -> ImageArray:
    """Modify intensity by changing contrast (alpha) and brightness (beta)."""
    modified_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return modified_image


attack_functions: dict[WatermarkAttack, tuple[Callable, dict]] = {
    WatermarkAttack.BLUR: (blur_filter, {"ksize": 5}),
    WatermarkAttack.SHARPEN: (sharpen_filter, {}),
    WatermarkAttack.MEDIAN: (median_filter, {"ksize": 5}),
    WatermarkAttack.JPEG_COMPRESSION: (jpeg_compression, {"quality": 50}),
    WatermarkAttack.SALT_AND_PEPPER_NOISE: (
        add_salt_and_pepper_noise,
        {"amount": 0.01},
    ),
    WatermarkAttack.GAUSSIAN_NOISE: (add_gaussian_noise, {"mean": 0, "sigma": 15}),
}


if __name__ == "__main__":
    # Example usage with an input image
    image = cv2.imread("watermarked_image.png")

    # Apply attacks
    blurred_image = blur_filter(image)
    sharpened_image = sharpen_filter(image)
    median_filtered_image = median_filter(image)
    compressed_image = jpeg_compression(image)
    salt_pepper_noisy_image = add_salt_and_pepper_noise(image)
    gaussian_noisy_image = add_gaussian_noise(image)
    cropped_image = crop_image(image)
    resized_image = resize_image(image)
    equalized_image = histogram_equalization(image)
    intensity_modified_image = intensity_modification(image)

    # Save or display results
    cv2.imwrite("blurred_image.png", blurred_image)
    cv2.imwrite("sharpened_image.png", sharpened_image)
    cv2.imwrite("median_filtered_image.png", median_filtered_image)
    cv2.imwrite("compressed_image.jpg", compressed_image)
    cv2.imwrite("salt_pepper_noisy_image.png", salt_pepper_noisy_image)
    cv2.imwrite("gaussian_noisy_image.png", gaussian_noisy_image)
    cv2.imwrite("cropped_image.png", cropped_image)
    cv2.imwrite("resized_image.png", resized_image)
    cv2.imwrite("equalized_image.png", equalized_image)
    cv2.imwrite("intensity_modified_image.png", intensity_modified_image)
