import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

ImageArray = np.ndarray


# PSNR Calculation
def psnr(original: ImageArray, watermarked: ImageArray) -> float:
    mse = np.mean((original - watermarked) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


# SSIM Calculation
def calculate_ssim(original: ImageArray, watermarked: ImageArray) -> float:
    return ssim(original, watermarked, multichannel=True)


# Simple Correlation Coefficient (SCC) and Normalized Correlation Coefficient (NCC)
def correlation_coefficient(original: ImageArray, extracted: ImageArray) -> float:
    original_flat = original.flatten()
    extracted_flat = extracted.flatten()
    return np.corrcoef(original_flat, extracted_flat)[0, 1]


def normalized_correlation_coefficient(
    original: ImageArray, extracted: ImageArray
) -> float:
    original_flat = original.flatten()
    extracted_flat = extracted.flatten()
    norm_corr = np.dot(original_flat, extracted_flat) / (
        np.linalg.norm(original_flat) * np.linalg.norm(extracted_flat)
    )
    return norm_corr


# Bit Error Rate (BER)
def bit_error_rate(original: ImageArray, extracted: ImageArray) -> float:
    # Convert images to binary format (assumes binary watermark)
    original_binary = (original > 127).astype(int)
    extracted_binary = (extracted > 127).astype(int)
    num_bit_errors = np.sum(original_binary != extracted_binary)
    total_bits = original_binary.size
    return num_bit_errors / total_bits


# Root Mean Squared Error (RMSE)
def rmse(original: ImageArray, watermarked: ImageArray) -> float:
    return np.sqrt(mean_squared_error(original.flatten(), watermarked.flatten()))


if __name__ == "__main__":
    # Sample usage with example images (assumes grayscale or RGB images loaded via cv2)
    original_img: ImageArray = cv2.imread(
        "./alps-tree-snow-nature-landscape-flowers-grass.jpg", cv2.IMREAD_COLOR
    )
    watermarked_img: ImageArray = cv2.imread(
        "./alps-tree-snow-nature-landscape-flowers-grass.jpg", cv2.IMREAD_COLOR
    )

    #
    extracted_watermark: ImageArray = cv2.imread(
        "extracted_watermark.png", cv2.IMREAD_GRAYSCALE
    )
    original_watermark: ImageArray = cv2.imread(
        "original_watermark.png", cv2.IMREAD_GRAYSCALE
    )

    # Calculate PSNR and SSIM
    psnr_value: float = psnr(original_img, watermarked_img)
    ssim_value: float = calculate_ssim(original_img, watermarked_img)

    # Calculate Correlation Coefficients
    scc_value: float = correlation_coefficient(original_watermark, extracted_watermark)
    ncc_value: float = normalized_correlation_coefficient(
        original_watermark, extracted_watermark
    )

    # Calculate Bit Error Rate
    ber_value: float = bit_error_rate(original_watermark, extracted_watermark)

    # Calculate RMSE
    rmse_value: float = rmse(original_img, watermarked_img)

    # Print results
    print("PSNR (Imperceptibility):", psnr_value)
    print("SSIM (Imperceptibility):", ssim_value)
    print("SCC (Robustness):", scc_value)
    print("NCC (Robustness):", ncc_value)
    print("Bit Error Rate (BER):", ber_value)
    print("RMSE (Compression Impact):", rmse_value)

    #
    # another file
    #
    
'''

'''

    
    def benchmark(img):
        pass
    
    # Apply attacks from attack.py
    from attack import (
        blur_filter,
        sharpen_filter,
        median_filter,
        jpeg_compression,
        add_salt_and_pepper_noise,
        add_gaussian_noise,
        crop_image,
        resize_image,
        histogram_equalization,
        intensity_modification,
    )

    # Apply different attacks
    attacked_images = {
        "blurred": blur_filter(watermarked_img),
        "sharpened": sharpen_filter(watermarked_img),
        "median_filtered": median_filter(watermarked_img),
        "compressed": jpeg_compression(watermarked_img),
        "salt_pepper_noisy": add_salt_and_pepper_noise(watermarked_img),
        "gaussian_noisy": add_gaussian_noise(watermarked_img),
        "cropped": crop_image(watermarked_img),
        "resized": resize_image(watermarked_img),
        "equalized": histogram_equalization(watermarked_img),
        "intensity_modified": intensity_modification(watermarked_img),
    }

    # Evaluate metrics for each attacked image
    for attack_name, attacked_img in attacked_images.items():
        psnr_value = psnr(original_img, attacked_img)
        ssim_value = calculate_ssim(original_img, attacked_img)
        rmse_value = rmse(original_img, attacked_img)
        print(f"Attack: {attack_name}")
        print(f"PSNR: {psnr_value}")
        print(f"SSIM: {ssim_value}")
        print(f"RMSE: {rmse_value}")
        print("-" * 30)
