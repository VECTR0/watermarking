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
