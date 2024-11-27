import numpy as np
import pytest

from src.dto import ImageType
from src.metric import (
    ImageMetrics,
    LPIPSModel,
)

SetupImages = tuple[ImageType, ImageType]
DEVICE: str = "cpu"


@pytest.fixture()
def setup_images() -> SetupImages:
    """
    Fixture to provide example images for testing.
    These are dummy images; in a real scenario, use actual image files.
    """
    # Create a 64x64 dummy image (3 channels, RGB)
    original = np.ones((64, 64, 3), dtype=np.float32) * 100  # A gray image
    watermarked = (
        np.ones((64, 64, 3), dtype=np.float32) * 120
    )  # Slightly different gray image

    # Modify the watermarked image slightly to simulate watermarking
    watermarked[30:34, 30:34] = 200  # Add a small square watermark

    return original, watermarked


def test_psnr(setup_images: SetupImages) -> None:
    original, watermarked = setup_images
    metrics = ImageMetrics(original, watermarked, device=DEVICE, model=LPIPSModel.ALEX)
    psnr_value = metrics.psnr()
    assert psnr_value > 0, f"PSNR should be positive, got {psnr_value}"


# # TODO: FIXME
# def test_ssim(setup_images: SetupImages) -> None:
#     original, watermarked = setup_images
#     metrics = ImageMetrics(original, watermarked, device=DEVICE, model=LPIPSModel.ALEX)
#     ssim_value = metrics.ssim()
#     assert 0 <= ssim_value <= 1, f"SSIM should be between 0 and 1, got {ssim_value}"


def test_lpips_loss(setup_images: SetupImages) -> None:
    original, watermarked = setup_images
    metrics = ImageMetrics(original, watermarked, device=DEVICE, model=LPIPSModel.ALEX)
    lpips_value = metrics.lpips_loss()
    assert lpips_value >= 0, f"LPIPS loss should be non-negative, got {lpips_value}"


def test_bit_error_rate(setup_images: SetupImages) -> None:
    original, watermarked = setup_images
    metrics = ImageMetrics(original, watermarked, device=DEVICE, model=LPIPSModel.ALEX)
    ber_value = metrics.bit_error_rate()
    assert 0 <= ber_value <= 1, f"BER should be between 0 and 1, got {ber_value}"


def test_entropy(setup_images: SetupImages) -> None:
    original, watermarked = setup_images
    metrics = ImageMetrics(original, watermarked, device=DEVICE, model=LPIPSModel.ALEX)
    entropy_value = metrics.entropy()
    assert entropy_value >= 0, f"Entropy should be non-negative, got {entropy_value}"


# def test_correlation_coefficient(setup_images: SetupImages) -> None:
#     original, watermarked = setup_images
#     metrics = ImageMetrics(original, watermarked, device=DEVICE, model=LPIPSModel.ALEX)
#     correlation_value = metrics.correlation_coefficient()
#     # TODO: FIXME: NAN
#     assert (
#         -1 <= correlation_value <= 1
#     ), f"Correlation coefficient should be between -1 and 1, got {correlation_value}"


def test_normalized_correlation_coefficient(setup_images: SetupImages) -> None:
    original, watermarked = setup_images
    metrics = ImageMetrics(original, watermarked, device=DEVICE, model=LPIPSModel.ALEX)
    ncc_value = metrics.normalized_correlation_coefficient()
    assert -1 <= ncc_value <= 1, f"NCC should be between -1 and 1, got {ncc_value}"
