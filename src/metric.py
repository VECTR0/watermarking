from enum import Enum

import cv2
import lpips
import numpy as np
import torch
from pydantic import BaseModel
from skimage.measure import shannon_entropy
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms

from src.config import Logger, config, logger
from src.dto import DecodingMetricsModel, ImageMetricsModel
from src.types import ImageType


class LPIPSModel(Enum):
    ALEX = "alex"
    VGG = "vgg"

    def get_lpips_model(self, device: str = "cpu") -> lpips.LPIPS:
        """
        Get the LPIPS model based on the enum value.
        :param device: The device to load the model on ('cpu' or 'cuda').
        :return: The LPIPS model.
        """
        match self:
            case LPIPSModel.ALEX:
                return lpips.LPIPS(net="alex").to(device)
            case LPIPSModel.VGG:
                return lpips.LPIPS(net="vgg").to(device)
            case _:
                msg = "Invalid LPIPS model selected"
                raise ValueError(msg)


def get_decoding_metrics_model(
    watermark: bytes, decoded: bytes
) -> DecodingMetricsModel:
    return DecodingMetricsModel(
        Correlation_Coefficient=correlation_coefficient(watermark, decoded),
        Normalized_Correlation_Coefficient=normalized_correlation_coefficient(
            watermark, decoded
        ),
        Bit_Error_Rate=bit_error_rate(watermark, decoded),
        Mean_Squared_Error=mean_squared_error(watermark, decoded),
    )


def correlation_coefficient(a: bytes, b: bytes) -> float:
    """
    Calculate the Pearson Correlation Coefficient (SCC).
    """
    original_flat = np.frombuffer(a, dtype=np.uint8)
    watermarked_flat = np.frombuffer(b, dtype=np.uint8)
    if np.std(original_flat) == 0 or np.std(watermarked_flat) == 0:
        return float("inf")
    correlation_matrix = np.corrcoef(original_flat, watermarked_flat)
    return correlation_matrix[0, 1]


def normalized_correlation_coefficient(a: bytes, b: bytes) -> float:
    """
    Calculate the Normalized Correlation Coefficient (NCC).
    """
    original_flat = np.frombuffer(a, dtype=np.uint8)
    watermarked_flat = np.frombuffer(b, dtype=np.uint8)
    div = np.linalg.norm(original_flat) * np.linalg.norm(watermarked_flat)
    if div == 0:
        return float("inf")
    return np.dot(original_flat, watermarked_flat) / div


def bit_error_rate(a: bytes, b: bytes) -> float:
    """
    Calculate the Bit Error Rate (BER).
    This function assumes that both `a` and `b` are byte sequences that represent binary data (0 or 1) without any thresholding.
    """
    if len(a) != len(b):
        raise ValueError("Input byte arrays must have the same length.")

    original_binary = np.unpackbits(np.frombuffer(a, dtype=np.uint8))
    watermarked_binary = np.unpackbits(np.frombuffer(b, dtype=np.uint8))

    num_bit_errors = np.sum(original_binary != watermarked_binary)
    total_bits = original_binary.size

    return num_bit_errors / total_bits


def mean_squared_error(a: bytes, b: bytes) -> float:
    original_binary = np.unpackbits(np.frombuffer(a, dtype=np.uint8))
    watermarked_binary = np.unpackbits(np.frombuffer(b, dtype=np.uint8))
    return np.mean((original_binary - watermarked_binary) ** 2)


class ImageMetrics:
    def __init__(self) -> None:
        try:
            model_quali = torch.hub.load(
                repo_or_dir="miccunifi/QualiCLIP", source="github", model="QualiCLIP"
            )
            self.model_quali = model_quali.eval().to(config.device)
            self.lpips_loss_fn = LPIPSModel.ALEX.get_lpips_model(config.device)
        except Exception as e:
            # TODO: FIXME: Handle CUDA multi-processing (spawn instead of fork)
            logger.log(f"Cannot initialize models: {e}", level=Logger.ERROR)

    def get_all(self, original: ImageType, watermarked: ImageType) -> ImageMetricsModel:
        """
        Calculate all metrics and return them as a dictionary.

        :return: Dictionary where keys are metric names and values are calculated metric values.
        """
        assert (
            original.shape == watermarked.shape
        ), "Images must have the same dimensions."
        self.original = original
        self.watermarked = watermarked

        # TODO: remove unnecessary float casting
        return ImageMetricsModel(
            PSNR=float(self.__psnr()),
            SSIM_RGB=float(self.__ssim_per_channel()),
            SSIM_Greyscale=float(self.__ssim_greyscale()),
            Bit_Error_Rate=float(self.__bit_error_rate()),
            Mean_Squared_Error=float(self.__mean_squared_error()),
            Entropy=float(self.__entropy()),
            Average_Pixel_Error=float(self.__average_pixel_error()),
            # QualiCLIP_original=float("nan"),  # TODO fix ML models performance
            QualiCLIP_original=float(self.__quali_clip(original=True)),
            # QualiCLIP_watermarked=float("nan"),
            QualiCLIP_watermarked=float(self.__quali_clip(original=False)),
            # LPIPS_Loss=float("nan"),
            LPIPS_Loss=self.__lpips_loss(),
        )

    def _to_tensor(self, img: ImageType) -> torch.Tensor:
        """
        Convert a numpy image (H, W, C) to a PyTorch tensor (1, C, H, W) normalized to [-1, 1].
        """
        img = np.moveaxis(img, -1, 0)  # Move channels to the first dimension
        img = torch.tensor(img, dtype=torch.float32) / 255.0  # Normalize to [0, 1]
        img = (img - 0.5) * 2.0  # Normalize to [-1, 1]
        return img.unsqueeze(0)  # Add batch dimension

    def __psnr(self) -> float:
        """
        Calculate the Peak Signal-to-Noise Ratio (PSNR).
        """
        mse = np.mean((self.original - self.watermarked) ** 2)
        if mse == 0:
            return float("inf")
        max_pixel = 255.0
        return 20 * np.log10(max_pixel / np.sqrt(mse))

    def __ssim_per_channel(self) -> float:
        """for watermark robustness"""
        r = ssim(self.original[..., 0], self.watermarked[..., 0], data_range=255)
        g = ssim(self.original[..., 1], self.watermarked[..., 1], data_range=255)
        b = ssim(self.original[..., 2], self.watermarked[..., 2], data_range=255)
        return (r + g + b) / 3

    def __ssim_greyscale(self) -> float:
        """
        for watermark invisibility
        """
        gray1 = cv2.cvtColor(self.original, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(self.watermarked, cv2.COLOR_RGB2GRAY)

        return ssim(gray1, gray2, data_range=255)

    def __bit_error_rate(self) -> float:
        """
        Calculate the Bit Error Rate (BER).
        Assumes binary watermarking (0 or 1).
        """
        q = 127
        original_binary = (self.original > q).astype(int)
        watermarked_binary = (self.watermarked > q).astype(int)
        num_bit_errors = np.sum(original_binary != watermarked_binary)
        total_bits = original_binary.size
        return num_bit_errors / total_bits

    def __mean_squared_error(self) -> float:
        """
        Calculate the Mean Squared Error (MSE).
        """
        return np.mean((self.original - self.watermarked) ** 2)

    def __entropy(self) -> float:
        """
        Calculate the Shannon Entropy of the image.
        A higher entropy value means more randomness.
        """
        return shannon_entropy(self.watermarked)

    def __average_pixel_error(self) -> float:
        """
        Calculate the Average Pixel Error.
        """
        return np.mean(np.abs(self.original - self.watermarked))

    def __image_difference(self) -> ImageType:
        """
        Calculate the absolute difference between the original and watermarked image.
        """
        return np.abs(self.original - self.watermarked)

    def __lpips_loss(self) -> float:
        """
        Calculate the LPIPS loss between the original and watermarked images.
        """
        try:
            # Convert images to tensors
            original_tensor = self._to_tensor(self.original)
            watermarked_tensor = self._to_tensor(self.watermarked)

            # Calculate the LPIPS loss
            loss = self.lpips_loss_fn(original_tensor, watermarked_tensor)
            return loss.item()

        except Exception:
            return float("nan")

    def __quali_clip(self, *, original: bool = True) -> float:
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        img_ = self.original if original else self.watermarked
        img = preprocess(img_).unsqueeze(0).to(config.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            return self.model_quali(img).item()

    def __repr__(self) -> str:
        return f"""ImageMetrics(
        original_shape={self.original.shape},
        watermarked_shape={self.watermarked.shape},
        model_quali={self.model_quali.name}
    )"""
