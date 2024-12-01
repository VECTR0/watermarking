from enum import Enum

import lpips
import numpy as np
import torch
from pydantic import BaseModel
from skimage.measure import shannon_entropy
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms

from src.config import Logger, config, logger
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


class ImageMetricsModel(BaseModel):
    PSNR: float
    SSIM: float
    Correlation_Coefficient: float
    Normalized_Correlation_Coefficient: float
    Bit_Error_Rate: float
    Mean_Squared_Error: float
    Entropy: float
    Average_Pixel_Error: float
    QualiCLIP_original: float
    QualiCLIP_watermarked: float
    LPIPS_Loss: float


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
            SSIM=float(self.__ssim()),
            Correlation_Coefficient=float(self.__correlation_coefficient()),
            Normalized_Correlation_Coefficient=float(
                self.__normalized_correlation_coefficient()
            ),
            Bit_Error_Rate=float(self.__bit_error_rate()),
            Mean_Squared_Error=float(self.__mean_squared_error()),
            Entropy=float(self.__entropy()),
            Average_Pixel_Error=float(self.__average_pixel_error()),
            QualiCLIP_original=float(self.__quali_clip(original=True)),
            QualiCLIP_watermarked=float(self.__quali_clip(original=False)),
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

    def __ssim(self, win_size: int = 7) -> float:
        # TODO: fix function
        return float("nan")
        """
        Calculate the Structural Similarity Index (SSIM) between the original and watermarked images.

        :param win_size: The window size used for SSIM computation (must be odd and <= image dimensions).
        :return: SSIM score (range: -1 to 1), where 1 indicates perfect similarity.
        """
        # Ensure images are in the correct range [0, 255] for SSIM
        if self.original.max() <= 1:
            self.original = (self.original * 255).astype(np.uint8)
        if self.watermarked.max() <= 1:
            self.watermarked = (self.watermarked * 255).astype(np.uint8)

        # Ensure images are large enough to accommodate the window size
        if min(self.original.shape[:2]) < win_size:
            msg = f"Image dimensions are too small for the specified win_size ({win_size})."
            raise ValueError(msg)

        # Calculate and return SSIM with specified window size
        return ssim(
            self.original, self.watermarked, multichannel=True, win_size=win_size
        )

    def __correlation_coefficient(self) -> float:
        # TODO: fix function
        return float("nan")
        """
        Calculate the Pearson Correlation Coefficient (SCC).
        """
        original_flat = self.original.flatten()
        watermarked_flat = self.watermarked.flatten()
        return np.corrcoef(original_flat, watermarked_flat)[0, 1]

    def __normalized_correlation_coefficient(self) -> float:
        """
        Calculate the Normalized Correlation Coefficient (NCC).
        """
        original_flat = self.original.flatten()
        watermarked_flat = self.watermarked.flatten()
        return np.dot(original_flat, watermarked_flat) / (
            np.linalg.norm(original_flat) * np.linalg.norm(watermarked_flat)
        )

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
