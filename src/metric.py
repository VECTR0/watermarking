from enum import Enum
import lpips
import numpy as np
import torch
import torchvision.transforms as transforms
from scipy import stats
from skimage import img_as_float
from skimage.measure import shannon_entropy
from skimage.metrics import structural_similarity as ssim

from src.dto import ImageType

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

model = torch.hub.load(  # TODO fix ftfy
    repo_or_dir="miccunifi/QualiCLIP", source="github", model="QualiCLIP"
)
model.eval().to(device)


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
                raise ValueError("Invalid LPIPS model selected")


class ImageMetrics:
    def __init__(
        self,
        original: ImageType,
        watermarked: ImageType,
        device: str = "cpu",
        model: LPIPSModel = LPIPSModel.ALEX,
    ) -> None:
        """
        Initialize the ImageMetrics object with original and watermarked images.
        :param original: The original image (numpy array).
        :param watermarked: The watermarked image (numpy array).
        :param device: The device to run LPIPS on ('cpu' or 'cuda').
        :param model: The LPIPS model to use ('alex' or 'vgg').
        """
        self.original = original
        self.watermarked = watermarked
        self.device = device
        self.model = model

        self.loss_fn = self.model.get_lpips_model()

    @staticmethod
    def get_all(
        source_image: ImageType, watermarked_img: ImageType
    ) -> dict[str, float]:
        """
        Calculate all metrics and return them as a dictionary.

        :return: Dictionary where keys are metric names and values are calculated metric values.
        """
        assert (
            source_image.shape == watermarked_img.shape
        ), "Images must have the same dimensions."
        metric = ImageMetrics(source_image, watermarked_img)
        metrics = {
            "PSNR": float(metric.psnr()),
            # "SSIM": metric.ssim(),
            # "Correlation Coefficient": metric.correlation_coefficient(),
            "Normalized Correlation Coefficient": float(
                metric.normalized_correlation_coefficient()
            ),
            "Bit Error Rate": float(metric.bit_error_rate()),
            "Mean Squared Error": float(metric.mean_squared_error()),
            "Entropy": float(metric.entropy()),
            "Average Pixel Error": float(metric.average_pixel_error()),
            "QualiCLIP_original": float(metric.quali_clip(original=True)),
            "QualiCLIP_watermarked": float(metric.quali_clip(original=False)),
        }

        # LPIPS loss is optional (requires PyTorch and the appropriate model)
        try:
            metrics["LPIPS Loss"] = metric.lpips_loss()
        except Exception as e:
            metrics["LPIPS Loss"] = float(
                "nan"
            )  # Handle cases where LPIPS calculation fails

        return metrics

    def _to_tensor(self, img: ImageType) -> torch.Tensor:
        """
        Convert a numpy image (H, W, C) to a PyTorch tensor (1, C, H, W) normalized to [-1, 1].
        """
        img = np.moveaxis(img, -1, 0)  # Move channels to the first dimension
        img = torch.tensor(img, dtype=torch.float32) / 255.0  # Normalize to [0, 1]
        img = (img - 0.5) * 2.0  # Normalize to [-1, 1]
        return img.unsqueeze(0)  # Add batch dimension

    def psnr(self) -> float:
        """
        Calculate the Peak Signal-to-Noise Ratio (PSNR).
        """
        mse = np.mean((self.original - self.watermarked) ** 2)
        if mse == 0:
            return float("inf")
        max_pixel = 255.0
        return 20 * np.log10(max_pixel / np.sqrt(mse))

    def ssim(self, win_size: int = 7) -> float:
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

    def correlation_coefficient(self) -> float:
        """
        Calculate the Pearson Correlation Coefficient (SCC).
        """
        original_flat = self.original.flatten()
        watermarked_flat = self.watermarked.flatten()
        return np.corrcoef(original_flat, watermarked_flat)[0, 1]

    def normalized_correlation_coefficient(self) -> float:
        """
        Calculate the Normalized Correlation Coefficient (NCC).
        """
        original_flat = self.original.flatten()
        watermarked_flat = self.watermarked.flatten()
        return np.dot(original_flat, watermarked_flat) / (
            np.linalg.norm(original_flat) * np.linalg.norm(watermarked_flat)
        )

    def bit_error_rate(self) -> float:
        """
        Calculate the Bit Error Rate (BER).
        Assumes binary watermarking (0 or 1).
        """
        Q = 127
        original_binary = (self.original > Q).astype(int)
        watermarked_binary = (self.watermarked > Q).astype(int)
        num_bit_errors = np.sum(original_binary != watermarked_binary)
        total_bits = original_binary.size
        return num_bit_errors / total_bits

    def mean_squared_error(self) -> float:
        """
        Calculate the Mean Squared Error (MSE).
        """
        return np.mean((self.original - self.watermarked) ** 2)

    def entropy(self) -> float:
        """
        Calculate the Shannon Entropy of the image.
        A higher entropy value means more randomness.
        """
        return shannon_entropy(self.watermarked)

    def average_pixel_error(self) -> float:
        """
        Calculate the Average Pixel Error.
        """
        return np.mean(np.abs(self.original - self.watermarked))

    def image_difference(self) -> ImageType:
        """
        Calculate the absolute difference between the original and watermarked image.
        """
        return np.abs(self.original - self.watermarked)

    def lpips_loss(self) -> float:
        """
        Calculate the LPIPS loss between the original and watermarked images.
        """
        # Convert images to torch tensors and normalize to [-1, 1] for LPIPS
        original_tensor = self._to_tensor(self.original)
        watermarked_tensor = self._to_tensor(self.watermarked)

        # Calculate the LPIPS loss
        loss = self.loss_fn(original_tensor, watermarked_tensor)
        return loss.item()

    def quali_clip(self, *, original: bool = True) -> float:
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
        img = preprocess(img_).unsqueeze(0).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            score = model(img).item()
        return score

    def __repr__(self) -> str:
        return f"ImageMetrics(original_shape={self.original.shape}, watermarked_shape={self.watermarked.shape}, model={self.model.name})"


# TODO: add RMSE, CLIP - IQA
