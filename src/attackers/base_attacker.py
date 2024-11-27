import cv2
import numpy as np
from src.attacker import Attacker, AttackingResults
from src.dto import Dto


class BlurAttacker(Attacker):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size

    def attack(self, dto: Dto) -> AttackingResults:
        kernel_size = (
            self.kernel_size * 2 + 1,
            self.kernel_size * 2 + 1,
        )  # Rozmiar jądra rozmycia
        blurred_image = cv2.GaussianBlur(dto.watermarked_image, kernel_size, 0)
        return blurred_image

    def get_name(self) -> str:
        return self.__class__.__name__ + " " + str(self.kernel_size)


class JpegCompressionAttacker(Attacker):
    def __init__(self, quality: int) -> None:
        super().__init__()
        self.quality = quality

    def attack(self, dto: Dto) -> AttackingResults:
        print("Applying JPEG Compression Attack...")
        encode_param = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            self.quality,
        ]
        _, encoded_image = cv2.imencode(".jpg", dto.watermarked_image, encode_param)
        decompressed_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
        return decompressed_image

    def get_name(self) -> str:
        return self.__class__.__name__ + " " + str(self.quality)


class NoiseAdditionAttacker(Attacker):
    def __init__(self, intesity: float) -> None:
        super().__init__()
        self.intesity = intesity

    def attack(self, dto: Dto) -> AttackingResults:
        print("Applying Noise Addition Attack...")
        noise = np.random.normal(0, self.intesity, dto.watermarked_image.shape).astype(
            np.float32
        )
        noisy_image = np.clip(dto.watermarked_image + noise, 0, 255).astype(np.uint8)
        return noisy_image

    def get_name(self) -> str:
        return self.__class__.__name__ + " " + str(self.intesity)


# class CropAttacker(Attacker): #TODO add suport in metric for different image sizes
#     def __init__(self) -> None:
#         super().__init__()

#     def attack(self, dto: Dto) -> AttackingResults:
#         print("Applying Crop Attack...")
#         h, w, _ = dto.watermarked_image.shape
#         cropped_image = dto.watermarked_image[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
#         return cv2.resize(cropped_image, (w, h))  # Skalowanie do pierwotnego rozmiaru


# class ScalingAttacker(Attacker):
#     def __init__(self) -> None:
#         super().__init__()

#     def attack(self, dto: Dto) -> AttackingResults:
#         print("Applying Scaling Attack...")
#         h, w, _ = dto.watermarked_image.shape
#         scaled_image = cv2.resize(dto.watermarked_image, (w // 2, h // 2))  # Skalowanie w dół
#         return cv2.resize(scaled_image, (w, h))  # Przywrócenie rozmiaru


class CloneStampAttacker(Attacker):
    def __init__(self) -> None:
        super().__init__()

    def attack(self, dto: Dto) -> AttackingResults:
        print("Applying Clone Stamp Attack...")
        clone_image = dto.watermarked_image.copy()
        h, w, _ = dto.watermarked_image.shape
        x, y = w // 4, h // 4
        patch = clone_image[y : y + h // 4, x : x + w // 4]
        clone_image[y + h // 4 : y + h // 2, x + w // 4 : x + w // 2] = patch
        return clone_image


class WarpingAttacker(Attacker):
    def __init__(self) -> None:
        super().__init__()

    def attack(self, dto: Dto) -> AttackingResults:
        h, w, _ = dto.watermarked_image.shape
        src_points = np.float32([[0, 0], [w, 0], [0, h]])
        dst_points = np.float32([[0, 0], [w, h // 4], [w // 4, h]])
        matrix = cv2.getAffineTransform(src_points, dst_points)
        warped_image = cv2.warpAffine(dto.watermarked_image, matrix, (w, h))
        return warped_image


# class LabeledSampleAttacker(Attacker):
#     def __init__(self) -> None:
#         super().__init__()

#     def attack(self, dto: Dto) -> AttackingResults:
#         print("Applying Labeled Sample Attack...")
#         # Załóżmy, że mamy dostęp do oznaczonych obrazów w dto.labeled_samples
#         sample = np.random.choice(dto.labeled_samples)  # Losowy wybór obrazu
#         overlayed_image = cv2.addWeighted(dto.watermarked_image, 0.5, sample, 0.5, 0)
#         return overlayed_image


class OverlayAttacker(Attacker):
    def __init__(self) -> None:
        super().__init__()

    def attack(self, dto: Dto) -> AttackingResults:
        print("Applying Overlay Attack...")
        overlay = np.zeros_like(dto.watermarked_image)
        h, w, _ = dto.watermarked_image.shape
        cv2.rectangle(
            overlay, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (255, 0, 0), -1
        )
        overlayed_image = cv2.addWeighted(dto.watermarked_image, 0.7, overlay, 0.3, 0)
        return overlayed_image
