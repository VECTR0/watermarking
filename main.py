import os

from src.attacker import Attacker
from src.attackers.base_attacker import (
    BlurAttacker,
    CloneStampAttacker,
    JpegCompressionAttacker,
    NoiseAdditionAttacker,
    OverlayAttacker,
    WarpingAttacker,
)
from src.attackers.basic_attacker import BasicAttacker
from src.dto import Dto
from src.dto_saver import DtoSaver
from src.metric import ImageMetrics
from src.utils import log
from src.watermarker import Watermarker
from src.watermarkers.basic_watermarker import BasicWatermarker
from src.watermarkers.dwt_dct_svd_watermarker import DwtDctSvdWatermarker
from src.watermarkers.dwt_dct_watermarker import DwtDctWatermarker
from src.watermarkers.fft_watermarker import FFTWatermarker
from src.watermarkers.lsb_watermarker import LSBWatermarker
from src.watermarkers.riva_gan_watermarker import RivaGanWatermarker

# Watermarkers
# BasicWatermarker()
# simple
LSBWatermarker()
FFTWatermarker()
# more complex
DwtDctWatermarker()
DwtDctSvdWatermarker()
# TODO only 32 bit watermarks supported
RivaGanWatermarker()

# Attackers
# BasicAttacker()
BlurAttacker(kernel_size=1)
BlurAttacker(kernel_size=3)
JpegCompressionAttacker(quality=5)
JpegCompressionAttacker(quality=50)
JpegCompressionAttacker(quality=75)
NoiseAdditionAttacker(intesity=0.1)
CloneStampAttacker()
OverlayAttacker()
WarpingAttacker()


def get_image_paths(
    folder_path: str, extensions: tuple[str, str, str] = ("png", "jpg", "jpeg")
) -> list[str]:
    """
    Get a list of image paths from a folder.

    Parameters:
        folder_path (str): Path to the folder containing images.
        extensions (tuple): File extensions to include (default: png, jpg, jpeg).

    Returns:
        list: List of image file paths.
    """
    image_paths = []

    # Normalize the folder path and iterate through files in the directory

    for file in os.listdir(folder_path):
        if file.lower().endswith(extensions):
            image_paths.append(os.path.join(folder_path, file))

    return image_paths


# TODO make proper dataset loading
# dataset = ["image1.png"]
dataset = get_image_paths(
    "/home/adam/.cache/kagglehub/datasets/felicepollano/watermarked-not-watermarked-images/versions/1/wm-nowm/train/watermark"
    # "~/.cache/kagglehub/datasets/felicepollano/watermarked-not-watermarked-images/versions/1/wm-nowm/train/no-watermark"
)

dto_saver = DtoSaver()

for i, filepath in enumerate(dataset):
    src_dto = Dto(filepath)
    src_dto.watermark = (
        "test"  # TODO FIXME 4chars hardcoded (32bits) to ensure rivaGan works
    )
    log(f"Processing {filepath}... type: {type(src_dto.source_image)}")

    for watermark in Watermarker.get_all():
        watermark_name = watermark.get_name()
        dto = (
            src_dto.copy()
        )  # TODO maybe reduce cloning source image and use reference from src_dto
        # encode
        dto.watermark_method = watermark_name
        encoded_image, encoding_time = watermark.encode(dto)
        dto.watermarked_image = encoded_image
        dto.encoding_time = encoding_time
        # decode
        try:
            decoded_watermarked_str, decoding_watermarked_time = watermark.decode(
                dto.watermarked_image
            )
            dto.decoded_watermark = decoded_watermarked_str
            dto.decoding_time = decoding_watermarked_time
        except:
            log("decoded failed :-()")

        # metric
        dto.watermarked_analysis_results = ImageMetrics.get_all(
            dto.source_image, dto.watermarked_image
        )

        for attacker in Attacker.get_all():
            attack_name = attacker.get_name()
            attacked_dto = dto.copy()
            # attack
            attacked_dto.attack_method = attack_name
            try:
                attacked_image = attacker.attack(dto)
                attacked_dto.attacked_image = attacked_image
                # decode
                decoded_attacked_str, decoding_attacked_time = (
                    watermark.decode(  # TODO use and save returned values
                        attacked_dto.attacked_image
                    )
                )
                # metric
                attacked_dto.attacked_analysis_results = ImageMetrics.get_all(
                    attacked_dto.source_image, attacked_dto.attacked_image
                )
            except:
                # handle attack errror and decode error
                log("decoded attack failed :-()")

            dto_saver.add(attacked_dto)
            # TODO: maybe modify saver class to decrease dto object metric info duplication

    if (i > 0 and i % 1 == 0) or i == len(dataset) - 1:
        dto_saver.save_to_file(f"output_{i}.json", clear_current_list=True)
