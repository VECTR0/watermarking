import concurrent
import os

from src.attacker import (
    Attacker,
    BlurAttacker,
    CloneStampAttacker,
    JpegCompressionAttacker,
    NoiseAdditionAttacker,
    OverlayAttacker,
    WarpingAttacker,
)
from src.config import Logger, config, logger
from src.dto import Dto
from src.dto_saver import DtoSaver
from src.metric import ImageMetrics
from src.watermarker import Watermarker
from src.watermarkers.dwt_dct_svd_watermarker import DwtDctSvdWatermarker
from src.watermarkers.dwt_dct_watermarker import DwtDctWatermarker
from src.watermarkers.fft_watermarker import FFTWatermarker
from src.watermarkers.lsb_watermarker import LSBWatermarker
from src.watermarkers.riva_gan_watermarker import RivaGanWatermarker


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

    return [
        os.path.join(folder_path, fp)
        for fp in os.listdir(folder_path)
        if fp.lower().endswith(extensions)
    ]


def process_image(
    filepath: str, dto_saver: "DtoSaver", img_metric: ImageMetrics
) -> None:
    src_dto = Dto(filepath)
    src_dto.watermark = (
        config.default_watermark
        # TODO: FIXME 4chars hardcoded (32bits) to ensure rivaGan works
        # Maybe move to global scope, because it sholud not change?
    )

    for watermark in Watermarker.get_all():
        watermark_name = watermark.get_name()
        dto = src_dto.copy()

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

        except Exception as e:
            logger.log(
                f"""Decoded watermark failed for path: {dto.filepath},
                with method {watermark_name}.
                Error: {e!s}""",
                level=Logger.ERROR,
            )
        # metric
        dto.watermarked_analysis_results = img_metric.get_all(
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
                # # decode
                # decoded_attacked_str, decoding_attacked_time = watermark.decode(
                #     attacked_dto.attacked_image
                # )
                attacked_dto.attacked_analysis_results = img_metric.get_all(
                    attacked_dto.source_image, attacked_dto.attacked_image
                )
            except Exception as e:
                logger.log(
                    f"""Decoded attack failed for path: {dto.filepath},
                    with attack method {attack_name}.
                    Error: {e!s}""",
                    level=Logger.ERROR,
                )
            dto_saver.add(attacked_dto)


def split_dataset(
    dataset: list[str], num_chunks: int, *, trim_to: int | None = None
) -> list[list[str]]:
    dataset_len = (
        trim_to if trim_to is not None and trim_to < len(dataset) else len(dataset)
    )
    if dataset_len < num_chunks:
        msg = f"The dataset length: {dataset_len} is smaller than the number of chunks (cores): {num_chunks}."
        raise ValueError(msg)
    dataset = dataset[:dataset_len]
    chunk_size = dataset_len // num_chunks
    return [dataset[i : i + chunk_size] for i in range(0, dataset_len, chunk_size)]


def process_chunk(
    n_core: int,
    image_paths: list[str],
) -> None:
    dto_saver = DtoSaver()
    img_metric = ImageMetrics()

    for fp in image_paths:
        process_image(fp, dto_saver, img_metric)
    dto_saver.save_to_file(
        os.path.join(config.output_path, f"output_no_watermark_core={n_core}.json"),
        clear_current_list=True,
    )


def main() -> None:
    # Watermarkers
    # TODO: fix lsb, fft
    # LSBWatermarker()
    # FFTWatermarker()
    DwtDctWatermarker()
    DwtDctSvdWatermarker()
    RivaGanWatermarker()

    # Attackers
    BlurAttacker(kernel_size=1)
    # BlurAttacker(kernel_size=3)
    # JpegCompressionAttacker(quality=5)
    # JpegCompressionAttacker(quality=50)
    # JpegCompressionAttacker(quality=75)
    # NoiseAdditionAttacker(intesity=0.1)
    # CloneStampAttacker()
    # OverlayAttacker()
    # WarpingAttacker()

    # TODO: from .env
    dataset = get_image_paths(config.dataset_path)

    dataset_chunks = split_dataset(dataset, config.cores, trim_to=12)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        try:
            futures = [
                executor.submit(process_chunk, i, chunk)
                for i, chunk in enumerate(dataset_chunks)
            ]
        except Exception as e:
            logger.log(f"Process failed: {e}", level=Logger.ERROR)

        concurrent.futures.wait(futures)


if __name__ == "__main__":
    main()
