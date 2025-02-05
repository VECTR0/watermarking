import concurrent
import json
import os
from pathlib import Path
import random
import cv2
from tqdm import tqdm

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
from src.dto import Dto, DtoAttack, DtoDecode, DtoWatermark
from src.metric import ImageMetrics, get_decoding_metrics_model
from src.utils import measure_time
from src.watermarker import Watermarker
from src.watermarkers.dwt_dct_svd_watermarker import DwtDctSvdWatermarker
from src.watermarkers.dwt_dct_watermarker import DwtDctWatermarker
from src.watermarkers.fft_watermarker import FFTWatermarker
from src.watermarkers.lsb_watermarker import LSBWatermarker
from src.watermarkers.naive_watermarker import NaiveWatermarker
from src.watermarkers.riva_gan_watermarker import RivaGanWatermarker
from src.watermarkers.vine_watermarker import VineWatermarker


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


def process_image(filepath: str, img_metric: ImageMetrics) -> None:
    source_image = cv2.imread(filepath)
    w, h = source_image.shape[:2]
    logger.log(
        f"Processing image: {w}x{h}px {os.path.basename(filepath)}", level=Logger.DEBUG
    )
    dto = Dto(
        filepath=filepath, watermark=config.default_watermark
    )  # Maybe move to global scope, because it sholud not change?

    for watermark in Watermarker.get_all():
        logger.log(f"  W: {watermark.get_name()}", level=Logger.DEBUG)
        # encode
        watermark_name = watermark.get_name()
        encoded_image, encoding_time = watermark.encode(source_image, dto.watermark)
        # decode
        decoded_watermarked, decoding_watermarked_time = None, None
        decoding_watermarked_analysis_results = None
        try:
            decoded_watermarked, decoding_watermarked_time = watermark.decode(
                encoded_image, source=source_image
            )
            if decoded_watermarked is not None:
                decoding_watermarked_analysis_results = get_decoding_metrics_model(
                    watermark=decoded_watermarked.encode(),
                    decoded=dto.watermark.encode(),
                )
        except Exception as e:
            logger.log(
                f"""Decoded watermark failed for path: {filepath},
                with method {watermark_name}.
                Error: {e!s}""",
                level=Logger.ERROR,
            )
        decoding_watermarked_results = DtoDecode(
            decoded_watermark=decoded_watermarked,
            decoding_time=decoding_watermarked_time,
            decoding_metrics=decoding_watermarked_analysis_results,
        )
        # metric
        watermarked_analysis_results = img_metric.get_all(source_image, encoded_image)

        dtoWatermark = DtoWatermark(
            name=watermark_name,
            encoding_time=encoding_time,
            decoding_results=decoding_watermarked_results,
            analysis_results=watermarked_analysis_results,
        )
        dto.watermarks.append(dtoWatermark)

        for attacker in Attacker.get_all():
            logger.log(f"    A: {attacker.get_name()}", level=Logger.DEBUG)
            # attack
            attack_name = attacker.get_name()
            attacked_image, attacking_time = None, None
            try:
                attacked_image, attacking_time = measure_time(attacker.attack)(
                    encoded_image
                )
            except Exception as e:
                logger.log(
                    f"""Decoded attack failed for path: {filepath},
                    with attack method {attack_name}.
                    Error: {e!s}""",
                    level=Logger.ERROR,
                )

            # decode
            decoded_attacked, decoding_attacked_time = None, None
            decoding_atacked_analysis_results = None
            if attacked_image is not None:
                try:
                    decoded_attacked, decoding_attacked_time = watermark.decode(
                        attacked_image, source=source_image
                    )
                    if decoded_attacked is not None:
                        decoding_atacked_analysis_results = get_decoding_metrics_model(
                            watermark=decoded_attacked.encode(),
                            decoded=dto.watermark.encode(),
                        )
                except Exception as e:
                    logger.log(
                        f"""Decoded watermark failed for path: {filepath},
                        with method {watermark_name}.
                        Error: {e!s}""",
                        level=Logger.ERROR,
                    )
            decoding_results = DtoDecode(
                decoded_watermark=decoded_attacked,
                decoding_time=decoding_attacked_time,
                decoding_metrics=decoding_atacked_analysis_results,
            )
            # analysis
            attacked_analysis_results = None
            if attacked_image is not None:
                attacked_analysis_results = img_metric.get_all(
                    source_image, attacked_image
                )

            dtoAttack = DtoAttack(
                name=attack_name,
                attacking_time=attacking_time,
                decoding_results=decoding_results,
                analysis_results=attacked_analysis_results,
            )
            dtoWatermark.attacks.append(dtoAttack)
    stripped_source_filename = os.path.basename(filepath)
    filename = os.path.join(config.output_path, f"{stripped_source_filename}.json")
    try:
        with Path(filename).open("w") as fp:
            serialized_data = dto.model_dump_json()
            fp.write(serialized_data)
        logger.log(f"Data saved to {filename}.", level=Logger.INFO)
    except json.JSONDecodeError as e:
        logger.log(f"JSON decoding error: {e}", level=Logger.ERROR)
    except Exception as e:
        logger.log(f"Failed to save data: {e}", level=Logger.ERROR)


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
    img_metric = ImageMetrics()

    for fp in tqdm(image_paths):
        process_image(fp, img_metric)


def main() -> None:
    # Watermarkers
    # TODO: fix lsb, fft
    # LSBWatermarker()
    # FFTWatermarker()
    
    VineWatermarker()
    
    DwtDctWatermarker()
    DwtDctSvdWatermarker()
    NaiveWatermarker(amount=0.05, scale=10, watermark_length=4)
    NaiveWatermarker(amount=0.1, scale=8, watermark_length=4)
    NaiveWatermarker(amount=0.2, scale=4, watermark_length=4)
    NaiveWatermarker(amount=0.2, scale=2, watermark_length=4)
    RivaGanWatermarker()

    # Attackers
    BlurAttacker(kernel_size=1)
    BlurAttacker(kernel_size=3)
    BlurAttacker(kernel_size=11)
    BlurAttacker(kernel_size=31)
    JpegCompressionAttacker(quality=5)
    JpegCompressionAttacker(quality=50)
    JpegCompressionAttacker(quality=75)
    NoiseAdditionAttacker(intesity=0.1)
    NoiseAdditionAttacker(intesity=0.5)
    NoiseAdditionAttacker(intesity=0.9)
    # CloneStampAttacker()
    OverlayAttacker()
    # WarpingAttacker()

    # TODO: from .env
    dataset = get_image_paths(config.dataset_path)
    logger.log(f"Found {len(dataset)} images in the dataset.", level=Logger.INFO)
    random.shuffle(dataset)

    dataset_chunks = split_dataset(dataset, config.cores, trim_to=200)

    process_chunk(0, dataset_chunks[0])

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     try:
    #         futures = [
    #             executor.submit(process_chunk, i, chunk)
    #             for i, chunk in enumerate(dataset_chunks)
    #         ]
    #     except Exception as e:
    #         logger.log(f"Process failed: {e}", level=Logger.ERROR)

    #     concurrent.futures.wait(futures)


if __name__ == "__main__":
    main()
