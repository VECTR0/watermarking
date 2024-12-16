import concurrent
import json
import os
from pathlib import Path
import cv2

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


def process_image(filepath: str, img_metric: ImageMetrics) -> None:
    source_image = cv2.imread(filepath)
    w, h = source_image.shape[:2]
    print(f"Processing image: {w}x{h}px {os.path.basename(filepath)}")
    dto = Dto(
        filepath=filepath, watermark=config.default_watermark
    )  # Maybe move to global scope, because it sholud not change?

    for watermark in Watermarker.get_all():
        print(f"  W: {watermark.get_name()}")
        # encode
        watermark_name = watermark.get_name()
        encoded_image, encoding_time = watermark.encode(source_image, dto.watermark)
        # decode
        decoded_watermarked, decoding_watermarked_time = None, float("nan")
        try:
            decoded_watermarked, decoding_watermarked_time = watermark.decode(
                encoded_image
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
            decoding_metrics=None,
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
            print(f"    A: {attacker.get_name()}")
            # attack
            attack_name = attacker.get_name()
            attacked_image, attacking_time = None, float("nan")
            try:
                attacked_image = attacker.attack(encoded_image)  # TODO add time?
                attacking_time = float("nan")  # TODO add time?
            except Exception as e:
                logger.log(
                    f"""Decoded attack failed for path: {filepath},
                    with attack method {attack_name}.
                    Error: {e!s}""",
                    level=Logger.ERROR,
                )

            # decode
            decoded_attacked, decoding_attacked_time = None, float("nan")
            if attacked_image is not None:
                try:
                    decoded_attacked, decoding_attacked_time = watermark.decode(
                        attacked_image
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
                decoding_metrics=None,
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
            # TODO: FIXME: now is different json format - is it ok?
            serialized_data = dto.model_dump_json()
            print(serialized_data)
            print(f"   Saving to: {filename}")
            # json.dump(serialized_data, fp, indent=4)
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

    for fp in image_paths:
        process_image(fp, img_metric)


def main() -> None:
    print("Starting...")
    # Watermarkers
    # TODO: fix lsb, fft
    # LSBWatermarker()
    # FFTWatermarker()
    DwtDctWatermarker()
    DwtDctSvdWatermarker()
    # RivaGanWatermarker()

    # Attackers
    # BlurAttacker(kernel_size=1)
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
    print(f"Found {len(dataset)} images in the dataset.")

    dataset_chunks = split_dataset(dataset, config.cores, trim_to=1)
    print(f"Split dataset into {len(dataset_chunks)} chunks.")

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
