import json
from pathlib import Path

from src.config import Logger, logger
from src.dto import Dto, DtoLog


class DtoSaver:
    def __init__(self) -> None:
        self._current_list: list[DtoLog] = []

    def add(self, dto: Dto) -> None:
        dto_log = DtoLog(
            filepath=dto.filepath,
            watermark_method=dto.watermark_method,
            attack_method=dto.attack_method,
            encoding_time=dto.encoding_time,
            decoding_time=dto.decoding_time,
            watermarked_analysis_results=dto.watermarked_analysis_results,
            # TODO: Instead of serializing numpy arrays directly to JSON
            # maybe add metrics for attack analysis data?
        )

        self._current_list.append(dto_log)

    def save_to_file(self, filename: str, *, clear_current_list: bool = False) -> None:
        try:
            with Path(filename).open("w") as fp:
                # TODO: FIXME: now is different json format - is it ok?
                serialized_data = [dto.model_dump_json() for dto in self._current_list]
                json.dump(serialized_data, fp, indent=4)
            logger.log(f"Data saved to {filename}.", level=Logger.INFO)
        except json.JSONDecodeError as e:
            logger.log(f"JSON decoding error: {e}", level=Logger.ERROR)
        except Exception as e:
            logger.log(f"Failed to save data: {e}", level=Logger.ERROR)

        if clear_current_list:
            self._current_list.clear()
