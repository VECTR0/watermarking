import json
import typing

from src.dto import Dto


class DtoSaver:
    def __init__(self) -> None:
        self._current_list: list[dict[str, typing.Any]] = []

    def add(self, dto: Dto) -> None:
        self._current_list.append(
            {
                "filepath": dto.filepath,
                "watermarked_analysis_results": dto.watermarked_analysis_results,
                "attacked_analysis_results": dto.attacked_analysis_results,
                "watermark_method": dto.watermark_method,
                "attack_method": dto.attack_method,  # TODO add attack and method params logging
                "encoding_time": dto.encoding_time,
                "decoding_time": dto.decoding_time,
            }
        )

    def save_to_file(self, filename: str, *, clear_current_list: bool = False) -> None:
        try:
            with open(filename, "w") as file:
                json.dump(self._current_list, file, indent=4)
            print(f"Data saved to {filename}.")
        except Exception as e:
            print(f"Failed to save data: {e}")

        if clear_current_list:
            self._current_list.clear()
