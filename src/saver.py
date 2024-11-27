import json

from src.dto import Dto


class DtoSaver:
    def __init__(self) -> None:
        # Internal list to store extracted data from Dto objects
        self._current_list = []

    def add(self, dto: Dto) -> None:
        """Extracts non-loaded image data from the Dto and adds it to the internal list."""
        self._current_list.append(
            {
                "filepath": dto.filepath,
                "watermarked_analysis_results": dto.watermarked_analysis_results,
                "attacked_analysis_results": dto.attacked_analysis_results,
            }
        )

    def save_to_file(self, filename: str, *, clear_current_list: bool = False) -> None:
        """Saves the internal list to a file in JSON format."""
        try:
            with open(filename, "w") as file:
                json.dump(self._current_list, file, indent=4)
            print(f"Data saved to {filename}.")
        except Exception as e:
            print(f"Failed to save data: {e}")

        # Optionally clear the internal list after saving
        if clear_current_list:
            self._current_list.clear()
