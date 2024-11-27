from src.dto import Dto
from src.watermarker import Watermarker

class BasicWatermarker(Watermarker):
    def __init__(self):
        super().__init__()

    def watermark(self, dto: Dto) -> Dto:
        print("Applying Basic Watermark...")
        dto.watermarked_image = "BasicWatermarkedImage"
        return dto