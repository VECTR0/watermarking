from src.analyser import Analyser
from src.dto import AnalysisResults, Dto, ImageType
from src.watermarker_invisible import WatermarkInvisibleMethod


class DwtDctAnalyser(Analyser):
    def __init__(self) -> None:
        super().__init__()

    def analyse_watermarked(self, dto: Dto) -> AnalysisResults:
        print("Analysing watermarked image with Basic Analyser...")
        watermarked_image, watermark = dto.watermarked_image, dto.watermark
        assert isinstance(watermarked_image, ImageType), "sorry bro"

        decoded_watermark = WatermarkInvisibleMethod.DWT_DCT.decode(watermarked_image)

        new_dto = dto.copy()
        new_dto.decoded_watermark = decoded_watermark

        return {"MyValue": 123.123}

    def analyse_attacked(self, dto: Dto) -> AnalysisResults:
        print("Analysing attacked image with Basic Analyser...")
        return {"MyValue": 0.123}
