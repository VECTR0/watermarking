from src.analyser import Analyser
from src.dto import AnalysisResults, Dto


class BasicAnalyser(Analyser):
    def __init__(self) -> None:
        super().__init__()

    def analyse_watermarked(self, dto: Dto) -> AnalysisResults:
        print("Analysing watermarked image with Basic Analyser...")
        return {"MyValue": 123.123}

    def analyse_attacked(self, dto: Dto) -> AnalysisResults:
        print("Analysing attacked image with Basic Analyser...")
        return {"MyValue": 0.123}
