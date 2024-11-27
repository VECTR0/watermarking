from src.dto import Dto
from src.analyser import Analyser
from src.analyser import AnalysisResults

class BasicAnalyser(Analyser):
    def __init__(self):
        super().__init__()

    def analyse_watermarked(self, dto: Dto) -> AnalysisResults:
        print("Analysing watermarked image with Basic Analyser...")
        results = {"MyValue": 123.123}
        return results

    def analyse_attacked(self, dto: Dto) -> AnalysisResults:
        print("Analysing attacked image with Basic Analyser...")
        results = {"MyValue": 0.123}
        return results