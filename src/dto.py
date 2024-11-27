AnalysisResults = dict[str,float|str]

class Dto:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.loaded_image = self.load_image()
        self.watermarked_image = None
        self.watermarked_analysis_results: list[AnalysisResults] = []
        self.attacked_image = None
        self.attacked_analysis_results: list[AnalysisResults] = []

    def load_image(self):
        print(f"Loading image from {self.filepath}")
        return f"Image({self.filepath})"

    def save(self):
        print(f"Saving analysis results for {self.filepath}...")

    def add_watermarked_results(self, results: AnalysisResults):
        self.watermarked_analysis_results.append(results)

    def add_attacked_results(self, results: AnalysisResults):
        self.attacked_analysis_results.append(results)
