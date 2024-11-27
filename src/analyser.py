from .dto import AnalysisResults, Dto


class Analyser:
    _all_analysers: list["Analyser"] = []

    def __init__(self) -> None:
        Analyser._all_analysers.append(self)

    @staticmethod
    def get_all() -> list["Analyser"]:
        return Analyser._all_analysers

    def analyse_watermarked(self, dto: Dto) -> AnalysisResults:
        raise NotImplementedError

    def analyse_attacked(self, dto: Dto) -> AnalysisResults:
        raise NotImplementedError
