from .dto import AnalysisResults, Dto
import time


def measure_time(func: callable, *args, **kwargs):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Method '{func.__name__}' took {execution_time:.4f} seconds to execute.")
        return result, execution_time

    return wrapper


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
