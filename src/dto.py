from pydantic import BaseModel

from src.metric import ImageMetricsModel
from src.types import AttackingAnalysisResults, ImageType


class DtoLog(BaseModel):
    filepath: str
    watermarked_analysis_results: ImageMetricsModel
    watermark_method: str
    attack_method: str
    encoding_time: float
    decoding_time: float


class DecodingMetricsModel(BaseModel):
    pass


class DtoDecode(BaseModel):
    decoded_watermark: str | None  # TODO change to bytes?
    decoding_time: float
    decoding_metrics: DecodingMetricsModel | None


class DtoAttack(BaseModel):
    name: str
    attacking_time: float
    decoding_results: DtoDecode
    analysis_results: ImageMetricsModel | None


class DtoWatermark(BaseModel):
    name: str
    encoding_time: float
    decoding_results: DtoDecode
    analysis_results: ImageMetricsModel | None
    attacks: list[DtoAttack] = []


class Dto(BaseModel):
    filepath: str
    watermark: str
    watermarks: list[DtoWatermark] = []
