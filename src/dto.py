from pydantic import BaseModel


class ImageMetricsModel(BaseModel):
    PSNR: float
    SSIM_RGB: float
    SSIM_Greyscale: float
    Bit_Error_Rate: float
    Mean_Squared_Error: float
    Entropy: float
    Average_Pixel_Error: float
    QualiCLIP_original: float
    QualiCLIP_watermarked: float
    LPIPS_Loss: float


class DtoLog(BaseModel):
    filepath: str
    watermarked_analysis_results: ImageMetricsModel
    watermark_method: str
    attack_method: str
    encoding_time: float
    decoding_time: float


class DecodingMetricsModel(BaseModel):
    Correlation_Coefficient: float
    Normalized_Correlation_Coefficient: float
    Bit_Error_Rate: float
    Mean_Squared_Error: float


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
