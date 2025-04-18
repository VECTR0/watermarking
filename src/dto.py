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
    LPIPS_Loss: float | None


class DecodingMetricsModel(BaseModel):
    Correlation_Coefficient: float | None
    Normalized_Correlation_Coefficient: float | None
    Bit_Error_Rate: float
    Mean_Squared_Error: float


class DtoDecode(BaseModel):
    decoded_watermark: str | None  # TODO change to bytes?
    decoding_time: float | None
    decoding_metrics: DecodingMetricsModel | None


class DtoAttack(BaseModel):
    name: str
    attacking_time: float | None
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
