from enum import Enum

import torch
from pydantic.fields import Field
from pydantic_settings import BaseSettings


class Logger(Enum):
    NO = "NO"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"

    def log(self, message: str, *, level: "Logger | None" = None) -> None:
        log_level = self if level is None else level
        match log_level:
            case Logger.NO:
                return
            case Logger.DEBUG | Logger.INFO | Logger.WARN | Logger.ERROR:
                print(message)  # noqa: T201
            case _:
                raise NotImplementedError


class Config(BaseSettings):
    cores: int = Field(4, description="Number of available cores")

    dataset_path: str = Field(..., description="Path to the dataset directory")
    output_path: str = Field("./analysis", description="Path to save results")

    default_watermark: str = Field("test", description="Default watermark string")

    logger: Logger = Field(
        Logger.NO.value, description="Log level: NO, DEBUG, INFO, WARN, ERROR"
    )

    device: str = Field("cpu")

    class Config:
        env_prefix = "APP_"  # Required environment variable prefix in .env
        env_file = ".env"
        # Optional .env file for loading environment variables - will override


config = Config()
device = torch.device(config.device)
logger = config.logger

logger.log(f"Config: {config.model_dump()}", level=Logger.INFO)
