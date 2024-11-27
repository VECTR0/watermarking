from src.attacker import Attacker
from src.attackers.basic_attacker import BasicAttacker
from src.attackers.base_attacker import (
    BlurAttacker,
    CloneStampAttacker,
    JpegCompressionAttacker,
    NoiseAdditionAttacker,
    OverlayAttacker,
    WarpingAttacker,
)
from src.dto import Dto
from src.dto_saver import DtoSaver
from src.watermarker import Watermarker
from src.watermarkers.basic_watermarker import BasicWatermarker
from src.watermarkers.dwt_dct_svd_watermarker import DwtDctSvdWatermarker
from src.watermarkers.dwt_dct_watermarker import DwtDctWatermarker
from src.watermarkers.riva_gan_watermarker import RivaGanWatermarker
from src.metric import ImageMetrics

# Watermarkers
# BasicWatermarker()
DwtDctWatermarker()
DwtDctSvdWatermarker()
# TODO only 32 bit watermarks supported
# RivaGanWatermarker()

# Attackers
# BasicAttacker()
BlurAttacker(kernel_size=1)
BlurAttacker(kernel_size=3)
CloneStampAttacker,
JpegCompressionAttacker(quality=5)
JpegCompressionAttacker(quality=50)
JpegCompressionAttacker(quality=75)
NoiseAdditionAttacker(intesity=0.1)
OverlayAttacker()
WarpingAttacker()

# TODO make proper dataset loading
dataset = ["image1.png"]

dto_saver = DtoSaver()

for i, filepath in enumerate(dataset):
    src_dto = Dto(filepath)
    src_dto.watermark = "test watermark"
    print(f"Processing {filepath}... type: {type(src_dto.source_image)}")

    for watermark in Watermarker.get_all():
        watermark_name = watermark.get_name()
        dto = (
            src_dto.copy()
        )  # TODO maybe reduce cloning source image and use reference from src_dto
        # encode
        dto.watermark_method = watermark_name
        encoded_image, encoding_time = watermark.encode(dto)
        dto.watermarked_image = encoded_image
        dto.encoding_time = encoding_time
        # decode
        try:
            decoded_watermarked_str, decoding_watermarked_time = watermark.decode(
                dto.watermarked_image
            )
            dto.decoded_watermark = decoded_watermarked_str
            dto.decoding_time = decoding_watermarked_time
        except:
            print("decoded failed :-()")

        # metric
        dto.watermarked_analysis_results = ImageMetrics.get_all(
            dto.source_image, dto.watermarked_image
        )

        for attacker in Attacker.get_all():
            attack_name = attacker.get_name()
            attacked_dto = dto.copy()
            # attack
            attacked_dto.attack_method = attack_name
            attacked_image = attacker.attack(dto)
            attacked_dto.attacked_image = attacked_image
            # decode
            try:
                decoded_attacked_str, decoding_attacked_time = (
                    watermark.decode(  # TODO use and save returned values
                        attacked_dto.attacked_image
                    )
                )
            except:
                print("decoded attack failed :-()")

            # metric
            attacked_dto.attacked_analysis_results = ImageMetrics.get_all(
                attacked_dto.source_image, attacked_dto.attacked_image
            )

            dto_saver.add(attacked_dto)
            # TODO: maybe modify saver class to decrease dto object metric info duplication

    if (i > 0 and i % 100 == 0) or i == len(dataset) - 1:
        dto_saver.save_to_file(f"output_{i}.json", clear_current_list=True)
