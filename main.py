from src.attacker import Attacker
from src.attackers.basic_attacker import BasicAttacker
from src.dto import Dto
from src.dto_saver import DtoSaver
from src.watermarker import Watermarker
from src.watermarkers.basic_watermarker import BasicWatermarker
from src.metric import ImageMetrics

# Watermarkers
BasicWatermarker()

# Attackers
BasicAttacker()

dataset = ["image1.png", "image2.png"]

dto_saver = DtoSaver()

for i, filepath in enumerate(dataset):
    src_dto = Dto(filepath)

    for watermark in Watermarker.get_all():
        watermark_name = watermark.get_name()
        dto = (
            src_dto.copy()
        )  # TODO maybe reduce cloning source image and use reference from src_dto
        # encode
        dto.watermark_method = watermark_name
        encoded_image, encoding_time = watermark.encode(dto)
        dto.watermarked_image = encoded_image
        dto.watermarking_time = encoding_time
        # decode
        decoded_watermarked_str, decoding_watermarked_time = watermark.decode(dto)
        dto.decoded_watermark = decoded_watermarked_str
        dto.decoding_time = decoding_watermarked_time
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
            decoded_attacked_str, decoding_attacked_time = watermark.decode(
                attacked_dto
            )
            # metric
            attacked_dto.attacked_analysis_results = ImageMetrics.get_all(
                attacked_dto.source_image, attacked_dto.attacked_image
            )

            dto_saver.add(attacked_dto)
            # TODO: maybe modify saver class to decrease dto object metric info duplication

    if (i > 0 and i % 100 == 0) or i == len(dataset) - 1:
        dto_saver.save_to_file(f"output_{i}.json", clear_current_list=True)
