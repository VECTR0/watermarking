from src.analyser import Analyser
from src.analysers.basic_analyser import BasicAnalyser
from src.attacker import Attacker
from src.attackers.basic_attacker import BasicAttacker
from src.dto import Dto
from src.dto_saver import DtoSaver
from src.watermarker import Watermarker
from src.watermarkers.basic_watermarker import BasicWatermarker

# Watermarkers
BasicWatermarker()

# Analysers
BasicAnalyser()

# Attackers
BasicAttacker()

dataset = ["image1.png", "image2.png"]

dto_saver = DtoSaver()

for i, filepath in enumerate(dataset):
    src_dto = Dto(filepath)

    for watermark in Watermarker.get_all():
        dto = watermark.watermark(src_dto)

        for analyser in Analyser.get_all():
            watermarked_results = analyser.analyse_watermarked(dto)
            dto.add_watermarked_results(watermarked_results)

        for attacker in Attacker.get_all():
            attacked_dto = attacker.attack(dto)

            for analyser in Analyser.get_all():
                attacked_results = analyser.analyse_attacked(attacked_dto)
                attacked_dto.add_attacked_results(attacked_results)

            dto_saver.add(attacked_dto)

    if (i > 0 and i % 100 == 0) or i == len(dataset) - 1:
        dto_saver.save_to_file(f"output_{i}.json", clear_current_list=True)
