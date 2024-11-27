import numpy as np

from src.attacker import Attacker
from src.dto import Dto


class BasicAttacker(Attacker):
    def __init__(self) -> None:
        super().__init__()

    def attack(self, dto: Dto) -> Dto:
        print("Applying Noise Attack...")
        dto.attacked_image = np.ones((64, 64, 3), dtype=np.float32) * 100
        return dto
