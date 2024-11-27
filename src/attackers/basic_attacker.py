from src.attacker import Attacker
from src.dto import Dto


class BasicAttacker(Attacker):
    def __init__(self):
        super().__init__()

    def attack(self, dto: Dto) -> Dto:
        print("Applying Noise Attack...")
        dto.attacked_image = "BasicAttackedImage.png"
        return dto