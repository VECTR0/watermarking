import numpy as np

from src.attacker import Attacker, AttackingResults
from src.dto import Dto


class BasicAttacker(Attacker):
    def __init__(self) -> None:
        super().__init__()

    def attack(self, dto: Dto) -> AttackingResults:
        print("Applying Noise Attack...")
        return np.ones((64, 64, 3), dtype=np.float32) * 100

    def get_name(self) -> str:
        return self.__class__.__name__
