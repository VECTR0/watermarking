from src.dto import Dto
from src.dto import ImageType

AttackingResults = ImageType


class Attacker:
    _all_attacks: list["Attacker"] = []

    def __init__(self) -> None:
        Attacker._all_attacks.append(self)

    @staticmethod
    def get_all() -> list["Attacker"]:
        return Attacker._all_attacks

    def attack(self, dto: Dto) -> AttackingResults:
        raise NotImplementedError

    def get_name(self) -> str:
        return self.__class__.__name__
