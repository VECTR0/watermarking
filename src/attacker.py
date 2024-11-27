from src.dto import Dto


class Attacker:
    _all_attacks: list['Attacker'] = []

    def __init__(self):
        Attacker._all_attacks.append(self)

    @staticmethod
    def get_all() -> list['Attacker']:
        return Attacker._all_attacks

    def attack(self, dto: Dto) -> Dto:
        raise NotImplementedError()