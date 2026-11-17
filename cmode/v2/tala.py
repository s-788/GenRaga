"""Parametric tala. Gene = one akshara-length cell (4 subdivisions), so cycle length = aksharas."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Tala:
    name: str
    aksharas: int
    anga_starts: tuple          # akshara indices where an anga begins (0 = sama)
    notated: bool               # do we have WRITTEN material in this tala? (only Adi)

    @property
    def subs(self): return self.aksharas * 4
    @property
    def onset_subs(self): return tuple(a * 4 for a in self.anga_starts)


TALAS = {
    "adi": Tala("Ādi", 8, (0, 4, 6), True),
    "rupaka": Tala("Rūpaka", 6, (0, 2), False),
    "misra_chapu": Tala("Miśra chāpu", 7, (0, 3, 5), False),
    "khanda_chapu": Tala("Khaṇḍa chāpu", 5, (0, 2), False),
}
