import math
from dataclasses import dataclass

from scipy import constants as _sc

HARTREE_EV: float = _sc.physical_constants["Hartree energy in eV"][0]

ALPHA: float = _sc.fine_structure

BOHR_RADIUS_PM: float = _sc.physical_constants["Bohr radius"][0] * 1e12

BOHR_RADIUS_FM: float = _sc.physical_constants["Bohr radius"][0] * 1e15

B0_TESLA: float = _sc.physical_constants["atomic unit of mag. flux density"][0]

E0_V_PER_M: float = _sc.physical_constants["atomic unit of electric field"][0]

@dataclass(frozen=True)
class FundamentalConstants:
    hbar: float
    e: float
    m_e: float
    eps0: float
    c: float

    def __post_init__(self) -> None:
        for name in ("hbar", "e", "m_e", "eps0", "c"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a real number, got {value!r}")
            v = float(value)
            if not math.isfinite(v) or v <= 0.0:
                raise ValueError(
                    f"{name} must be finite and positive, got {value!r}"
                )
            object.__setattr__(self, name, v)

    @classmethod
    def codata(cls) -> "FundamentalConstants":
        return cls(
            hbar=_sc.hbar,
            e=_sc.elementary_charge,
            m_e=_sc.electron_mass,
            eps0=_sc.epsilon_0,
            c=_sc.speed_of_light,
        )

    @property
    def alpha(self) -> float:
        return self.e**2 / (4 * math.pi * self.eps0 * self.hbar * self.c)

    @property
    def bohr_radius(self) -> float:
        return 4 * math.pi * self.eps0 * self.hbar**2 / (self.m_e * self.e**2)

    @property
    def hartree_energy(self) -> float:
        return self.hbar**2 / (self.m_e * self.bohr_radius**2)

    @property
    def hartree_ev(self) -> float:
        return self.hartree_energy / self.e

    @property
    def bohr_pm(self) -> float:
        return self.bohr_radius * 1e12

    @property
    def bohr_fm(self) -> float:
        return self.bohr_radius * 1e15

    @property
    def b0_tesla(self) -> float:
        return self.hbar / (self.e * self.bohr_radius**2)

    @property
    def e0_v_per_m(self) -> float:
        return self.hartree_energy / (self.e * self.bohr_radius)
