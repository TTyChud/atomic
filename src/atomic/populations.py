
import math
from dataclasses import dataclass

from scipy import constants as _sc

from atomic.provenance import Fidelity, Provenance, Quantity

__all__ = [
    "Level",
    "ThermalConditions",
    "ThermalState",
    "boltzmann_fractions",
    "hydrogen_levels",
    "level_column_fraction",
    "level_degeneracy",
    "line_emissivity",
    "partition_function",
    "saha_ionization_fraction",
]

K_EV: float = _sc.physical_constants["Boltzmann constant in eV/K"][0]

_SAHA_COEFF: float = 2.0 * math.pi * _sc.m_e * _sc.k / (_sc.h**2)

_L_SYMBOLS = "spdfghiklmnoqrtuv"

def _l_symbol(l: int) -> str:
    return _L_SYMBOLS[l] if l < len(_L_SYMBOLS) else f"l={l}"

_LTE = (
    "LTE (local thermodynamic equilibrium): one temperature sets both the "
    "level populations and the ionization",
    "optically thin: no radiative transfer, no self-absorption, no escape "
    "probability",
)

@dataclass(frozen=True)
class Level:

    n: int
    label: str
    energy_ev: float
    degeneracy: int

@dataclass(frozen=True)
class ThermalConditions:

    temperature_k: float
    electron_density_cm3: float

@dataclass(frozen=True)
class ThermalState:

    conditions: ThermalConditions
    ionized_fraction: Quantity
    partition_function: Quantity

def level_degeneracy(l: int, j: float | None) -> int:
    return int(round(2.0 * j + 1.0)) if j is not None else 2 * (2 * l + 1)

def hydrogen_levels(n_max: int, fine_structure: bool = False) -> tuple[Level, ...]:
    if n_max < 1:
        raise ValueError(f"n_max must be >= 1, got {n_max}")
    ionization_ev = _sc.physical_constants["Rydberg constant times hc in eV"][0]
    levels: list[Level] = []
    for n in range(1, n_max + 1):
        energy = ionization_ev * (1.0 - 1.0 / (n * n))
        for l in range(n):
            if fine_structure:
                for j in ([0.5] if l == 0 else [l - 0.5, l + 0.5]):
                    levels.append(Level(
                        n=n, label=f"{n}{_l_symbol(l)}{j:g}",
                        energy_ev=energy, degeneracy=level_degeneracy(l, j),
                    ))
            else:
                levels.append(Level(
                    n=n, label=f"{n}{_l_symbol(l)}",
                    energy_ev=energy, degeneracy=level_degeneracy(l, None),
                ))
    return tuple(levels)

def _check_temperature(temperature_k: float) -> None:
    if temperature_k <= 0.0:
        raise ValueError(f"temperature must be > 0 K, got {temperature_k}")

def _truncation_note(levels: tuple[Level, ...]) -> tuple[str, ...]:
    n_max = max(x.n for x in levels)
    return (
        f"the partition function is truncated at n_max={n_max}: the exact sum "
        "diverges, since infinitely many bound states crowd the ionization "
        "limit carrying unbounded statistical weight",
    )

def partition_function(levels: tuple[Level, ...], temperature_k: float) -> Quantity:
    _check_temperature(temperature_k)
    kt = K_EV * temperature_k
    value = sum(x.degeneracy * math.exp(-x.energy_ev / kt) for x in levels)
    return Quantity(
        value=value,
        unit="dimensionless",
        label=f"U(T={temperature_k:g} K)",
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method="summed U = sum_i g_i exp(-E_i / kT) over the supplied levels",
            assumptions=_LTE + _truncation_note(levels),
            refinement=(
                "an occupation-probability treatment (pressure ionization / "
                "lowering of the ionization potential) would supply a physical "
                "cutoff in place of n_max"
            ),
        ),
    )

def boltzmann_fractions(
    levels: tuple[Level, ...], temperature_k: float
) -> tuple[Quantity, ...]:
    _check_temperature(temperature_k)
    kt = K_EV * temperature_k
    weights = [x.degeneracy * math.exp(-x.energy_ev / kt) for x in levels]
    total = sum(weights)
    truncation = _truncation_note(levels)
    return tuple(
        Quantity(
            value=w / total,
            unit="dimensionless",
            label=f"N({x.label})/N_neutral at {temperature_k:g} K",
            provenance=Provenance(
                fidelity=Fidelity.APPROXIMATION,
                method="Boltzmann: N_i/N = g_i exp(-E_i / kT) / U(T)",
                assumptions=_LTE + truncation,
                refinement=(
                    "non-LTE level kinetics (collisional-radiative balance) "
                    "would replace the single temperature"
                ),
            ),
        )
        for x, w in zip(levels, weights, strict=True)
    )

def saha_ionization_fraction(
    temperature_k: float,
    electron_density_cm3: float,
    chi_ev: float,
    u_neutral: float = 2.0,
    u_ion: float = 1.0,
) -> Quantity:
    _check_temperature(temperature_k)
    if electron_density_cm3 <= 0.0:
        raise ValueError(
            f"electron density must be > 0 cm^-3, got {electron_density_cm3}"
        )
    kt = K_EV * temperature_k
    n_e_m3 = electron_density_cm3 * 1e6
    ratio = (
        2.0 * (u_ion / u_neutral)
        * (_SAHA_COEFF * temperature_k) ** 1.5
        * math.exp(-chi_ev / kt)
        / n_e_m3
    )
    return Quantity(
        value=ratio / (1.0 + ratio),
        unit="dimensionless",
        label=f"ionized fraction at {temperature_k:g} K, n_e={electron_density_cm3:g} cm^-3",
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method=(
                "Saha: n_II n_e / n_I = 2 (U_II/U_I) "
                "(2 pi m_e k T / h^2)^(3/2) exp(-chi/kT)"
            ),
            assumptions=_LTE + (
                "the electron density is a caller-set control: it is NOT solved "
                "self-consistently with the ionization it drives",
                f"uses chi = {chi_ev:g} eV, U_ion = {u_ion:g}, "
                f"U_neutral = {u_neutral:g}",
                "ideal gas: no Coulomb interaction between the charged species, "
                "no lowering of the ionization potential",
                "a single ionization stage only",
            ),
            refinement=(
                "solving n_e together with the ionization, and lowering chi for "
                "the plasma environment, would remove both idealizations"
            ),
        ),
    )

def level_column_fraction(
    lower_fraction: float,
    neutral_fraction: float,
) -> Quantity:
    value = neutral_fraction * lower_fraction
    return Quantity(
        value=value,
        unit="dimensionless",
        label="lower-level fraction",
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method="N_l/N_element = (1 - x) (N_l/N_neutral)",
            assumptions=_LTE + (
                f"the neutral fraction 1 - x = {neutral_fraction:.4g} comes "
                "from Saha; a fully ionized gas has no bound electrons left to "
                "absorb",
                "counts per atom of the element (neutral + ionized), so a "
                "column density counts the whole element and this fraction "
                "selects the absorbing level",
            ),
            refinement=(
                "departures from LTE would set the lower level's population "
                "independently of the temperature the profile width uses"
            ),
        ),
    )

def line_emissivity(
    upper_fraction: float,
    neutral_fraction: float,
    einstein_a: float,
    photon_energy_ev: float,
) -> Quantity:
    value = neutral_fraction * upper_fraction * einstein_a * photon_energy_ev
    return Quantity(
        value=value,
        unit="eV/s per atom",
        label="line emissivity",
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method="eps = (1 - x) (N_u/N_neutral) A h nu",
            assumptions=_LTE + (
                f"the neutral fraction 1 - x = {neutral_fraction:.4g} comes "
                "from Saha; a fully ionized gas emits no bound-bound lines at "
                "all",
                "counts per atom of the element (neutral + ionized), so the "
                "denominator is fixed as the gas ionizes",
                "counts spontaneous emission only: no stimulated emission, no "
                "absorption back out of the beam",
            ),
            refinement=(
                "radiative transfer through a finite optical depth would turn "
                "this number into a predicted observed brightness"
            ),
        ),
    )
