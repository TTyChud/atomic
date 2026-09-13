from dataclasses import dataclass

from scipy import constants as _sc

from atomic.provenance import Fidelity, Provenance, Quantity


@dataclass(frozen=True)
class System:
    key: str
    name: str
    Z: int
    mu_ratio: Quantity
    m_over_M: float
    description: str

    nuclear_radius: Quantity | None = None

def _mass_ratio(constant_name: str) -> tuple[float, float]:
    value, _unit, unc = _sc.physical_constants[constant_name]
    return value, unc

_A0_M = _sc.physical_constants["Bohr radius"][0]

def _radius_quantity(
    r_m: float, unc_m: float, nucleus: str, source: str
) -> Quantity:
    return Quantity(
        value=r_m / _A0_M,
        unit="bohr",
        label=f"nuclear rms charge radius ({nucleus})",
        provenance=Provenance(
            fidelity=Fidelity.EXACT,
            method=f"the measured rms charge radius of the {nucleus}, {source}",
            assumptions=(
                "this is a reference measurement, not a prediction",
                "the Coulomb engine still treats the nucleus as a point charge",
            ),
            error_estimate=unc_m / _A0_M,
        ),
    )

def _codata_radius(constant_name: str, nucleus: str) -> Quantity:
    r_m, _unit, unc_m = _sc.physical_constants[constant_name]
    return _radius_quantity(
        r_m, unc_m, nucleus, f"CODATA (scipy.constants: {constant_name})"
    )

_TRITON_RADIUS = _radius_quantity(
    1.7591e-15, 0.0363e-15, "triton",
    "Angeli & Marinova (2013), At. Data Nucl. Data Tables 99, 69",
)

def _codata_system(
    key: str, name: str, Z: int, nucleus_constant: str, description: str,
    orbiter_constant: str | None = None,
    nuclear_radius: Quantity | None = None,
) -> System:
    R_nuc, u_nuc = _mass_ratio(nucleus_constant)
    if orbiter_constant is None:
        r_orb, u_orb, orb_name = 1.0, 0.0, "electron"
    else:
        r_orb, u_orb = _mass_ratio(orbiter_constant)
        orb_name = orbiter_constant.split("-")[0]
    mu = r_orb * R_nuc / (r_orb + R_nuc)
    rel_unc = (u_nuc / R_nuc if R_nuc else 0.0) + (u_orb / r_orb if r_orb else 0.0)
    return System(
        key=key,
        name=name,
        Z=Z,
        mu_ratio=Quantity(
            value=mu,
            unit="m_e",
            label=f"mu/m_e ({name})",
            provenance=Provenance(
                fidelity=Fidelity.EXACT,
                method=(
                    "mu/m_e = m_orb M / (m_orb + M) from CODATA mass "
                    f"ratios (scipy.constants: {nucleus_constant}"
                    + (f", {orbiter_constant})" if orbiter_constant else ")")
                ),
                assumptions=(f"a {orb_name} is the orbiting particle here",),
                error_estimate=mu * rel_unc,
            ),
        ),
        m_over_M=r_orb / R_nuc,
        description=description,
        nuclear_radius=nuclear_radius,
    )

_POSITRONIUM = System(
    key="ps",
    name="Positronium",
    Z=1,
    mu_ratio=Quantity(
        value=0.5,
        unit="m_e",
        label="mu/m_e (Positronium)",
        provenance=Provenance(
            fidelity=Fidelity.EXACT,
            method="mu = m_e/2 exactly, because the electron and positron weigh the same",
            assumptions=("an electron orbits here, and the 'nucleus' is a positron",),
            error_estimate=0.0,
        ),
    ),
    m_over_M=1.0,
    description="Electron bound to a positron; recoil is O(1), fine structure "
    "unreliable at alpha^2 (error estimate says so).",
)

_SYSTEMS: tuple[System, ...] = (
    _codata_system("h", "Hydrogen", 1, "proton-electron mass ratio",
                   "Ordinary hydrogen: electron + proton.",
                   nuclear_radius=_codata_radius("proton rms charge radius", "proton")),
    _codata_system("d", "Deuterium", 1, "deuteron-electron mass ratio",
                   "Heavy hydrogen: electron + deuteron.",
                   nuclear_radius=_codata_radius("deuteron rms charge radius", "deuteron")),
    _codata_system("t", "Tritium", 1, "triton-electron mass ratio",
                   "Radioactive hydrogen isotope: electron + triton.",
                   nuclear_radius=_TRITON_RADIUS),
    _codata_system("mu-h", "Muonic hydrogen", 1, "proton-electron mass ratio",
                   "Muon orbiting a proton: ~186x smaller, ~186x deeper.",
                   orbiter_constant="muon-electron mass ratio",
                   nuclear_radius=_codata_radius("proton rms charge radius", "proton")),
    _POSITRONIUM,
    _codata_system("he+", "Helium ion He+", 2, "alpha particle-electron mass ratio",
                   "One-electron helium: Z=2 scaling on real helium-4.",
                   nuclear_radius=_codata_radius(
                       "alpha particle rms charge radius", "alpha particle")),
)

def list_systems() -> tuple[System, ...]:
    return _SYSTEMS

def get_system(key: str) -> System:
    for s in _SYSTEMS:
        if s.key == key:
            return s
    raise KeyError(f"unknown system {key!r}; available: {[s.key for s in _SYSTEMS]}")

def emitter_mass(system: System) -> Quantity:
    x = system.m_over_M
    mu = system.mu_ratio.value
    if x <= 0.0:
        return Quantity(
            value=float("inf"),
            unit="kg",
            label=f"M_atom ({system.key})",
            provenance=Provenance(
                fidelity=Fidelity.APPROXIMATION,
                method="uses an infinite nuclear mass, so the nucleus cannot recoil",
                assumptions=(
                    "this preset carries no nuclear mass (m_over_M = 0), so the "
                    "atom is infinitely heavy and every thermal velocity comes "
                    "out zero; a real ion of this Z has a finite mass and a "
                    "finite Doppler width",
                ),
                refinement="supply a nuclear mass to get a real thermal width",
            ),
        )
    ratio = mu * (1.0 + x) ** 2 / x
    return Quantity(
        value=ratio * _sc.m_e,
        unit="kg",
        label=f"M_atom ({system.key})",
        provenance=Provenance(
            fidelity=system.mu_ratio.provenance.fidelity,
            method=(
                "inverts M_atom = mu (1 + x)^2 / x with x = m_orb/M_nuc, out of "
                f"[{system.mu_ratio.provenance.method}]"
            ),
            assumptions=system.mu_ratio.provenance.assumptions
            + ("the mass of the bound system is taken as the sum of its parts: "
               "the binding energy is ~1e-8 of the rest mass, so it is dropped",),
            error_estimate=(
                None if system.mu_ratio.provenance.error_estimate is None
                else system.mu_ratio.provenance.error_estimate
                * (1.0 + x) ** 2 / x * _sc.m_e
            ),
        ),
    )

def element_emitter_mass(element) -> Quantity:
    return Quantity(
        value=element.mass_u * _sc.physical_constants["atomic mass constant"][0],
        unit="kg",
        label=f"M_atom ({element.symbol})",
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method=(
                f"takes the standard atomic weight {element.mass_u:g} u "
                "(IUPAC/CIAAW) and multiplies by the atomic mass constant"
            ),
            assumptions=(
                "the natural terrestrial isotope mixture is taken as a single "
                "mean mass: the real mixture broadens and shifts lines by an "
                "isotope effect this model leaves out",
                "assumes a neutral atom: an ion has lost an electron's worth of "
                "mass, which is parts in 1e4 and below anything resolved here",
            ),
        ),
    )

def hydrogen_like(Z: int, mu_ratio: float = 1.0) -> System:
    if Z < 1:
        raise ValueError(f"Z must be >= 1, got {Z}")
    assumptions = (
        ("uses an infinite nuclear mass (mu_ratio = 1)",) if mu_ratio == 1.0
        else (f"caller supplied mu_ratio = {mu_ratio:g}",)
    )
    return System(
        key=f"z{Z}",
        name=f"Hydrogen-like Z={Z}",
        Z=Z,
        mu_ratio=Quantity(
            value=mu_ratio,
            unit="m_e",
            label=f"mu/m_e (Z={Z})",
            provenance=Provenance(
                fidelity=Fidelity.EXACT,
                method="the reduced-mass ratio supplied by the caller",
                assumptions=assumptions,
            ),
        ),
        m_over_M=0.0,
        description=f"Generic one-electron ion, Z={Z}.",
    )
