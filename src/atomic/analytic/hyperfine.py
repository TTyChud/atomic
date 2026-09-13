
import math
from dataclasses import dataclass

from scipy import constants as _sc

from atomic.analytic.hydrogen import energy
from atomic.constants import ALPHA
from atomic.provenance import Fidelity, Provenance, Quantity
from atomic.systems import System

_G_E = abs(_sc.physical_constants["electron g factor"][0])
_M_E_OVER_M_P = 1.0 / _sc.physical_constants["proton-electron mass ratio"][0]

_J_S = 0.5

_HF_ASSUMPTIONS = (
    "the Fermi contact interaction and s-states (l = 0) only, so the electron "
    "density at the nucleus is what drives the coupling",
    "a non-relativistic treatment: the electron g comes from the measured free-electron "
    "moment with an exact reduced mass, and bound-state QED beyond that is neglected",
    "the relativistic (Breit) correction ~ (Z alpha)^2 is neglected, and it grows with Z",
    "nuclear structure is neglected: finite size, the Zemach radius, the hyperfine anomaly",
    "the l > 0 orbital and spin-dipolar channel is deferred, so it is not in here",
)

_HF_METHOD = (
    "magnetic-dipole hyperfine, Fermi contact: "
    "A = (2/3) g_e g_I (m_e/m_p) alpha^2 (mu/m_e)^3 Z^3/n^3"
)

def hyperfine_constant(
    n: int, Z: int = 1, mu_ratio: float = 1.0, g_I: float = 0.0,
    alpha: float = ALPHA,
) -> Quantity:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if Z < 1:
        raise ValueError(f"Z must be >= 1, got {Z}")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"alpha must be finite and positive, got {alpha!r}")

    value = (
        (2.0 / 3.0) * _G_E * g_I * _M_E_OVER_M_P
        * alpha**2 * mu_ratio**3 * Z**3 / n**3
    )
    error = abs(value) * ((Z * alpha) ** 2 + 1e-4)
    method = _HF_METHOD
    fidelity = Fidelity.APPROXIMATION
    if not math.isclose(alpha, ALPHA, rel_tol=1e-12):
        method += f"; altered fine-structure constant alpha = {alpha:g} (real {ALPHA:g})"
        fidelity = Fidelity.COUNTERFACTUAL
    return Quantity(
        value=value,
        unit="hartree",
        label=f"A_hf {n}s (Z={Z}, mu/m_e={mu_ratio:g}, g_I={g_I:g})",
        provenance=Provenance(
            fidelity=fidelity,
            method=method,
            assumptions=_HF_ASSUMPTIONS,
            error_estimate=error,
            refinement=(
                "relativistic (Dirac) hyperfine, then bound-state QED "
                "and the nuclear Zemach correction, and add the l > 0 dipolar channel"
            ),
        ),
    )

@dataclass(frozen=True)
class HyperfineLevel:
    F: float
    shift: Quantity
    energy: Quantity

def _f_values(I: float, J: float) -> list[float]:
    lo = abs(I - J)
    n_steps = round(I + J - lo)
    return [lo + k for k in range(n_steps + 1)]

def hyperfine_levels(
    n: int, I: float, Z: int = 1, mu_ratio: float = 1.0, g_I: float = 0.0,
    alpha: float = ALPHA,
) -> list[HyperfineLevel]:
    A = hyperfine_constant(n, Z=Z, mu_ratio=mu_ratio, g_I=g_I, alpha=alpha)
    e_gross = energy(n, Z=Z, mu_ratio=mu_ratio).value
    J = _J_S
    ij = I * (I + 1.0)
    jj = J * (J + 1.0)

    out: list[HyperfineLevel] = []
    for F in _f_values(I, J):
        shift_val = 0.5 * A.value * (F * (F + 1.0) - ij - jj)
        shift = Quantity(
            value=shift_val,
            unit="hartree",
            label=f"dE_hf {n}s F={F:g}",
            provenance=Provenance(
                fidelity=A.provenance.fidelity,
                method=f"{A.provenance.method}; dE(F) = (A/2)[F(F+1) - I(I+1) - J(J+1)]",
                assumptions=_HF_ASSUMPTIONS,
                error_estimate=abs(shift_val) * ((Z * alpha) ** 2 + 1e-4),
                refinement=A.provenance.refinement,
            ),
        )
        out.append(HyperfineLevel(
            F=F,
            shift=shift,
            energy=Quantity(
                value=e_gross + shift_val,
                unit="hartree",
                label=f"E {n}s F={F:g} (Z={Z}, mu/m_e={mu_ratio:g})",
                provenance=shift.provenance,
            ),
        ))
    return out

@dataclass(frozen=True)
class Nucleus:
    name: str
    I: float
    g_I: float
    note: str = ""

def _gI_from_codata(moment_ratio_name: str, spin: float) -> float:
    return _sc.physical_constants[moment_ratio_name][0] / spin

_NUCLEI: dict[str, Nucleus] = {
    "h": Nucleus("proton", 0.5,
                 _gI_from_codata("proton mag. mom. to nuclear magneton ratio", 0.5)),
    "d": Nucleus("deuteron", 1.0,
                 _gI_from_codata("deuteron mag. mom. to nuclear magneton ratio", 1.0)),
    "t": Nucleus("triton", 0.5,
                 _gI_from_codata("triton mag. mom. to nuclear magneton ratio", 0.5)),
    "he+": Nucleus("alpha particle (He-4)", 0.0, 0.0,
                   note="I = 0: a spin-0 nucleus has no magnetic moment, so there is "
                        "no hyperfine splitting to show (He-3 would split)"),
}

_UNAVAILABLE: dict[str, str] = {
    "ps": "positronium: the partner is a positron with a Bohr-magneton moment and "
          "annihilation contributes, so the nuclear-magneton contact formula does not apply",
    "mu-h": "muonic hydrogen: the orbiter is a muon whose own g-factor and mass "
            "enter the coupling, so the electron-contact formula does not apply",
}

@dataclass(frozen=True)
class HyperfineReport:
    available: bool
    n: int
    system_key: str
    nucleus_name: str | None = None
    I: float | None = None
    A: Quantity | None = None
    levels: tuple[HyperfineLevel, ...] = ()
    note: str | None = None
    reason: str | None = None

def hyperfine_report(n: int, system: System) -> HyperfineReport:
    key = system.key
    if key in _UNAVAILABLE:
        return HyperfineReport(
            available=False, n=n, system_key=key, reason=_UNAVAILABLE[key]
        )
    nucleus = _NUCLEI.get(key)
    if nucleus is None:
        return HyperfineReport(
            available=False, n=n, system_key=key,
            reason="no identified nucleus with a measured magnetic moment here",
        )

    mu = system.mu_ratio.value
    A = hyperfine_constant(n, Z=system.Z, mu_ratio=mu, g_I=nucleus.g_I)
    levels = hyperfine_levels(
        n, I=nucleus.I, Z=system.Z, mu_ratio=mu, g_I=nucleus.g_I
    )
    return HyperfineReport(
        available=True, n=n, system_key=key,
        nucleus_name=nucleus.name, I=nucleus.I, A=A,
        levels=tuple(levels), note=nucleus.note or None,
    )
