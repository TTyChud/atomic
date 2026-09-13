
import math
from dataclasses import dataclass

from atomic.analytic.dirac import dirac_energy
from atomic.analytic.fine_structure import level_energy
from atomic.analytic.hydrogen import validate_quantum_numbers
from atomic.constants import ALPHA, B0_TESLA
from atomic.provenance import Fidelity, Provenance, Quantity

_MU_B_AU = 0.5
MU_B_PER_TESLA = _MU_B_AU / B0_TESLA
_G2 = 2.0 * 0.00116

_Z_ASSUMPTIONS = (
    "the linear (paramagnetic) Zeeman term only, neglecting the diamagnetic B^2 term",
    "electron g_s = 2 exactly, neglecting an anomalous moment worth ~0.1% of the spin part",
    "coupling to other n manifolds is neglected",
    "the diagonal comes from the selected level model: alpha^2 fine structure or exact Dirac",
)

@dataclass(frozen=True)
class ZeemanSublevel:
    m_j: float
    branch: str
    j_label: float
    high_field_label: str
    energy: Quantity

def lande_g(l: int, j: float) -> float:
    s = 0.5
    return 1.0 + (j * (j + 1.0) + s * (s + 1.0) - l * (l + 1.0)) / (2.0 * j * (j + 1.0))

def _high_field_label(m_j: float, m_s: float) -> str:
    return f"m_l={m_j - m_s:g}, m_s={m_s:+g}"

def _mean_sq_radius(n: int, l: int, Z: int) -> float:
    return (n * n / (2.0 * Z * Z)) * (5.0 * n * n + 1.0 - 3.0 * l * (l + 1.0))

def zeeman_sublevels(
    n: int, l: int, Z: int = 1, mu_ratio: float = 1.0, m_over_M: float = 0.0,
    alpha: float = ALPHA, b_tesla: float = 0.0, dirac: bool = False,
) -> list[ZeemanSublevel]:
    validate_quantum_numbers(n, l)
    if Z < 1:
        raise ValueError(f"Z must be >= 1, got {Z}")
    if b_tesla < 0:
        raise ValueError(f"b_tesla must be >= 0, got {b_tesla}")

    def diag(j: float) -> Quantity:
        if dirac:
            return dirac_energy(n, j, Z=Z, mu_ratio=mu_ratio, alpha=alpha)
        return level_energy(
            n, l, j, Z=Z, mu_ratio=mu_ratio, m_over_M=m_over_M, alpha=alpha
        )

    muB_b = MU_B_PER_TESLA * b_tesla
    altered = not math.isclose(alpha, ALPHA, rel_tol=1e-12)
    fidelity = Fidelity.COUNTERFACTUAL if altered else Fidelity.APPROXIMATION
    diamag = 0.125 * (b_tesla / B0_TESLA) ** 2 * _mean_sq_radius(n, l, Z)
    denom = 2 * l + 1
    method = (
        "Breit-Rabi (fine structure + linear Zeeman, g_s=2) eigenvalue; "
        f"mu_B*B = {muB_b:.3e} hartree at B = {b_tesla:g} T"
        + ("; exact Dirac diagonal split by a perturbative linear-Zeeman model" if dirac else "")
        + (f"; altered alpha = {alpha:g} (real {ALPHA:g})" if altered else "")
    )

    def make(value: float, m_j: float, branch: str, j_label: float, m_s: float,
             underlying_err: float | None) -> ZeemanSublevel:
        err = (underlying_err or 0.0) + diamag + _G2 * abs(muB_b * m_j)
        return ZeemanSublevel(
            m_j=m_j, branch=branch, j_label=j_label,
            high_field_label=_high_field_label(m_j, m_s),
            energy=Quantity(
                value=value, unit="hartree",
                label=f"E_Zeeman {n},{l},m_j={m_j:g},{branch} (B={b_tesla:g}T)",
                provenance=Provenance(
                    fidelity=fidelity, method=method, assumptions=_Z_ASSUMPTIONS,
                    error_estimate=err,
                    refinement=(
                        "adding the diamagnetic (B^2) term, then going to "
                        "Paschen-Back beyond this two-effect model"
                    ),
                ),
            ),
        )

    j_up = l + 0.5
    e_up = diag(j_up)
    m_values = [(-(l + 0.5) + k) for k in range(2 * l + 2)]
    out: list[ZeemanSublevel] = []
    for m_j in m_values:
        stretched = abs(m_j) > l
        if l == 0 or stretched:
            zeeman_diag = muB_b * m_j * (2 * l + 2) / denom
            m_s = math.copysign(0.5, m_j)
            out.append(make(
                e_up.value + zeeman_diag, m_j, "single", j_up, m_s,
                e_up.provenance.error_estimate,
            ))
            continue
        e_dn = diag(l - 0.5)
        h00 = e_up.value + muB_b * m_j * (2 * l + 2) / denom
        h11 = e_dn.value + muB_b * m_j * (2 * l) / denom
        h01 = muB_b * math.sqrt((l + 0.5) ** 2 - m_j * m_j) / denom
        mean = 0.5 * (h00 + h11)
        disc = math.hypot(0.5 * (h00 - h11), h01)
        out.append(make(
            mean + disc, m_j, "upper", j_up, 0.5, e_up.provenance.error_estimate,
        ))
        out.append(make(
            mean - disc, m_j, "lower", l - 0.5, -0.5, e_dn.provenance.error_estimate,
        ))
    return out
