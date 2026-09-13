
import math

from atomic.analytic.hydrogen import energy, validate_quantum_numbers
from atomic.constants import ALPHA
from atomic.provenance import Fidelity, Provenance, Quantity

_G2 = 2.0 * 0.00116

_FS_ASSUMPTIONS = (
    "Pauli approximation to order alpha^2: spin-orbit + relativistic kinetic + Darwin",
    "electron g = 2 exactly, neglecting an anomalous moment worth ~0.1% of the splitting",
    "the reduced mass is scaled to leading order, and nuclear recoil O(m/M) is neglected",
    "no Lamb shift or QED, and no hyperfine structure",
)

def validate_j(l: int, j: float) -> None:
    if j < 0.5 or abs(abs(j - l) - 0.5) > 1e-12:
        raise ValueError(f"j must be l +/- 1/2 (and >= 1/2), got l={l}, j={j}")

def fine_structure_shift(
    n: int, l: int, j: float, Z: int = 1, mu_ratio: float = 1.0,
    m_over_M: float = 0.0, alpha: float = ALPHA,
) -> Quantity:
    validate_quantum_numbers(n, l)
    validate_j(l, j)
    value = -(mu_ratio * Z**4 * alpha**2 / (2.0 * n**4)) * (n / (j + 0.5) - 0.75)
    error = abs(value) * ((Z * alpha) ** 2 + m_over_M + _G2)
    altered = not math.isclose(alpha, ALPHA, rel_tol=1e-12)
    method = (
        "combined Pauli fine structure "
        "dE = -(mu' Z^4 alpha^2 / 2 n^4)(n/(j+1/2) - 3/4)"
    )
    if altered:
        method += f"; altered fine-structure constant alpha = {alpha:g} (real {ALPHA:g})"
    return Quantity(
        value=value,
        unit="hartree",
        label=f"dE_fs {n},{l},j={j:g} (Z={Z}, mu/m_e={mu_ratio:g})",
        provenance=Provenance(
            fidelity=Fidelity.COUNTERFACTUAL if altered else Fidelity.APPROXIMATION,
            method=method,
            assumptions=_FS_ASSUMPTIONS,
            error_estimate=error,
            refinement="solving Dirac hydrogen exactly instead (analytic/dirac.py)",
        ),
    )

def level_energy(
    n: int, l: int, j: float, Z: int = 1, mu_ratio: float = 1.0,
    m_over_M: float = 0.0, alpha: float = ALPHA,
) -> Quantity:
    bohr = energy(n, Z=Z, mu_ratio=mu_ratio)
    shift = fine_structure_shift(
        n, l, j, Z=Z, mu_ratio=mu_ratio, m_over_M=m_over_M, alpha=alpha
    )
    return Quantity(
        value=bohr.value + shift.value,
        unit="hartree",
        label=f"E {n},{l},j={j:g} (Z={Z}, mu/m_e={mu_ratio:g})",
        provenance=Provenance(
            fidelity=shift.provenance.fidelity,
            method=f"{bohr.provenance.method} + {shift.provenance.method}",
            assumptions=_FS_ASSUMPTIONS,
            error_estimate=shift.provenance.error_estimate,
            refinement=shift.provenance.refinement,
        ),
    )
