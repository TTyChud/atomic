
import math

from atomic.analytic.hydrogen import energy
from atomic.constants import ALPHA
from atomic.provenance import Fidelity, Provenance, Quantity

_DIRAC_ASSUMPTIONS = (
    "the exact eigenvalue of the one-body Dirac-Coulomb equation (point nucleus)",
    "no Lamb shift or QED radiative corrections, and in reality "
    "those split 2s1/2 from 2p1/2",
    "no hyperfine structure and no finite-nuclear-size correction",
    "the reduced mass enters by mu-scaling the rest energy, and two-body "
    "relativistic recoil is neglected",
)

def _validate(n: int, j: float, Z: int, alpha: float) -> None:
    if n < 1:
        raise ValueError(f"principal quantum number n must be >= 1, got {n}")
    if j < 0.5 or abs((j - 0.5) - round(j - 0.5)) > 1e-12:
        raise ValueError(f"j must be a half-integer >= 1/2, got {j}")
    if (j + 0.5) > n:
        raise ValueError(f"j={j} is not allowed for n={n} (need j <= n-1/2)")
    if Z < 1:
        raise ValueError(f"nuclear charge Z must be >= 1, got {Z}")
    if Z * alpha >= j + 0.5:
        raise ValueError(
            f"supercritical: Z*alpha = {Z * alpha:g} >= j+1/2 = {j + 0.5:g}; "
            "the point-Coulomb Dirac solution is not real here"
        )

def dirac_energy(
    n: int, j: float, Z: int = 1, mu_ratio: float = 1.0, alpha: float = ALPHA
) -> Quantity:
    _validate(n, j, Z, alpha)
    gamma = math.sqrt((j + 0.5) ** 2 - (Z * alpha) ** 2)
    d = n - (j + 0.5) + gamma
    x = (Z * alpha / d) ** 2
    s = math.sqrt(1.0 + x)
    e_bind = (mu_ratio / alpha**2) * (-x / (s * (1.0 + s)))

    altered = not math.isclose(alpha, ALPHA, rel_tol=1e-12)
    bohr = energy(n, Z=Z, mu_ratio=mu_ratio).value
    omitted = abs(bohr) * (Z * alpha) ** 3
    method = "exact Dirac-Coulomb energy E(n,j) = mu*c^2([1+(Za/D)^2]^(-1/2) - 1)"
    if altered:
        method += f"; altered fine-structure constant alpha = {alpha:g} (real {ALPHA:g})"
    return Quantity(
        value=e_bind,
        unit="hartree",
        label=f"E_Dirac {n},j={j:g} (Z={Z}, mu/m_e={mu_ratio:g})",
        provenance=Provenance(
            fidelity=Fidelity.COUNTERFACTUAL if altered else Fidelity.EXACT,
            method=method,
            assumptions=_DIRAC_ASSUMPTIONS,
            error_estimate=omitted,
            refinement=(
                "adding QED and the Lamb shift (the 2s-2p splitting), "
                "then hyperfine structure"
            ),
        ),
    )

def dirac_fine_splitting(
    n: int, l: int, Z: int = 1, mu_ratio: float = 1.0, alpha: float = ALPHA
) -> float:
    if l < 1:
        raise ValueError(f"fine splitting needs l >= 1, got {l}")
    hi = dirac_energy(n, l + 0.5, Z=Z, mu_ratio=mu_ratio, alpha=alpha).value
    lo = dirac_energy(n, l - 0.5, Z=Z, mu_ratio=mu_ratio, alpha=alpha).value
    return hi - lo
