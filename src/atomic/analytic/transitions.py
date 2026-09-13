
import math
from functools import lru_cache

import numpy as np
from scipy import constants as _sc
from scipy.special import roots_laguerre

from atomic.analytic.hydrogen import (
    _radial_eval,
    _validate_physical,
    energy,
    validate_quantum_numbers,
)
from atomic.analytic.wigner import wigner_6j
from atomic.constants import ALPHA
from atomic.provenance import Fidelity, Provenance, Quantity

_T_AU = _sc.physical_constants["atomic unit of time"][0]

_ONE_ELECTRON = (
    "one-electron hydrogenic wavefunctions, which are exact",
    "the electric-dipole (E1) approximation, neglecting higher multipoles",
    "no fine-structure, relativistic or QED correction to the rate",
)

def _gauss_laguerre_nodes(n: int, n2: int) -> int:
    return (n + n2 + 3) // 2

@lru_cache(maxsize=4096)
def _dipole_value_and_error(
    n: int, l: int, n2: int, l2: int, kappa: float
) -> tuple[float, float, int]:
    nodes = _gauss_laguerre_nodes(n, n2)
    coarse = _dipole_quadrature(n, l, n2, l2, kappa, nodes)
    fine = _dipole_quadrature(n, l, n2, l2, kappa, 2 * nodes)
    return fine, abs(fine - coarse), 2 * nodes

def _dipole_quadrature(n: int, l: int, n2: int, l2: int, kappa: float, nodes: int) -> float:
    a = kappa * (1.0 / n + 1.0 / n2)
    x, w = roots_laguerre(nodes)
    r = x / a
    integrand = _radial_eval(n, l, r, kappa) * _radial_eval(n2, l2, r, kappa) * r**3 * np.exp(x)
    return float(np.dot(w, integrand) / a)

def dipole_radial_integral(
    n: int, l: int, n2: int, l2: int, Z: int = 1, mu_ratio: float = 1.0,
) -> Quantity:
    validate_quantum_numbers(n, l)
    validate_quantum_numbers(n2, l2)
    _validate_physical(Z, mu_ratio)
    kappa = Z * mu_ratio
    a, b = sorted(((n, l), (n2, l2)))
    value, err, nodes = _dipole_value_and_error(a[0], a[1], b[0], b[1], kappa)
    return Quantity(
        value=value,
        unit="bohr",
        label=f"<{n2},{l2}|r|{n},{l}> (Z={Z}, mu/m_e={mu_ratio:g})",
        provenance=Provenance(
            fidelity=Fidelity.NUMERICAL,
            method=(
                f"Gauss-Laguerre quadrature of the exact R_nl dipole integral "
                f"({nodes} nodes; the integrand is exp(-a r) times a degree-"
                f"{n + n2 + 1} polynomial, so the rule is exact bar roundoff)"
            ),
            assumptions=_ONE_ELECTRON,
            error_estimate=err,
            refinement="the closed-form Gordon hypergeometric dipole integral",
        ),
    )

def f_from_radial_dipole(
    dE_hartree: float, l_low: int, l_up: int, dipole_bohr: float
) -> float:
    return (
        (2.0 / 3.0) * dE_hartree
        * (max(l_low, l_up) / (2.0 * l_low + 1.0)) * dipole_bohr**2
    )

def A_from_radial_dipole(
    dE_hartree: float, l_up: int, l_low: int, dipole_bohr: float,
    alpha: float = ALPHA,
) -> float:
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"alpha must be finite and positive, got {alpha!r}")
    a_au = (
        (4.0 / 3.0) * alpha**3 * dE_hartree**3
        * (max(l_up, l_low) / (2.0 * l_up + 1.0)) * dipole_bohr**2
    )
    return a_au / _T_AU

def _forbidden(kind: str, label: str, unit: str) -> Quantity:
    return Quantity(
        value=0.0,
        unit=unit,
        label=label,
        provenance=Provenance(
            fidelity=Fidelity.EXACT,
            method=f"electric-dipole selection rule: {kind} (Delta l = +/-1 required)",
            assumptions=("this is a selection-rule zero, not a numerical underflow",),
            error_estimate=0.0,
        ),
    )

def oscillator_strength(
    n_low: int, l_low: int, n_up: int, l_up: int, Z: int = 1, mu_ratio: float = 1.0,
) -> Quantity:
    validate_quantum_numbers(n_low, l_low)
    validate_quantum_numbers(n_up, l_up)
    dE = energy(n_up, Z=Z, mu_ratio=mu_ratio).value - energy(n_low, Z=Z, mu_ratio=mu_ratio).value
    if dE <= 0.0:
        raise ValueError(
            "for absorption the upper level must be above the lower "
            f"(got E({n_up}) <= E({n_low}))"
        )
    label = f"f {n_low}{l_low}->{n_up}{l_up}"
    if abs(l_up - l_low) != 1:
        return _forbidden("Delta l != +/-1", label, "dimensionless")
    R = dipole_radial_integral(n_low, l_low, n_up, l_up, Z=Z, mu_ratio=mu_ratio)
    f = f_from_radial_dipole(dE, l_low, l_up, R.value)
    rerr = (R.provenance.error_estimate or 0.0)
    f_err = 2.0 * abs(f) * (rerr / abs(R.value)) if R.value != 0.0 else 0.0
    return Quantity(
        value=f,
        unit="dimensionless",
        label=label,
        provenance=Provenance(
            fidelity=Fidelity.NUMERICAL,
            method="f = (2/3) dE (l_max/(2l+1)) |R|^2, R from dipole_radial_integral",
            assumptions=_ONE_ELECTRON,
            error_estimate=f_err,
            refinement=R.provenance.refinement,
        ),
    )

def einstein_A(
    n_up: int, l_up: int, n_low: int, l_low: int, Z: int = 1, mu_ratio: float = 1.0,
    alpha: float = ALPHA,
) -> Quantity:
    validate_quantum_numbers(n_up, l_up)
    validate_quantum_numbers(n_low, l_low)
    dE = energy(n_up, Z=Z, mu_ratio=mu_ratio).value - energy(n_low, Z=Z, mu_ratio=mu_ratio).value
    label = f"A {n_up}{l_up}->{n_low}{l_low}"
    if abs(l_up - l_low) != 1 or dE <= 0.0:
        return _forbidden("no E1 decay channel", label, "s^-1")
    R = dipole_radial_integral(n_up, l_up, n_low, l_low, Z=Z, mu_ratio=mu_ratio)
    a_s = A_from_radial_dipole(dE, l_up, l_low, R.value, alpha=alpha)
    rerr = (R.provenance.error_estimate or 0.0)
    a_err = 2.0 * abs(a_s) * (rerr / abs(R.value)) if R.value != 0.0 else 0.0
    method = "A = (4/3) alpha^3 dE^3 (l_max/(2l'+1)) |R|^2 / t_au (atomic-time unit)"
    fidelity = Fidelity.NUMERICAL
    if not math.isclose(alpha, ALPHA, rel_tol=1e-12):
        method += f"; altered fine-structure constant alpha = {alpha:g} (real {ALPHA:g})"
        fidelity = Fidelity.COUNTERFACTUAL
    return Quantity(
        value=a_s,
        unit="s^-1",
        label=label,
        provenance=Provenance(
            fidelity=fidelity,
            method=method,
            assumptions=_ONE_ELECTRON,
            error_estimate=a_err,
            refinement=R.provenance.refinement,
        ),
    )

def _validate_j(l: int, j: float, name: str) -> None:
    allowed = [l - 0.5, l + 0.5] if l > 0 else [0.5]
    if not any(abs(j - a) < 1e-9 for a in allowed):
        raise ValueError(
            f"{name}: j must be l +/- 1/2 for l = {l} (allowed {allowed}), got {j}"
        )

def _fine_branching(l_up: int, j_up: float, l_low: int, j_low: float) -> float:
    return (2.0 * j_low + 1.0) * wigner_6j(j_low, 1, j_up, l_up, 0.5, l_low) ** 2

_FINE_METHOD = (
    "A = (4/3) alpha^3 dE^3 (2j+1) {j 1 j'; l' 1/2 l}^2 l_max |R|^2 / t_au; "
    "the 6j symbol splits the multiplet rate across j (spin is a spectator)"
)

def einstein_A_fine(
    n_up: int, l_up: int, j_up: float,
    n_low: int, l_low: int, j_low: float,
    Z: int = 1, mu_ratio: float = 1.0,
    dE_hartree: float | None = None,
    alpha: float = ALPHA,
) -> Quantity:
    validate_quantum_numbers(n_up, l_up)
    validate_quantum_numbers(n_low, l_low)
    _validate_j(l_up, j_up, "upper level")
    _validate_j(l_low, j_low, "lower level")
    dE = (
        dE_hartree if dE_hartree is not None
        else energy(n_up, Z=Z, mu_ratio=mu_ratio).value
        - energy(n_low, Z=Z, mu_ratio=mu_ratio).value
    )
    label = f"A {n_up}{l_up}(j={j_up})->{n_low}{l_low}(j={j_low})"
    if abs(l_up - l_low) != 1 or dE <= 0.0:
        return _forbidden("no E1 decay channel", label, "s^-1")
    branch = _fine_branching(l_up, j_up, l_low, j_low)
    if branch == 0.0:
        return _forbidden("6j triangle rule (Delta j = 0, +/-1)", label, "s^-1")
    R = dipole_radial_integral(n_up, l_up, n_low, l_low, Z=Z, mu_ratio=mu_ratio)
    l_max = max(l_up, l_low)
    a_s = (4.0 / 3.0) * alpha**3 * dE**3 * branch * l_max * R.value**2 / _T_AU
    rerr = R.provenance.error_estimate or 0.0
    method = _FINE_METHOD
    fidelity = Fidelity.NUMERICAL
    if not math.isclose(alpha, ALPHA, rel_tol=1e-12):
        method += f"; altered fine-structure constant alpha = {alpha:g} (real {ALPHA:g})"
        fidelity = Fidelity.COUNTERFACTUAL
    return Quantity(
        value=a_s,
        unit="s^-1",
        label=label,
        provenance=Provenance(
            fidelity=fidelity,
            method=method,
            assumptions=_ONE_ELECTRON
            + ("the rate is resolved by j; the transition energy is the gross, n-only value",),
            error_estimate=2.0 * abs(a_s) * (rerr / abs(R.value)) if R.value else 0.0,
            refinement=R.provenance.refinement,
        ),
    )

def oscillator_strength_fine(
    n_low: int, l_low: int, j_low: float,
    n_up: int, l_up: int, j_up: float,
    Z: int = 1, mu_ratio: float = 1.0,
    dE_hartree: float | None = None,
) -> Quantity:
    validate_quantum_numbers(n_low, l_low)
    validate_quantum_numbers(n_up, l_up)
    _validate_j(l_low, j_low, "lower level")
    _validate_j(l_up, j_up, "upper level")
    dE = (
        dE_hartree if dE_hartree is not None
        else energy(n_up, Z=Z, mu_ratio=mu_ratio).value
        - energy(n_low, Z=Z, mu_ratio=mu_ratio).value
    )
    if dE <= 0.0:
        raise ValueError(
            "for absorption the upper level must be above the lower "
            f"(got E({n_up}) <= E({n_low}))"
        )
    label = f"f {n_low}{l_low}(j={j_low})->{n_up}{l_up}(j={j_up})"
    if abs(l_up - l_low) != 1:
        return _forbidden("Delta l != +/-1", label, "dimensionless")
    branch = _fine_branching(l_low, j_low, l_up, j_up)
    if branch == 0.0:
        return _forbidden("6j triangle rule (Delta j = 0, +/-1)", label, "dimensionless")
    R = dipole_radial_integral(n_low, l_low, n_up, l_up, Z=Z, mu_ratio=mu_ratio)
    l_max = max(l_low, l_up)
    f = (2.0 / 3.0) * dE * branch * l_max * R.value**2
    rerr = R.provenance.error_estimate or 0.0
    return Quantity(
        value=f,
        unit="dimensionless",
        label=label,
        provenance=Provenance(
            fidelity=Fidelity.NUMERICAL,
            method=(
                "f = (2/3) dE (2j'+1) {j' 1 j; l 1/2 l'}^2 l_max |R|^2; "
                "the 6j symbol splits the multiplet strength across j"
            ),
            assumptions=_ONE_ELECTRON
            + ("the strength is resolved by j; the transition energy is the gross value",),
            error_estimate=2.0 * abs(f) * (rerr / abs(R.value)) if R.value else 0.0,
            refinement=R.provenance.refinement,
        ),
    )

def lifetime_fine(
    n: int, l: int, j: float, Z: int = 1, mu_ratio: float = 1.0,
    alpha: float = ALPHA,
) -> Quantity:
    validate_quantum_numbers(n, l)
    _validate_j(l, j, "level")
    total = 0.0
    var = 0.0
    for n2 in range(1, n):
        for l2 in (l - 1, l + 1):
            if not 0 <= l2 < n2:
                continue
            for j2 in ([l2 - 0.5, l2 + 0.5] if l2 > 0 else [0.5]):
                a = einstein_A_fine(n, l, j, n2, l2, j2, Z=Z, mu_ratio=mu_ratio, alpha=alpha)
                total += a.value
                var += (a.provenance.error_estimate or 0.0) ** 2
    if total <= 0.0:
        value, err = math.inf, 0.0
    else:
        value, err = 1.0 / total, math.sqrt(var) / total**2
    return Quantity(
        value=value,
        unit="s",
        label=f"tau {n}{l}(j={j}) (Z={Z}, mu/m_e={mu_ratio:g})",
        provenance=Provenance(
            fidelity=Fidelity.NUMERICAL,
            method="tau = 1 / sum A(n l j -> n' l' j'), over E1 channels only",
            assumptions=_ONE_ELECTRON
            + ("summed over every lower dipole-allowed fine level (n' < n)",),
            error_estimate=err,
            refinement="adding higher multipoles and QED corrections to the rate",
        ),
    )

def lifetime(n: int, l: int, Z: int = 1, mu_ratio: float = 1.0, alpha: float = ALPHA) -> Quantity:
    validate_quantum_numbers(n, l)
    total = 0.0
    var = 0.0
    for n2 in range(1, n):
        for l2 in (l - 1, l + 1):
            if 0 <= l2 < n2:
                a = einstein_A(n, l, n2, l2, Z=Z, mu_ratio=mu_ratio, alpha=alpha)
                total += a.value
                var += (a.provenance.error_estimate or 0.0) ** 2
    if total <= 0.0:
        value, err = math.inf, 0.0
    else:
        value, err = 1.0 / total, math.sqrt(var) / total**2
    return Quantity(
        value=value,
        unit="s",
        label=f"tau {n}{l} (Z={Z}, mu/m_e={mu_ratio:g})",
        provenance=Provenance(
            fidelity=Fidelity.NUMERICAL,
            method="tau = 1 / sum_{lower} A(n l -> n' l'), over E1 channels only",
            assumptions=_ONE_ELECTRON
            + ("summed over every lower dipole-allowed level (n' < n)",),
            error_estimate=err,
            refinement="resolving the rates by fine structure and adding higher multipoles",
        ),
    )
