
from dataclasses import dataclass

from atomic.analytic.hydrogen import energy
from atomic.constants import E0_V_PER_M
from atomic.provenance import Fidelity, Provenance, Quantity

_S_ASSUMPTIONS = (
    "second-order perturbation theory (linear + quadratic), neglecting "
    "third and higher orders",
    "the field is static, so this manifold is a resonance rather than a true bound state, "
    "and field ionization is neglected",
    "gross structure only, neglecting fine structure and its low-field crossover",
    "a non-relativistic treatment, so this is independent of alpha: altering "
    "alpha does not move this shift",
)

@dataclass(frozen=True)
class StarkSublevel:
    n1: int
    n2: int
    m: int
    k: int
    energy: Quantity

def stark_sublevels(
    n: int, Z: int = 1, mu_ratio: float = 1.0, field_mv_per_m: float = 0.0,
) -> list[StarkSublevel]:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if Z < 1:
        raise ValueError(f"Z must be >= 1, got {Z}")
    if field_mv_per_m < 0:
        raise ValueError(f"field_mv_per_m must be >= 0, got {field_mv_per_m}")

    f_au = field_mv_per_m * 1e6 / E0_V_PER_M
    e_bohr = energy(n, Z=Z, mu_ratio=mu_ratio).value
    zm = Z * mu_ratio
    f_ion = (Z ** 3) * (mu_ratio ** 2) / (16.0 * n ** 4)

    method = (
        "parabolic Stark, 2nd-order perturbation theory (linear + quadratic); "
        f"F = {field_mv_per_m:g} MV/m = {f_au:.3e} a.u. "
        f"(F/F_ion = {f_au / f_ion:.2e})"
    )

    out: list[StarkSublevel] = []
    for m in range(-(n - 1), n):
        am = abs(m)
        for n1 in range(0, n - am):
            n2 = n - am - 1 - n1
            k = n1 - n2
            lin = 1.5 * n * k * f_au / zm
            quad = (
                -(1.0 / 16.0) * n ** 4
                * (17 * n * n - 3 * k * k - 9 * m * m + 19)
                * f_au * f_au / (Z ** 4 * mu_ratio ** 3)
            )
            value = e_bohr + lin + quad
            err = abs(quad) * (f_au / f_ion) if f_au > 0.0 else 0.0
            out.append(StarkSublevel(
                n1=n1, n2=n2, m=m, k=k,
                energy=Quantity(
                    value=value, unit="hartree",
                    label=f"E_Stark {n},n1={n1},n2={n2},m={m} (F={field_mv_per_m:g}MV/m)",
                    provenance=Provenance(
                        fidelity=Fidelity.APPROXIMATION, method=method,
                        assumptions=_S_ASSUMPTIONS, error_estimate=err,
                        refinement=(
                            "going to third-order Stark, then to the full "
                            "non-perturbative (field-ionization) resonance treatment"
                        ),
                    ),
                ),
            ))
    return out
