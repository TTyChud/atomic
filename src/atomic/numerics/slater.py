
from dataclasses import dataclass

import numpy as np

__all__ = [
    "MultipoleGeometry",
    "multipole_geometry",
    "pair_potential",
    "slater_f",
    "slater_g",
]

def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    increments = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    out = np.empty_like(y)
    out[0] = 0.0
    np.cumsum(increments, out=out[1:])
    return out

@dataclass(frozen=True)
class MultipoleGeometry:

    r: np.ndarray
    half_dr: np.ndarray
    r_k: np.ndarray
    r_k1: np.ndarray
    r_inv_k1: np.ndarray

    def potential(self, p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
        density = p_a * p_b
        inner = self._cumulative(density * self.r_k)
        outer_total = self._cumulative(density * self.r_inv_k1)
        return inner / self.r_k1 + (outer_total[-1] - outer_total) * self.r_k

    def _cumulative(self, y: np.ndarray) -> np.ndarray:
        out = np.empty_like(y)
        out[0] = 0.0
        np.cumsum((y[1:] + y[:-1]) * self.half_dr, out=out[1:])
        return out

def multipole_geometry(r: np.ndarray, k: int) -> MultipoleGeometry:
    if k < 0:
        raise ValueError(f"multipole order k must be >= 0, got {k}")
    if r[0] <= 0.0:
        raise ValueError(
            f"the radial grid must start strictly above zero, got r[0]={r[0]!r}; "
            "at r = 0 the outer integrand is 0 * inf = nan, and np.cumsum then "
            "propagates that nan across the whole grid"
        )
    return MultipoleGeometry(
        r=r,
        half_dr=0.5 * np.diff(r),
        r_k=r**k,
        r_k1=r ** (k + 1),
        r_inv_k1=r ** (-(k + 1)),
    )

def pair_potential(
    p_a: np.ndarray, p_b: np.ndarray, r: np.ndarray, k: int
) -> np.ndarray:
    if not (p_a.shape == p_b.shape == r.shape):
        raise ValueError("orbitals and grid must have the same shape")
    return multipole_geometry(r, k).potential(p_a, p_b)

def slater_f(p_a: np.ndarray, p_b: np.ndarray, r: np.ndarray, k: int) -> float:
    return float(np.trapezoid(p_a**2 * pair_potential(p_b, p_b, r, k), r))

def slater_g(p_a: np.ndarray, p_b: np.ndarray, r: np.ndarray, k: int) -> float:
    return float(np.trapezoid(p_a * p_b * pair_potential(p_a, p_b, r, k), r))
