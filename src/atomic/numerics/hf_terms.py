
from dataclasses import dataclass

import numpy as np

from atomic.analytic.wigner import wigner_3j
from atomic.numerics.slater import MultipoleGeometry, multipole_geometry

__all__ = [
    "ExchangeOperator",
    "Subshell",
    "direct_potential",
    "exchange_apply",
    "exchange_coefficient",
    "exchange_operator",
    "same_shell_coefficient",
]

@dataclass(frozen=True)
class Subshell:

    n: int
    l: int
    q: int
    p: np.ndarray

def same_shell_coefficient(l_a: int, k: int, q_a: int) -> float:
    if k <= 0:
        raise ValueError(f"same-shell exchange needs k > 0, got {k}")
    tj = wigner_3j(l_a, k, l_a, 0, 0, 0)
    return (q_a - 1) * ((2 * l_a + 1) / (4 * l_a + 1)) * tj * tj

def exchange_coefficient(l_a: int, k: int, l_b: int, q_b: int) -> float:
    tj = wigner_3j(l_a, k, l_b, 0, 0, 0)
    return 0.5 * q_b * tj * tj

def direct_potential(
    subshells: tuple[Subshell, ...], a_index: int, r: np.ndarray
) -> np.ndarray:
    a = subshells[a_index]
    geometry = multipole_geometry(r, 0)
    v = (a.q - 1) * geometry.potential(a.p, a.p)
    for i, b in enumerate(subshells):
        if i != a_index:
            v = v + b.q * geometry.potential(b.p, b.p)
    return v

@dataclass(frozen=True)
class ExchangeOperator:

    terms: tuple[tuple[float, np.ndarray, MultipoleGeometry], ...]

    def apply(self, psi: np.ndarray) -> np.ndarray:
        out = np.zeros_like(psi)
        for coefficient, partner, geometry in self.terms:
            out += coefficient * geometry.potential(partner, psi) * partner
        return out

def exchange_operator(
    subshells: tuple[Subshell, ...], a_index: int, r: np.ndarray
) -> ExchangeOperator:
    a = subshells[a_index]
    geometries: dict[int, MultipoleGeometry] = {}

    def geometry(k: int) -> MultipoleGeometry:
        if k not in geometries:
            geometries[k] = multipole_geometry(r, k)
        return geometries[k]

    terms: list[tuple[float, np.ndarray, MultipoleGeometry]] = []

    for k in range(2, 2 * a.l + 1, 2):
        c = same_shell_coefficient(a.l, k, a.q)
        if c:
            terms.append((c, a.p, geometry(k)))

    for i, b in enumerate(subshells):
        if i == a_index:
            continue
        for k in range(abs(a.l - b.l), a.l + b.l + 1):
            c = exchange_coefficient(a.l, k, b.l, b.q)
            if c:
                terms.append((c, b.p, geometry(k)))

    return ExchangeOperator(terms=tuple(terms))

def exchange_apply(
    subshells: tuple[Subshell, ...], a_index: int, psi: np.ndarray, r: np.ndarray
) -> np.ndarray:
    return exchange_operator(subshells, a_index, r).apply(psi)
