from collections.abc import Callable

import numpy as np

from atomic.provenance import Fidelity, Provenance

_GSZ_NEUTRAL_DK: dict[int, tuple[float, float]] = {
    2: (0.381, 1.77),
    3: (0.462, 1.75),
    4: (0.769, 1.88),
    5: (0.970, 2.00),
    6: (0.939, 2.13),
    7: (0.848, 2.27),
    8: (0.735, 2.41),
    9: (0.663, 2.59),
    10: (0.558, 2.71),
    11: (0.584, 2.85),
    12: (0.670, 3.01),
    13: (0.860, 3.17),
    14: (0.988, 3.26),
    15: (1.055, 3.33),
    18: (1.045, 3.50),
}

def gsz_parameters(z: int, n_electrons: int) -> tuple[float, float]:
    if z < 1:
        raise ValueError(f"Z must be >= 1, got {z}")
    if not 1 <= n_electrons <= z + 1:
        raise ValueError(f"N must be in [1, Z+1], got {n_electrons} (Z={z})")
    if n_electrons != z or z not in _GSZ_NEUTRAL_DK:
        raise ValueError(
            f"no sourced GSZ parameters for Z={z}, N={n_electrons}; only "
            f"neutral He..P and Ar are vendored (Szydlik & Green 1974)"
        )
    d, k = _GSZ_NEUTRAL_DK[z]
    return d, k * d

def _omega(r: np.ndarray, d: float, h: float) -> np.ndarray:
    with np.errstate(over="ignore"):
        return 1.0 / (h * np.expm1(r / d) + 1.0)

def z_eff(z: int, n_electrons: int, r: np.ndarray) -> np.ndarray:
    core = float(z - n_electrons + 1)
    r = np.asarray(r, dtype=float)
    if n_electrons == 1:
        return np.full_like(r, core)
    d, h = gsz_parameters(z, n_electrons)
    return core + (n_electrons - 1) * _omega(r, d, h)

def screened_potential(z: int, n_electrons: int) -> Callable[[np.ndarray], np.ndarray]:
    def v(r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        return -z_eff(z, n_electrons, r) / r
    return v

def screening_provenance(z: int, n_electrons: int) -> Provenance:
    return Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method=(
            "Green-Sellin-Zachor screened central potential, "
            "Szydlik-Green (1974) neutral-atom (d, K) parameters"
        ),
        assumptions=(
            f"an independent-particle central field for Z={z}, N={n_electrons}",
            "no self-consistency here; the potential depends only on (Z, N)",
            "the nuclear mass is taken as infinite (mu_ratio = 1)",
        ),
        error_estimate=None,
        refinement=(
            "the self-consistent Hartree-Fock in hf_atom.py removes this model error"
        ),
    )
