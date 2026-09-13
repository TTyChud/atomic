import dataclasses
from dataclasses import dataclass

import numpy as np

from atomic.atoms import Configuration, aufbau_configuration
from atomic.hf_atom import hf_total_radial_density
from atomic.provenance import Fidelity, Field, Provenance, Quantity
from atomic.screened_atom import screened_total_radial_density

_WEAKNESS = {
    Fidelity.EXACT: 0,
    Fidelity.NUMERICAL: 1,
    Fidelity.APPROXIMATION: 2,
    Fidelity.COUNTERFACTUAL: 3,
}

def _weaker(a: Fidelity, b: Fidelity) -> Fidelity:
    return a if _WEAKNESS[a] >= _WEAKNESS[b] else b

def _common_grid(a: Field, b: Field, points: int) -> np.ndarray:
    lo = max(a.grid[0], b.grid[0])
    hi = min(a.grid[-1], b.grid[-1])
    if not lo < hi:
        raise ValueError(
            f"the two solver boxes do not overlap: [{a.grid[0]:.3g}, "
            f"{a.grid[-1]:.3g}] and [{b.grid[0]:.3g}, {b.grid[-1]:.3g}]"
        )
    return np.geomspace(lo, hi, points)

def _resample(f: Field, grid: np.ndarray) -> Field:
    return dataclasses.replace(
        f,
        values=np.interp(grid, f.grid, f.values),
        grid=grid,
        provenance=dataclasses.replace(
            f.provenance,
            method=f.provenance.method
            + "; resampled by linear interpolation onto the common comparison grid",
        ),
    )

def _displaced_charge(grid: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.trapezoid(np.abs(a - b), grid))

def _window_loss(f: Field, grid: np.ndarray) -> float:
    inside = np.trapezoid(np.interp(grid, f.grid, f.values), grid)
    return abs(float(np.trapezoid(f.values, f.grid) - inside))

SHELL_LABELS = ("K", "L", "M", "N", "O")

@dataclass(frozen=True)
class ShellPeak:
    label: str
    gsz_radius: float | None
    hf_radius: float | None
    gsz_depth: float | None
    hf_depth: float | None

@dataclass(frozen=True)
class DensityComparison:
    grid: np.ndarray
    gsz: Field
    hf: Field
    displaced_charge: Quantity
    shells: tuple[ShellPeak, ...]
    provenance: Provenance

_NOISE_FLOOR = 1e-8

def _peaks_with_depth(
    grid: np.ndarray, values: np.ndarray
) -> list[tuple[float, float | None]]:
    floor = _NOISE_FLOOR * float(np.max(values))
    big = values > floor
    maxima = [
        i
        for i in range(1, len(values) - 1)
        if big[i] and values[i] > values[i - 1] and values[i] >= values[i + 1]
    ]
    minima = [
        i
        for i in range(1, len(values) - 1)
        if values[i] < values[i - 1] and values[i] <= values[i + 1]
    ]
    out: list[tuple[float, float | None]] = []
    for i in maxima:
        before = [j for j in minima if j < i]
        if not before:
            out.append((float(grid[i]), None))
            continue
        valley = values[before[-1]]
        out.append((float(grid[i]), float((values[i] - valley) / values[i])))
    return out

def _shell_table(
    grid: np.ndarray, gsz: np.ndarray, hf: np.ndarray, config: Configuration
) -> tuple[ShellPeak, ...]:
    n_shells = len({n for (n, _), _ in config})
    peaks = {"GSZ": _peaks_with_depth(grid, gsz), "HF": _peaks_with_depth(grid, hf)}
    for name, found in peaks.items():
        if len(found) > n_shells:
            radii = ", ".join(f"{r:.4g}" for r, _ in found)
            raise ValueError(
                f"the {name} density has {len(found)} maxima at r = {radii} bohr "
                f"but the configuration occupies only {n_shells} shells, so that "
                f"is an unresolved orbital and not a shell"
            )
    rows = []
    for i in range(n_shells):
        g = peaks["GSZ"][i] if i < len(peaks["GSZ"]) else (None, None)
        h = peaks["HF"][i] if i < len(peaks["HF"]) else (None, None)
        rows.append(
            ShellPeak(
                label=SHELL_LABELS[i],
                gsz_radius=g[0], gsz_depth=g[1],
                hf_radius=h[0], hf_depth=h[1],
            )
        )
    return tuple(rows)

def compare_total_densities(
    z: int,
    n_electrons: int,
    *,
    config: Configuration | None = None,
    exchange: bool = True,
    pauli: bool = True,
    points: int = 800,
) -> DensityComparison:
    if n_electrons != z:
        raise ValueError(
            f"the GSZ screening parameters are fitted to neutral atoms, so this "
            f"comparison needs N = Z; got Z={z}, N={n_electrons}"
        )
    cfg = aufbau_configuration(n_electrons, pauli) if config is None else config
    hf = hf_total_radial_density(
        z, n_electrons, config=cfg, exchange=exchange, pauli=pauli, points=points
    )
    gsz = screened_total_radial_density(z, n_electrons, config=cfg, points=points)

    grid = _common_grid(gsz, hf, points)
    gsz_r = _resample(gsz, grid)
    hf_r = _resample(hf, grid)
    displaced = _displaced_charge(grid, gsz_r.values, hf_r.values)

    bar = (
        (hf.provenance.error_estimate or 0.0)
        + (gsz.provenance.error_estimate or 0.0)
        + _window_loss(gsz, grid)
        + _window_loss(hf, grid)
    )
    fidelity = _weaker(gsz.provenance.fidelity, hf.provenance.fidelity)
    altered = [
        name
        for name, on in (("exchange", exchange), ("the occupancy cap", pauli))
        if not on
    ]
    method = (
        "half the L1 norm of D_HF - D_GSZ on the common log grid, in electrons"
    )
    if altered:
        method += f"; {' and '.join(altered)} switched off on the Hartree-Fock side"
    provenance = Provenance(
        fidelity=fidelity,
        method=method,
        assumptions=(
            "both densities are resampled by linear interpolation onto a shared grid",
            "the window is the intersection of the two solver boxes, and the "
            "charge outside it is measured and folded into the error estimate",
            "Hartree-Fock is not truth: this is the distance between two "
            "approximations, never the error in one of them",
        ),
        error_estimate=bar,
        refinement=(
            "a correlated method (configuration interaction, coupled cluster) "
            "would give both models a reference to be measured against, rather "
            "than only against each other"
        ),
    )
    return DensityComparison(
        grid=grid,
        gsz=gsz_r,
        hf=hf_r,
        displaced_charge=Quantity(
            value=displaced,
            unit="electrons",
            label="charge the two models place differently",
            provenance=provenance,
        ),
        shells=_shell_table(grid, gsz_r.values, hf_r.values, cfg),
        provenance=provenance,
    )
