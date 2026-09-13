
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from atomic.analytic.angular import validate_angular
from atomic.analytic.hydrogen import validate_quantum_numbers
from atomic.analytic.wavefunction import evaluate_state
from atomic.atoms import Configuration
from atomic.hf_atom import evaluate_hf_state
from atomic.numerics.marching_tets import (
    Mesh,
    connected_components,
    enclosed_volume,
    marching_tets,
    surface_area,
)
from atomic.provenance import Fidelity, Provenance, Quantity
from atomic.screened_atom import evaluate_screened_state

GRID_SIZES = (48, 64, 96, 128)

_BOX_CAPTURE = 0.999

_TAIL_SAMPLES = 1024

_EVAL_CHUNK = 200_000

@dataclass(frozen=True)
class Isosurface:

    vertices: np.ndarray
    triangles: np.ndarray
    vertex_phase: np.ndarray
    target_fraction: float
    enclosed_fraction: Quantity
    level: Quantity
    escaped_fraction: Quantity
    mesh_volume: Quantity
    voxel_volume: Quantity
    area: Quantity
    components: int
    half_width: float
    resolution: int
    n: int
    l: int
    m: int
    Z: int
    basis: str
    label: str
    provenance: Provenance

    @property
    def outside_fraction(self) -> float:
        return 1.0 - self.enclosed_fraction.value

def default_half_width(n: int, Z: int = 1, mu_ratio: float = 1.0) -> float:
    return (2.0 * n * n + 6.0 * n) / (Z * mu_ratio)

def _density_on_grid(evaluator, half_width: float, resolution: int, progress=None):
    axis = np.linspace(-half_width, half_width, resolution)
    grid = np.zeros((resolution, resolution, resolution))
    assumptions: tuple[str, ...] = ()
    rows_per_chunk = max(1, _EVAL_CHUNK // (resolution * resolution))
    for i0 in range(0, resolution, rows_per_chunk):
        i1 = min(i0 + rows_per_chunk, resolution)
        xx, yy, zz = np.meshgrid(axis[i0:i1], axis, axis, indexing="ij")
        pos = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        psi = evaluator(pos)
        assumptions = psi.provenance.assumptions
        grid[i0:i1] = (np.abs(psi.values) ** 2).reshape(i1 - i0, resolution, resolution)
        if progress is not None:
            progress(i1 / resolution)
    return grid, axis, assumptions

def _cell_volume(half_width: float, resolution: int) -> float:
    return (2.0 * half_width / (resolution - 1)) ** 3

def _captured(grid: np.ndarray, half_width: float, resolution: int) -> float:
    return float(grid.sum() * _cell_volume(half_width, resolution))

def radial_mass(evaluator, l: int, m: int, r_max: float, samples: int = _TAIL_SAMPLES):
    n_theta = max(2, l + 2)
    n_phi = max(4, 2 * abs(m) + 2)
    u, w_u = np.polynomial.legendre.leggauss(n_theta)
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    sin_theta = np.sqrt(1.0 - u**2)

    directions = np.stack(
        [
            np.outer(sin_theta, np.cos(phi)).ravel(),
            np.outer(sin_theta, np.sin(phi)).ravel(),
            np.repeat(u, n_phi),
        ],
        axis=1,
    )
    weights = np.repeat(w_u, n_phi) / (2.0 * n_phi)

    r = np.linspace(0.0, float(r_max), samples + 1)
    pos = (r[:, None, None] * directions[None, :, :]).reshape(-1, 3)
    psi = evaluator(pos)
    sphere_mean = (np.abs(psi.values) ** 2).reshape(r.size, -1) @ weights
    shell = 4.0 * np.pi * r**2 * sphere_mean

    cumulative = np.zeros_like(r)
    cumulative[1:] = np.cumsum(0.5 * (shell[1:] + shell[:-1]) * np.diff(r))
    return r, cumulative

def _fit_box(evaluator, l: int, m: int, start: float, growths: int = 8):
    r_max = float(start)
    r, cumulative = radial_mass(evaluator, l, m, r_max)
    for _ in range(growths):
        if cumulative[-1] >= _BOX_CAPTURE:
            break
        r_max *= 1.6
        r, cumulative = radial_mass(evaluator, l, m, r_max)
    half_width = float(np.interp(_BOX_CAPTURE, cumulative, r))
    return half_width, r, cumulative

def solve_level(grid: np.ndarray, cell_volume: float, target: float) -> float:
    if not 0.0 < target < 1.0:
        raise ValueError(f"target fraction must be in (0, 1), got {target}")
    flat = np.sort(grid.reshape(-1))[::-1]
    cumulative = np.cumsum(flat) * cell_volume
    if cumulative[-1] < target:
        raise ValueError(
            f"the box holds {cumulative[-1]:.4f} of the probability, which cannot "
            f"enclose {target:.4f}: widen the box"
        )
    idx = int(np.searchsorted(cumulative, target))
    if idx == 0:
        return float(flat[0])
    span = cumulative[idx] - cumulative[idx - 1]
    t = 0.0 if span == 0.0 else (target - cumulative[idx - 1]) / span
    return float(flat[idx - 1] + t * (flat[idx] - flat[idx - 1]))

def fraction_above(grid: np.ndarray, cell_volume: float, level: float) -> float:
    return float(grid[grid >= level].sum() * cell_volume)

def _phase_at(evaluator, vertices: np.ndarray) -> np.ndarray:
    if vertices.shape[0] == 0:
        return np.zeros(0)
    return np.angle(evaluator(vertices).values)

def _build(
    evaluator,
    *,
    n: int,
    l: int,
    m: int,
    z: int,
    basis: str,
    target_fraction: float,
    resolution: int,
    half_width: float | None,
    start_half_width: float,
    fidelity: Fidelity,
    method_prefix: str,
    extra_assumptions: tuple[str, ...],
    refinement: str,
    progress: Callable[[float], None] | None,
) -> Isosurface:
    if resolution not in GRID_SIZES:
        raise ValueError(f"resolution must be one of {GRID_SIZES}, got {resolution}")
    if not 0.0 < target_fraction < 1.0:
        raise ValueError(f"target fraction must be in (0, 1), got {target_fraction}")

    requested = None if half_width is None else float(half_width)
    if requested is not None and requested <= 0.0:
        raise ValueError(f"half_width must be positive, got {requested}")

    probe = start_half_width if requested is None else max(start_half_width, requested)
    fitted, tail_r, tail_cumulative = _fit_box(evaluator, l, m, probe)
    half_width = fitted if requested is None else requested
    escaped = float(max(0.0, 1.0 - np.interp(half_width, tail_r, tail_cumulative)))
    normalization = float(tail_cumulative[-1])

    grid, axis, psi_assumptions = _density_on_grid(evaluator, half_width, resolution, progress)
    cell = _cell_volume(half_width, resolution)
    captured = _captured(grid, half_width, resolution)

    level = solve_level(grid, cell, target_fraction)
    achieved = fraction_above(grid, cell, level)

    spacing = float(axis[1] - axis[0])
    mesh: Mesh = marching_tets(grid, level, origin=(-half_width,) * 3, spacing=spacing)
    phase = _phase_at(evaluator, mesh.vertices)
    mesh_vol = enclosed_volume(mesh)
    voxel_vol = float((grid >= level).sum() * cell)
    components = connected_components(mesh)

    coarse = grid[::2, ::2, ::2]
    try:
        coarse_level = solve_level(coarse, 8.0 * cell, target_fraction)
        fraction_error = abs(fraction_above(grid, cell, coarse_level) - achieved)
        coarse_mesh = marching_tets(
            coarse, coarse_level, origin=(-half_width,) * 3, spacing=2.0 * spacing
        )
        volume_error = abs(enclosed_volume(coarse_mesh) - mesh_vol)
    except ValueError:
        fraction_error = None
        volume_error = None

    outside = 1.0 - achieved
    assumptions = psi_assumptions + extra_assumptions + (
        f"the surface encloses {achieved:.4f} of the electron, so the electron is "
        f"found outside the surface {outside:.1%} of the time: an orbital has no "
        f"boundary, and this contour is a choice of enclosed fraction, not an edge",
        f"{escaped:.2e} of the probability lies outside the sphere of radius "
        f"{half_width:g} bohr inscribed in the box, an upper bound on what no contour "
        f"drawn in the box can enclose",
        f"the {resolution}^3 sum of |psi|^2 dV over the box closes to {captured:.6f} "
        f"and the radial quadrature puts {normalization:.6f} of the state inside "
        f"{tail_r[-1]:g} bohr: each checks the other, and neither is assumed",
        f"mesh volume {mesh_vol:.4f} bohr^3 against {voxel_vol:.4f} bohr^3 counted "
        f"by cells above the level: the difference is the triangulation's discretization",
        f"the surface comes out in {components} piece{'' if components == 1 else 's'} on "
        f"a {spacing:.3f} bohr cell; a gap narrower than that is not resolved, so lobes "
        f"separated by less appear joined. A p orbital's lobes touch at the node, and "
        f"the looser the contour the nearer the node they part",
        (
            f"the level was solved on a {resolution}^3 grid; halving it moves the "
            f"enclosed fraction by {fraction_error:.2e} and the enclosed volume by "
            f"{volume_error:.2e} bohr^3 "
            f"({volume_error / mesh_vol if mesh_vol > 0 else float('nan'):.2%}). The second "
            f"number is the honest one for the shape: the level is nearly grid-"
            f"independent, so the fraction converges long before the surface does"
            if fraction_error is not None
            else f"the level was solved on a {resolution}^3 grid; the halved grid "
            f"could not hold {target_fraction:.4g} of the probability, so no "
            f"grid-halving estimate is quoted"
        ),
    )
    provenance = Provenance(
        fidelity=fidelity,
        method=(
            f"drew a {method_prefix} isosurface at the |psi|^2 level enclosing "
            f"{target_fraction:.4g} of the electron, by marching tetrahedra on a "
            f"{resolution}^3 grid of half-width {half_width:g} bohr"
        ),
        assumptions=assumptions,
        error_estimate=fraction_error,
        refinement=refinement,
    )

    def scalar(value, unit, label, error=None):
        return Quantity(
            value=float(value),
            unit=unit,
            label=label,
            provenance=Provenance(
                fidelity=fidelity,
                method=provenance.method,
                assumptions=assumptions,
                error_estimate=error,
                refinement=refinement,
            ),
        )

    return Isosurface(
        vertices=mesh.vertices,
        triangles=mesh.triangles,
        vertex_phase=phase,
        target_fraction=float(target_fraction),
        enclosed_fraction=scalar(achieved, "1", "enclosed probability", fraction_error),
        level=scalar(level, "bohr^-3", "|psi|^2 contour level"),
        escaped_fraction=scalar(escaped, "1", "probability outside the box"),
        mesh_volume=scalar(
            mesh_vol, "bohr^3", "volume enclosed by the surface", volume_error
        ),
        voxel_volume=scalar(voxel_vol, "bohr^3", "volume of cells above the level"),
        area=scalar(surface_area(mesh), "bohr^2", "surface area"),
        components=components,
        half_width=half_width,
        resolution=resolution,
        n=n,
        l=l,
        m=m,
        Z=z,
        basis=basis,
        label=f"|psi_{n},{l},{m}|^2 contour enclosing {target_fraction:.0%}",
        provenance=provenance,
    )

def isosurface(
    n: int,
    l: int,
    m: int,
    target_fraction: float = 0.9,
    basis: str = "complex",
    Z: int = 1,
    mu_ratio: float = 1.0,
    resolution: int = 96,
    half_width: float | None = None,
    progress: Callable[[float], None] | None = None,
) -> Isosurface:
    validate_quantum_numbers(n, l)
    validate_angular(l, m)
    if basis not in ("complex", "real"):
        raise ValueError(f"basis must be 'complex' or 'real', got {basis!r}")

    def evaluator(pos):
        return evaluate_state(n, l, m, pos, Z=Z, mu_ratio=mu_ratio, basis=basis)

    return _build(
        evaluator,
        n=n, l=l, m=m, z=Z, basis=basis,
        target_fraction=target_fraction,
        resolution=resolution,
        half_width=half_width,
        start_half_width=default_half_width(n, Z, mu_ratio),
        fidelity=Fidelity.NUMERICAL,
        method_prefix="closed-form psi_nlm",
        extra_assumptions=(
            "psi is exact; the grid, the level solve and the triangulation are not",
        ),
        refinement="raise the grid resolution or widen the box",
        progress=progress,
    )

def screened_isosurface(
    z: int,
    n_electrons: int,
    n: int,
    l: int,
    m: int,
    target_fraction: float = 0.9,
    basis: str = "complex",
    resolution: int = 96,
    half_width: float | None = None,
    progress: Callable[[float], None] | None = None,
) -> Isosurface:
    validate_quantum_numbers(n, l)
    validate_angular(l, m)

    def evaluator(pos):
        return evaluate_screened_state(z, n_electrons, n, l, m, pos, basis=basis)

    z_net = max(z - n_electrons + 1, 1)
    return _build(
        evaluator,
        n=n, l=l, m=m, z=z, basis=basis,
        target_fraction=target_fraction,
        resolution=resolution,
        half_width=half_width,
        start_half_width=default_half_width(n, z_net, 1.0),
        fidelity=Fidelity.APPROXIMATION,
        method_prefix="screened numerical psi_nlm",
        extra_assumptions=(
            "the enclosed fraction is exact for the screened model's psi, which is "
            "itself an approximation to the atom",
        ),
        refinement="raise the grid resolution, widen the box, or use Hartree-Fock",
        progress=progress,
    )

def hf_isosurface(
    z: int,
    n_electrons: int,
    n: int,
    l: int,
    m: int,
    target_fraction: float = 0.9,
    basis: str = "complex",
    resolution: int = 96,
    half_width: float | None = None,
    progress: Callable[[float], None] | None = None,
    *,
    config: Configuration | None = None,
    exchange: bool = True,
    pauli: bool = True,
) -> Isosurface:
    validate_quantum_numbers(n, l)
    validate_angular(l, m)

    def evaluator(pos):
        return evaluate_hf_state(
            z, n_electrons, n, l, m, pos,
            basis=basis, config=config, exchange=exchange, pauli=pauli,
        )

    fidelity = evaluator(np.zeros((1, 3))).provenance.fidelity
    counterfactual = fidelity is Fidelity.COUNTERFACTUAL
    z_net = max(z - n_electrons + 1, 1)
    return _build(
        evaluator,
        n=n, l=l, m=m, z=z, basis=basis,
        target_fraction=target_fraction,
        resolution=resolution,
        half_width=half_width,
        start_half_width=default_half_width(n, z_net, 1.0),
        fidelity=fidelity,
        method_prefix=(
            "Hartree psi_nlm" if counterfactual else "Hartree-Fock psi_nlm"
        ),
        extra_assumptions=(
            (
                "the enclosed fraction is exact for the orbital this solve "
                "produced, and that solve is a counterfactual: the fraction is "
                "not a statement about any real atom"
            )
            if counterfactual
            else (
                "the enclosed fraction is exact for the Hartree-Fock orbital, "
                "which neglects correlation"
            ),
        ),
        refinement=(
            "turn the altered rule back on"
            if counterfactual
            else "raise the grid resolution or widen the box"
        ),
        progress=progress,
    )
