
import math
from dataclasses import dataclass

import numpy as np
from scipy import constants as _sc

from atomic.provenance import Fidelity, Field, Provenance, Quantity

__all__ = [
    "AbsorbingLine",
    "AbsorptionSpectrum",
    "CurveOfGrowth",
    "SIGMA_INTEGRAL",
    "absorb",
    "absorption_spectrum",
    "cross_section",
    "curve_of_growth",
    "default_columns",
    "equivalent_width",
    "optical_depth",
    "transmission",
]

SIGMA_INTEGRAL: float = _sc.e**2 / (4.0 * _sc.epsilon_0 * _sc.m_e * _sc.c)

_SLAB = (
    "models a uniform absorbing slab: one temperature, one density, no depth "
    "structure",
    "pure absorption: no source function and no re-emission into the beam, so "
    "a line saturates but never reverses. A self-absorbed core needs a "
    "temperature gradient through a stratified atmosphere, which this is not",
    "no stimulated emission correction (a factor 1 - g_l N_u / g_u N_l), "
    "negligible unless the populations approach inversion",
    "no continuous opacity: the continuum is taken as flat and unabsorbed",
)

_SATURATED_PAD: float = 40.0

@dataclass(frozen=True)
class CurveOfGrowth:

    column_density: np.ndarray
    equivalent_width: np.ndarray
    regime: tuple[str, ...]
    slope: np.ndarray
    tau_centre: np.ndarray
    damping_parameter: float
    window_nm: float
    wavelength_nm: float
    oscillator_strength: float
    provenance: Provenance

def cross_section(
    oscillator_strength: float,
    wavelength_nm: float,
    profile: np.ndarray,
) -> np.ndarray:
    if oscillator_strength < 0.0:
        raise ValueError(f"f must be >= 0, got {oscillator_strength}")
    if wavelength_nm <= 0.0:
        raise ValueError(f"wavelength must be > 0, got {wavelength_nm}")
    lam_m = wavelength_nm * 1e-9
    return (
        SIGMA_INTEGRAL * oscillator_strength
        * np.asarray(profile, dtype=float) * 1e9
        * lam_m**2 / _sc.c
    )

def optical_depth(sigma: np.ndarray, column_density_m2: float) -> np.ndarray:
    if column_density_m2 < 0.0:
        raise ValueError(f"column density must be >= 0, got {column_density_m2}")
    return column_density_m2 * np.asarray(sigma, dtype=float)

def transmission(tau: np.ndarray) -> np.ndarray:
    return np.exp(-np.asarray(tau, dtype=float))

def equivalent_width(tau: np.ndarray, grid_nm: np.ndarray) -> Quantity:
    absorbed = 1.0 - transmission(tau)
    value = float(np.trapezoid(absorbed, grid_nm))
    return Quantity(
        value=value,
        unit="nm",
        label="equivalent width",
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method="W = integral (1 - exp(-tau)) dlambda over the grid",
            assumptions=_SLAB + (
                "W is instrument-independent by construction: convolving with "
                "a slit function redistributes flux inside the line and leaves "
                "the area removed from the continuum unchanged",
            ),
            refinement=(
                "a source function and a depth-stratified atmosphere would turn "
                "the line into one that can reverse as well as saturate"
            ),
        ),
    )

def _thin_limit_width(
    oscillator_strength: float, wavelength_nm: float, column_density_m2: float
) -> float:
    lam_m = wavelength_nm * 1e-9
    return (
        SIGMA_INTEGRAL / _sc.c * column_density_m2 * oscillator_strength * lam_m**2
    ) * 1e9

def _classify(tau_centre: float, damping_parameter: float) -> str:
    if tau_centre < 1.0:
        return "linear"
    if damping_parameter * tau_centre > 1.0:
        return "damping"
    return "saturated"

def default_columns(
    oscillator_strength: float,
    wavelength_nm: float,
    sigma_nm: float,
    gamma_nm: float,
    points: int = 70,
) -> np.ndarray:
    from atomic.broadening import voigt

    peak = float(cross_section(
        oscillator_strength, wavelength_nm,
        voigt(np.array([0.0]), sigma_nm, gamma_nm),
    )[0])
    if peak <= 0.0:
        raise ValueError("this line has no absorption cross-section")
    a = gamma_nm / (sigma_nm * math.sqrt(2.0)) if sigma_nm > 0 else 1.0
    n_thin = 1.0 / peak
    n_damp = n_thin / a if a > 0 else n_thin
    return np.geomspace(n_thin * 1e-5, n_damp * 1e4, points)

def curve_of_growth(
    oscillator_strength: float,
    wavelength_nm: float,
    sigma_nm: float,
    gamma_nm: float,
    columns_m2: np.ndarray,
    points: int = 4001,
    span_fwhm: float = 400.0,
) -> CurveOfGrowth:
    if sigma_nm <= 0.0 and gamma_nm <= 0.0:
        raise ValueError("a line with no width has no curve of growth")
    columns = np.asarray(columns_m2, dtype=float)
    if columns.ndim != 1 or columns.size < 2:
        raise ValueError("need at least two column densities")
    if np.any(columns <= 0.0):
        raise ValueError("column densities must be > 0 for a log-log curve")

    from atomic.broadening import voigt

    scale = max(2.3548 * sigma_nm, 2.0 * gamma_nm)

    def _sample(half: float):
        core = np.linspace(-3.0 * scale, 3.0 * scale, points // 2)
        wings = np.geomspace(3.0 * scale, half, points // 4)
        offs = np.unique(np.concatenate([core, wings, -wings]))
        phi = voigt(offs, sigma_nm, gamma_nm)
        s = cross_section(oscillator_strength, wavelength_nm, phi)
        g = wavelength_nm + offs
        w = np.array([
            equivalent_width(optical_depth(s, n), g).value for n in columns
        ])
        return offs, s, w

    half = span_fwhm * scale
    offsets, sigma_lambda, widths = _sample(half)
    for _ in range(12):
        if widths.max() <= 0.05 * (2.0 * half):
            break
        half *= 4.0
        offsets, sigma_lambda, widths = _sample(half)
    log_n, log_w = np.log10(columns), np.log10(np.maximum(widths, 1e-300))
    slope = np.gradient(log_w, log_n)
    sigma_peak = float(np.max(sigma_lambda))
    tau_centre = columns * sigma_peak
    a = gamma_nm / (sigma_nm * math.sqrt(2.0)) if sigma_nm > 0 else math.inf
    return CurveOfGrowth(
        column_density=columns,
        equivalent_width=widths,
        regime=tuple(_classify(t, a) for t in tau_centre),
        slope=slope,
        tau_centre=tau_centre,
        damping_parameter=a,
        window_nm=half,
        wavelength_nm=wavelength_nm,
        oscillator_strength=oscillator_strength,
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method=(
                "W(N) from tau = N sigma with "
                "sigma = (e^2/4 eps_0 m_e c) f phi lambda^2/c"
            ),
            assumptions=_SLAB + (
                f"integrated over +/-{half:.4g} nm ({half / scale:.0f} line "
                "widths), widening until the largest equivalent width was under "
                "5 percent of the window, so real Lorentzian wings carry the "
                "damping branch and not the edge of the integration",
                "the three branches are linear (slope 1, thin), saturated "
                "(slope ~0, black core, W grows only as sqrt(ln N)) and damping "
                "(slope 1/2, growth carried by the natural-width wings)",
                f"the branch is decided by the physics, not the slope: linear "
                f"while tau_centre < 1, damping once a tau_centre > 1 with the "
                f"Voigt damping parameter a = {a:.3g}, saturated in between",
            ),
            refinement=(
                "a real curve of growth is fitted to many lines of one species "
                "at once, which is how a Doppler width gets measured"
            ),
        ),
    )

@dataclass(frozen=True)
class AbsorbingLine:

    wavelength_nm: float
    label: str
    oscillator_strength: float
    lower_column_m2: float
    tau_centre: float
    regime: str
    thin_width_nm: float
    fwhm_nm: float

@dataclass(frozen=True)
class AbsorptionSpectrum:

    transmission: Field
    optical_depth: Field
    lines: tuple[AbsorbingLine, ...]
    column_density: Quantity
    equivalent_width: Quantity
    thin_limit_width: Quantity
    saturation: Quantity
    blends: tuple[tuple[str, str], ...]
    flux_closure: float

def _window_for(profiles, lo: float, hi: float) -> tuple[float, float]:
    from atomic.broadening import voigt

    reach = 0.0
    for p in profiles:
        peak = p.weight * float(voigt(np.zeros(1), p.sigma_nm, p.gamma_nm)[0])
        if peak <= 1.0:
            continue
        if p.sigma_nm > 0.0:
            reach = max(reach, p.sigma_nm * math.sqrt(2.0 * math.log(peak)))
        if p.gamma_nm > 0.0:
            reach = max(reach, math.sqrt(p.weight * p.gamma_nm / math.pi))
    pad = _SATURATED_PAD * reach
    return max(lo - pad, 0.5 * lo), hi + pad

def absorb(
    line_list,
    column_density_m2: float,
    emitter_mass: Quantity | None = None,
    hydrogenic: bool = True,
    resolving_power: float | None = None,
    window_nm: tuple[float, float] | None = None,
    max_points: int = 24_000,
) -> AbsorptionSpectrum:
    from atomic.broadening import synthesize, voigt
    from atomic.spectra import subshell_label

    if column_density_m2 < 0.0:
        raise ValueError(f"column density must be >= 0, got {column_density_m2}")
    usable = [
        ln for ln in line_list.lines
        if ln.oscillator_strength is not None and ln.lower_fraction is not None
    ]
    if not usable:
        raise ValueError(
            "absorption needs an oscillator strength and a lower-level "
            "population for every line: turn on intensities and supply thermal "
            "conditions. Without both there is nothing to absorb with, and the "
            "result would be a flat continuum drawn as if it were an answer"
        )

    def weight_fn(ln) -> float:
        if ln.oscillator_strength is None or ln.lower_fraction is None:
            return 0.0
        return _thin_limit_width(
            ln.oscillator_strength.value,
            ln.wavelength.value,
            column_density_m2 * ln.lower_fraction.value,
        )

    kwargs = dict(
        emitter_mass=emitter_mass,
        hydrogenic=hydrogenic,
        resolving_power=resolving_power,
        max_points=max_points,
        weight_fn=weight_fn,
        weight_label=("optical depth", "tau per nm"),
    )
    synth = synthesize(line_list, window_nm=window_nm, **kwargs)
    if window_nm is None:
        grid = synth.spectrum.grid
        wide = _window_for(synth.profiles, float(grid[0]), float(grid[-1]))
        if wide[0] < grid[0] or wide[1] > grid[-1]:
            synth = synthesize(line_list, window_nm=wide, **kwargs)

    grid = synth.spectrum.grid
    tau = np.clip(synth.spectrum.values, 0.0, None)
    trans = transmission(tau)
    measured = float(np.trapezoid(1.0 - trans, grid))

    detail: list[AbsorbingLine] = []
    for p, ln in zip(synth.profiles, synth.lines, strict=True):
        peak = p.weight * float(voigt(np.zeros(1), p.sigma_nm, p.gamma_nm)[0])
        a = (
            p.gamma_nm / (p.sigma_nm * math.sqrt(2.0))
            if p.sigma_nm > 0.0 else 1.0
        )
        detail.append(AbsorbingLine(
            wavelength_nm=p.wavelength_nm,
            label=subshell_label(ln),
            oscillator_strength=(
                ln.oscillator_strength.value
                if ln.oscillator_strength is not None else 0.0
            ),
            lower_column_m2=(
                column_density_m2 * ln.lower_fraction.value
                if ln.lower_fraction is not None else 0.0
            ),
            tau_centre=peak,
            regime=_classify(peak, a),
            thin_width_nm=p.weight,
            fwhm_nm=p.fwhm_nm,
        ))
    detail.sort(key=lambda d: d.wavelength_nm)

    thin_total = sum(d.thin_width_nm for d in detail)
    ratio = measured / thin_total if thin_total > 0.0 else 1.0

    blends: list[tuple[str, str]] = []
    for first, second in zip(detail, detail[1:], strict=False):
        gap = second.wavelength_nm - first.wavelength_nm
        if gap < first.fwhm_nm + second.fwhm_nm and min(
            first.tau_centre, second.tau_centre
        ) > 1e-3:
            blends.append((first.label, second.label))

    edge = float(max(1.0 - trans[0], 1.0 - trans[-1])) if trans.size else 0.0
    notes: list[str] = []
    if edge > 1e-3:
        notes.append(
            f"the spectrum is still absorbing {edge:.2%} of the continuum at "
            "the edge of the window, so the equivalent width below is an "
            "underestimate: widen the window or lower the column"
        )
    saturated = [d for d in detail if d.regime != "linear"]
    if saturated:
        notes.append(
            f"{len(saturated)} of {len(detail)} lines have a black core "
            "(tau at centre above 1), so their strengths no longer measure "
            "how much gas there is"
        )
    if blends:
        notes.append(
            f"{len(blends)} pair(s) of lines overlap within their own widths, "
            "so their transmissions multiply rather than their absorptions "
            "adding: the total is less than the sum of the parts by "
            "construction, not by approximation"
        )

    common = _SLAB + tuple(notes)
    return AbsorptionSpectrum(
        transmission=Field(
            values=trans,
            grid=grid,
            unit="I/I_0 (dimensionless)",
            grid_unit="nm (vacuum)",
            label="transmission",
            provenance=Provenance(
                fidelity=Fidelity.APPROXIMATION,
                method=(
                    "Beer-Lambert through a summed line list: "
                    "I/I_0 = exp(-sum_i N_i sigma_i(lambda))"
                ),
                assumptions=common + (
                    "one column density for the element; each line's own "
                    "lower-level fraction (LTE) selects its absorbers",
                    f"grid closure {synth.flux_closure:.4f}: the summed optical "
                    "depth integrates to this times the analytic total, measured "
                    "on the grid actually used",
                ),
                refinement=(
                    "a source function and a stratified atmosphere would let the "
                    "lines re-emit and reverse instead of only darkening"
                ),
            ),
        ),
        optical_depth=Field(
            values=tau,
            grid=grid,
            unit="dimensionless",
            grid_unit="nm (vacuum)",
            label="optical depth",
            provenance=Provenance(
                fidelity=Fidelity.APPROXIMATION,
                method="summed tau(lambda) = sum_i N_i sigma_i(lambda) over Voigt profiles",
                assumptions=common,
            ),
        ),
        lines=tuple(detail),
        column_density=Quantity(
            column_density_m2, "m^-2", "column density of the element",
            Provenance(
                fidelity=Fidelity.COUNTERFACTUAL,
                method="caller-chosen; a knob, not a measurement",
                assumptions=(
                    "counts atoms of the element per square metre of sight "
                    "line, neutral and ionized together",
                ),
            ),
        ),
        equivalent_width=Quantity(
            measured, "nm", "equivalent width of the whole spectrum",
            Provenance(
                fidelity=Fidelity.APPROXIMATION,
                method="W = integral (1 - I/I_0) dlambda over the full grid",
                assumptions=common,
            ),
        ),
        thin_limit_width=Quantity(
            thin_total, "nm", "summed thin-limit widths",
            Provenance(
                fidelity=Fidelity.APPROXIMATION,
                method="sum_i (e^2/4 eps_0 m_e c^2) N_i f_i lambda_i^2",
                assumptions=_SLAB + (
                    "what the lines would remove if none saturated and none "
                    "overlapped; an upper bound, not a prediction",
                ),
            ),
        ),
        saturation=Quantity(
            ratio, "dimensionless", "measured / thin-limit width",
            Provenance(
                fidelity=Fidelity.APPROXIMATION,
                method="W_measured / sum_i W_thin,i",
                assumptions=_SLAB + (
                    "1 means every line is optically thin and the spectrum is "
                    "a faithful census of the gas; below 1 means it is not, "
                    "and this is how much of the census is being lost",
                    "saturation and blending are folded together: both make the "
                    "whole absorb less than the sum of its lines, and on a "
                    "single grid they cannot be separated",
                ),
            ),
        ),
        blends=tuple(blends),
        flux_closure=synth.flux_closure,
    )

def absorption_spectrum(
    grid_nm: np.ndarray,
    sigma: np.ndarray,
    column_density_m2: float,
    label: str = "transmission",
) -> Field:
    tau = optical_depth(sigma, column_density_m2)
    peak = float(np.max(tau)) if tau.size else 0.0
    return Field(
        values=transmission(tau),
        grid=np.asarray(grid_nm, dtype=float),
        unit="I/I_0 (dimensionless)",
        grid_unit="nm (vacuum)",
        label=label,
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method="Beer-Lambert: I/I_0 = exp(-N sigma(lambda))",
            assumptions=_SLAB + (
                f"peak optical depth tau = {peak:.4g}"
                + (
                    ": optically thin, so the line depth is proportional to the "
                    "column and the strength still measures the amount of gas"
                    if peak < 0.5
                    else ": the core is saturated, so adding gas barely deepens "
                    "the line and its strength no longer measures the column"
                ),
            ),
        ),
    )
