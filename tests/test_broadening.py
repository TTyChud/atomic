
import math

import numpy as np
import pytest
from scipy import constants as _sc
from scipy import integrate

from atomic.analytic.transitions import einstein_A
from atomic.broadening import (
    doppler_sigma_nm,
    instrumental_sigma_nm,
    level_decay_rates,
    natural_gamma_nm,
    stark_span_estimate,
    synthesize,
    voigt,
    voigt_fwhm,
)
from atomic.populations import ThermalConditions
from atomic.spectra import transition_lines
from atomic.systems import emitter_mass, get_system, hydrogen_like

M_E = _sc.m_e

def test_hydrogen_atom_mass_is_proton_plus_electron():
    m = emitter_mass(get_system("h"))
    ratio = m.value / M_E
    proton_ratio = _sc.physical_constants["proton-electron mass ratio"][0]
    assert ratio == pytest.approx(proton_ratio + 1.0, rel=1e-9)
    assert ratio == pytest.approx(1837.15, rel=1e-4)

def test_positronium_mass_is_exactly_two_electrons():
    m = emitter_mass(get_system("ps"))
    assert m.value / M_E == pytest.approx(2.0, rel=1e-12)

def test_muonic_hydrogen_mass_is_muon_plus_proton():
    m = emitter_mass(get_system("mu-h"))
    muon = _sc.physical_constants["muon-electron mass ratio"][0]
    proton = _sc.physical_constants["proton-electron mass ratio"][0]
    assert m.value / M_E == pytest.approx(muon + proton, rel=1e-9)

def test_infinite_nucleus_gives_infinite_mass_and_says_so():
    m = emitter_mass(hydrogen_like(3))
    assert math.isinf(m.value)
    assert any("infinit" in a.lower() for a in m.provenance.assumptions)
    assert doppler_sigma_nm(500.0, 1e4, m.value) == 0.0

def test_lyman_alpha_natural_width_is_the_textbook_100_mhz():
    a = einstein_A(2, 1, 1, 0).value
    assert a == pytest.approx(6.2649e8, rel=2e-3)
    fwhm_hz = a / (2.0 * math.pi)
    assert fwhm_hz == pytest.approx(99.7e6, rel=5e-3)

def test_lyman_alpha_natural_width_in_nm():
    gamma = natural_gamma_nm(6.2649e8, 121.567)
    assert 2.0 * gamma == pytest.approx(4.915e-6, rel=1e-3)

def test_natural_width_is_zero_without_a_decay_channel():
    assert natural_gamma_nm(0.0, 121.567) == 0.0

def test_natural_width_scales_as_lambda_squared():
    a = natural_gamma_nm(1e8, 200.0)
    b = natural_gamma_nm(1e8, 400.0)
    assert b / a == pytest.approx(4.0, rel=1e-12)

def test_h_alpha_doppler_width_at_10000_k():
    m = emitter_mass(get_system("h"))
    sigma = doppler_sigma_nm(656.28, 1e4, m.value)
    fwhm = 2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma
    assert fwhm == pytest.approx(0.0468, rel=5e-3)

def test_doppler_width_grows_as_sqrt_t():
    m = emitter_mass(get_system("h")).value
    assert doppler_sigma_nm(500.0, 4e4, m) / doppler_sigma_nm(
        500.0, 1e4, m
    ) == pytest.approx(2.0, rel=1e-12)

def test_positronium_doppler_is_thirty_times_hydrogen():
    h = doppler_sigma_nm(500.0, 1e4, emitter_mass(get_system("h")).value)
    ps = doppler_sigma_nm(500.0, 1e4, emitter_mass(get_system("ps")).value)
    assert ps / h == pytest.approx(math.sqrt(1837.15 / 2.0), rel=1e-3)

def test_doppler_rejects_nonpositive_temperature():
    with pytest.raises(ValueError, match="temperature"):
        doppler_sigma_nm(500.0, 0.0, 1e-27)

def test_resolving_power_gives_the_width_it_promises():
    lam, r = 500.0, 20000.0
    fwhm = 2.0 * math.sqrt(2.0 * math.log(2.0)) * instrumental_sigma_nm(lam, r)
    assert lam / fwhm == pytest.approx(r, rel=1e-12)

def test_voigt_reduces_to_a_gaussian_when_gamma_is_zero():
    x = np.linspace(-5.0, 5.0, 201)
    sigma = 1.3
    expected = np.exp(-(x**2) / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))
    assert np.allclose(voigt(x, sigma, 0.0), expected, rtol=1e-12, atol=1e-15)

def test_voigt_reduces_to_a_lorentzian_when_sigma_is_zero():
    x = np.linspace(-5.0, 5.0, 201)
    gamma = 0.7
    expected = gamma / (math.pi * (x**2 + gamma**2))
    assert np.allclose(voigt(x, 0.0, gamma), expected, rtol=1e-12)

def test_voigt_approaches_the_lorentzian_continuously():
    x = np.linspace(-5.0, 5.0, 101)
    exact = voigt(x, 0.0, 0.7)
    nearly = voigt(x, 1e-7, 0.7)
    assert np.allclose(nearly, exact, rtol=1e-6, atol=1e-9)

@pytest.mark.parametrize(
    ("sigma", "gamma"), [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.2, 3.0), (3.0, 0.2)]
)
def test_voigt_area_is_one(sigma, gamma):
    area, _ = integrate.quad(
        lambda x: float(voigt(np.array([x]), sigma, gamma)[0]),
        -np.inf, np.inf, limit=400,
    )
    assert area == pytest.approx(1.0, rel=1e-6)

def test_voigt_refuses_a_line_with_no_width():
    with pytest.raises(ValueError, match="no width"):
        voigt(np.array([0.0]), 0.0, 0.0)

def test_voigt_fwhm_matches_a_direct_measurement():
    for sigma, gamma in [(1.0, 0.3), (0.3, 1.0), (1.0, 1.0), (2.0, 0.05)]:
        peak = float(voigt(np.array([0.0]), sigma, gamma)[0])
        x = np.linspace(0.0, 40.0 * (sigma + gamma), 400001)
        y = voigt(x, sigma, gamma)
        half = x[np.argmin(np.abs(y - peak / 2.0))]
        assert 2.0 * half == pytest.approx(voigt_fwhm(sigma, gamma), rel=2e-3)

def test_gaussian_and_lorentzian_fwhm_limits():
    assert voigt_fwhm(1.0, 0.0) == pytest.approx(2.3548, rel=1e-4)
    assert voigt_fwhm(0.0, 1.0) == pytest.approx(2.0, rel=1e-3)

def test_level_decay_rate_matches_the_analytic_lifetime():
    from atomic.analytic.transitions import lifetime

    sys_ = get_system("h")
    mu = sys_.mu_ratio.value
    lines = transition_lines(sys_, n_max=6, intensities=True)
    rates = level_decay_rates(lines.lines)
    for (n, l) in [(2, 1), (3, 1), (3, 2), (4, 0), (5, 3)]:
        assert rates[(n, l, None)] == pytest.approx(
            1.0 / lifetime(n, l, Z=sys_.Z, mu_ratio=mu).value, rel=1e-9
        )

def test_decay_rates_carry_the_reduced_mass():
    from atomic.analytic.transitions import lifetime

    sys_ = get_system("h")
    rates = level_decay_rates(
        transition_lines(sys_, n_max=3, intensities=True).lines
    )
    infinite_mass = 1.0 / lifetime(2, 1).value
    assert rates[(2, 1, None)] / infinite_mass == pytest.approx(
        sys_.mu_ratio.value, rel=1e-6
    )

def test_ground_state_has_no_decay_rate():
    lines = transition_lines(get_system("h"), n_max=4, intensities=True)
    rates = level_decay_rates(lines.lines)
    assert (1, 0, None) not in rates

def test_2s_has_no_e1_channel_so_no_natural_width():
    lines = transition_lines(get_system("h"), n_max=4, intensities=True)
    rates = level_decay_rates(lines.lines)
    assert (2, 0, None) not in rates

def test_stark_estimate_matches_the_griem_scaling_for_h_beta():
    griem = 2.0 * (1e14 / 1e17) ** (2.0 / 3.0)
    est = stark_span_estimate(4, 2, 486.1, 1e14)
    assert 0.5 < est.value / griem < 2.5
    assert est.value == pytest.approx(0.034, rel=0.1)

def test_stark_estimate_scales_as_density_to_the_two_thirds():
    a = stark_span_estimate(4, 2, 486.1, 1e14).value
    b = stark_span_estimate(4, 2, 486.1, 1e17).value
    assert b / a == pytest.approx(1000.0 ** (2.0 / 3.0), rel=1e-9)

def test_stark_estimate_declares_it_is_not_a_fwhm():
    est = stark_span_estimate(4, 2, 486.1, 1e14)
    assert any("NOT a FWHM" in a for a in est.provenance.assumptions)
    assert any("hydrogenic only" in a for a in est.provenance.assumptions)

def _hydrogen_thermal(n_max=5, t=1e4, ne=1e12):
    return transition_lines(
        get_system("h"), n_max=n_max, intensities=True,
        thermal=ThermalConditions(temperature_k=t, electron_density_cm3=ne),
    )

def test_synthesis_conserves_the_line_strengths():
    lines = _hydrogen_thermal()
    syn = synthesize(lines, emitter_mass=emitter_mass(get_system("h")))
    total = sum(p.weight for p in syn.profiles)
    integral = np.trapezoid(syn.spectrum.values, syn.spectrum.grid)
    assert integral == pytest.approx(total, rel=2e-3)
    assert syn.flux_closure == pytest.approx(integral / total, rel=1e-12)
    assert syn.flux_closure == pytest.approx(1.0, rel=2e-3)

def test_flux_closure_survives_a_lorentzian_dominated_spectrum():
    lines = transition_lines(get_system("h"), n_max=5, intensities=True)
    syn = synthesize(lines)
    assert syn.weight_kind == "rate"
    assert all(p.sigma_nm == 0.0 for p in syn.profiles)
    assert syn.flux_closure == pytest.approx(1.0, rel=5e-3)

def test_a_long_line_list_stays_accurate_and_quick():
    import time

    lines = transition_lines(
        get_system("h"), n_max=10, fine_structure=True, intensities=True,
        thermal=ThermalConditions(temperature_k=1e4, electron_density_cm3=1e12),
    )
    assert len(lines.lines) > 800
    start = time.perf_counter()
    syn = synthesize(lines, emitter_mass=emitter_mass(get_system("h")))
    elapsed = time.perf_counter() - start
    assert syn.flux_closure == pytest.approx(1.0, rel=5e-3)
    assert syn.spectrum.grid.size <= 24_000
    assert elapsed < 5.0

def test_fine_structure_window_never_goes_negative():
    lines = transition_lines(
        get_system("h"), n_max=6, fine_structure=True, intensities=True,
        thermal=ThermalConditions(temperature_k=1e4, electron_density_cm3=1e12),
    )
    syn = synthesize(lines, emitter_mass=emitter_mass(get_system("h")))
    assert syn.spectrum.grid.min() > 0.0
    assert np.all(np.isfinite(syn.spectrum.grid))
    assert np.all(np.isfinite(syn.spectrum.values))

def test_dropped_wing_flux_is_measured_and_tiny():
    syn = synthesize(
        _hydrogen_thermal(), emitter_mass=emitter_mass(get_system("h"))
    )
    note = [a for a in syn.spectrum.provenance.assumptions if "wings dropped" in a]
    assert note, "the wing cut must be disclosed, not assumed harmless"
    assert "not\nassumed" in note[0] or "not assumed" in note[0].replace("\n", " ")

def test_flux_closure_is_reported_in_the_provenance():
    syn = synthesize(
        _hydrogen_thermal(), emitter_mass=emitter_mass(get_system("h"))
    )
    assert any(
        "integrates to" in a for a in syn.spectrum.provenance.assumptions
    )

def test_every_line_centre_is_a_grid_point():
    lines = _hydrogen_thermal()
    syn = synthesize(lines, emitter_mass=emitter_mass(get_system("h")))
    grid = syn.spectrum.grid
    for p in syn.profiles:
        assert np.min(np.abs(grid - p.wavelength_nm)) < 1e-12 * p.wavelength_nm

def test_hotter_gas_gives_wider_lines():
    m = emitter_mass(get_system("h"))
    cool = synthesize(_hydrogen_thermal(t=5e3), emitter_mass=m)
    hot = synthesize(_hydrogen_thermal(t=2e4), emitter_mass=m)
    ratio = hot.profiles[0].fwhm_nm / cool.profiles[0].fwhm_nm
    assert ratio == pytest.approx(2.0, rel=0.05)

def test_doppler_dominates_natural_at_10000_k():
    syn = synthesize(
        _hydrogen_thermal(), emitter_mass=emitter_mass(get_system("h"))
    )
    optical = [p for p in syn.profiles if 400 < p.wavelength_nm < 700]
    assert optical
    for p in optical:
        assert p.sigma_nm > 100.0 * p.gamma_nm

def test_instrument_widens_every_line():
    m = emitter_mass(get_system("h"))
    lines = _hydrogen_thermal()
    sharp = synthesize(lines, emitter_mass=m)
    blurred = synthesize(lines, emitter_mass=m, resolving_power=2000.0)
    for a, b in zip(sharp.profiles, blurred.profiles, strict=True):
        assert b.fwhm_nm > a.fwhm_nm
        assert "instrumental" in b.terms

def test_no_width_source_refuses_to_draw():
    lines = transition_lines(get_system("h"), n_max=3, intensities=False)
    with pytest.raises(ValueError, match="zero width"):
        synthesize(lines)

def test_uniform_weighting_when_no_strengths_are_available():
    lines = transition_lines(get_system("h"), n_max=3, intensities=False)
    syn = synthesize(lines, resolving_power=5000.0)
    assert syn.weight_kind == "uniform"
    assert all(p.weight == 1.0 for p in syn.profiles)
    assert syn.spectrum.unit == "per nm"

def test_rate_weighting_without_thermal_conditions():
    lines = transition_lines(get_system("h"), n_max=4, intensities=True)
    syn = synthesize(lines, resolving_power=5000.0)
    assert syn.weight_kind == "rate"
    assert syn.spectrum.unit == "s^-1 per nm"

def test_stark_note_fires_at_high_density_and_not_at_low():
    m = emitter_mass(get_system("h"))
    thin = synthesize(_hydrogen_thermal(ne=1e8), emitter_mass=m)
    dense = synthesize(_hydrogen_thermal(ne=1e17), emitter_mass=m)
    assert thin.stark_note is None
    assert dense.stark_note is not None
    assert "Stark" in dense.stark_note

def test_synthesis_discloses_what_it_leaves_out():
    syn = synthesize(
        _hydrogen_thermal(), emitter_mass=emitter_mass(get_system("h"))
    )
    text = " ".join(syn.spectrum.provenance.assumptions)
    assert "collisional" in text
    assert "self-absorption" in text
    assert "two-photon" in text

def test_curve_is_finite_and_nonnegative():
    syn = synthesize(
        _hydrogen_thermal(), emitter_mass=emitter_mass(get_system("h"))
    )
    v = syn.spectrum.values
    assert np.all(np.isfinite(v))
    assert np.all(v >= 0.0)
    assert v.max() > 0.0

def test_fully_ionized_gas_gives_a_flat_zero_curve():
    lines = _hydrogen_thermal(t=3e5, ne=1e4)
    assert lines.thermal.ionized_fraction.value == pytest.approx(1.0)
    assert sum(ln.emissivity.value for ln in lines.lines) == 0.0
    syn = synthesize(lines, emitter_mass=emitter_mass(get_system("h")))
    assert np.all(syn.spectrum.values == 0.0)
    assert syn.flux_closure == 1.0
    assert any(
        "fully ionized" in a for a in syn.spectrum.provenance.assumptions
    )

def test_window_restricts_the_lines_and_the_grid():
    syn = synthesize(
        _hydrogen_thermal(), emitter_mass=emitter_mass(get_system("h")),
        window_nm=(400.0, 700.0),
    )
    assert syn.spectrum.grid.min() >= 400.0
    assert syn.spectrum.grid.max() <= 700.0
    assert all(400.0 <= p.wavelength_nm <= 700.0 for p in syn.profiles)
