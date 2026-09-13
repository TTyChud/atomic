
import math
from dataclasses import replace

import numpy as np
import pytest
from scipy import constants as sc

from atomic.broadening import (
    doppler_sigma_nm,
    level_decay_rates,
    natural_gamma_nm,
    voigt,
)
from atomic.populations import ThermalConditions
from atomic.provenance import Fidelity
from atomic.spectra import transition_lines
from atomic.systems import emitter_mass, get_system
from atomic.transfer import (
    SIGMA_INTEGRAL,
    absorb,
    cross_section,
    equivalent_width,
    optical_depth,
)

HOT = ThermalConditions(temperature_k=8000.0, electron_density_cm3=1e12)

def _hydrogen(n_max: int = 4, thermal: ThermalConditions | None = HOT):
    return transition_lines(
        get_system("h"), n_max=n_max, intensities=True, thermal=thermal
    )

def _mass():
    return emitter_mass(get_system("h"))

def _only(line_list, wavelength_nm: float):
    pick = min(line_list.lines, key=lambda ln: abs(ln.wavelength.value - wavelength_nm))
    return replace(line_list, lines=(pick,))

def test_thin_limit_recovers_the_analytic_sum():
    result = absorb(_hydrogen(), 1e16, emitter_mass=_mass())
    assert result.equivalent_width.value == pytest.approx(
        result.thin_limit_width.value, rel=0.01
    )
    assert result.saturation.value == pytest.approx(1.0, abs=0.01)
    assert all(d.regime == "linear" for d in result.lines)

def test_thin_limit_is_linear_in_the_column():
    a = absorb(_hydrogen(), 1e15, emitter_mass=_mass())
    b = absorb(_hydrogen(), 2e15, emitter_mass=_mass())
    assert b.equivalent_width.value == pytest.approx(
        2.0 * a.equivalent_width.value, rel=0.01
    )

def test_single_line_matches_the_phase_19_chain():
    line_list = _only(_hydrogen(), 121.567)
    line = line_list.lines[0]
    lam = line.wavelength.value
    column = 1e19

    rates = level_decay_rates(line_list.lines)
    gamma = natural_gamma_nm(
        rates.get((line.n_upper, line.l_upper, line.j_upper), 0.0)
        + rates.get((line.n_lower, line.l_lower, line.j_lower), 0.0),
        lam,
    )
    sigma = doppler_sigma_nm(lam, HOT.temperature_k, _mass().value)

    span = 200.0 * max(sigma, gamma)
    grid = np.linspace(lam - span, lam + span, 400_001)
    tau = optical_depth(
        cross_section(line.oscillator_strength.value, lam, voigt(grid - lam, sigma, gamma)),
        column * line.lower_fraction.value,
    )
    reference = equivalent_width(tau, grid)

    result = absorb(line_list, column, emitter_mass=_mass())
    assert result.equivalent_width.value == pytest.approx(reference.value, rel=0.02)

def test_saturation_drives_the_width_below_the_thin_sum():
    result = absorb(_hydrogen(), 1e21, emitter_mass=_mass())
    assert result.saturation.value < 0.5
    assert result.equivalent_width.value < result.thin_limit_width.value
    assert any(d.regime != "linear" for d in result.lines)
    assert any(
        "black core" in a for a in result.transmission.provenance.assumptions
    )

def test_equivalent_width_never_decreases_with_column():
    widths = [
        absorb(_hydrogen(), n, emitter_mass=_mass()).equivalent_width.value
        for n in np.geomspace(1e15, 1e24, 12)
    ]
    assert all(b >= a for a, b in zip(widths, widths[1:], strict=False))

def test_blended_lines_absorb_less_than_the_sum_of_the_parts():
    single = _only(_hydrogen(), 121.567)
    line = single.lines[0]
    shifted = replace(
        line,
        wavelength=replace(line.wavelength, value=line.wavelength.value * (1 + 2e-6)),
    )
    pair = replace(single, lines=(line, shifted))

    column = 3e19
    one = absorb(single, column, emitter_mass=_mass())
    two = absorb(pair, column, emitter_mass=_mass())

    assert one.equivalent_width.value < two.equivalent_width.value
    assert two.equivalent_width.value < 2.0 * one.equivalent_width.value
    assert two.blends

def test_transmission_stays_inside_zero_and_one():
    for column in (1e14, 1e20, 1e25):
        values = absorb(_hydrogen(), column, emitter_mass=_mass()).transmission.values
        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0 + 1e-12)

def test_lyman_is_opaque_where_balmer_is_transparent():
    result = absorb(_hydrogen(), 1e20, emitter_mass=_mass())
    lyman = min(result.lines, key=lambda d: abs(d.wavelength_nm - 121.567))
    balmer = min(result.lines, key=lambda d: abs(d.wavelength_nm - 656.3))

    assert lyman.tau_centre > 1.0
    assert balmer.tau_centre < 1.0
    assert lyman.lower_column_m2 > 1e3 * balmer.lower_column_m2

def test_regimes_are_classified_by_optical_depth():
    result = absorb(_hydrogen(), 1e23, emitter_mass=_mass())
    assert {d.regime for d in result.lines} & {"saturated", "damping"}
    for d in result.lines:
        if d.regime == "linear":
            assert d.tau_centre < 1.0
        else:
            assert d.tau_centre >= 1.0

def test_ionizing_the_gas_makes_it_transparent():
    thin = ThermalConditions(temperature_k=8_000.0, electron_density_cm3=1e8)
    ionized = ThermalConditions(temperature_k=200_000.0, electron_density_cm3=1e8)
    cool = absorb(_hydrogen(thermal=thin), 1e20, emitter_mass=_mass())
    hot = absorb(_hydrogen(thermal=ionized), 1e20, emitter_mass=_mass())

    assert hot.equivalent_width.value < 1e-6 * cool.equivalent_width.value
    assert np.all(hot.transmission.values > 1.0 - 1e-6)
    assert hot.saturation.value == pytest.approx(1.0, abs=0.01)
    assert all(d.regime == "linear" for d in hot.lines)

def test_window_grows_with_the_column():
    narrow = absorb(_hydrogen(), 1e16, emitter_mass=_mass())
    wide = absorb(_hydrogen(), 1e24, emitter_mass=_mass())
    narrow_span = narrow.transmission.grid[-1] - narrow.transmission.grid[0]
    wide_span = wide.transmission.grid[-1] - wide.transmission.grid[0]
    assert wide_span > narrow_span

def test_edge_absorption_is_measured_and_disclosed():
    result = absorb(
        _hydrogen(), 1e23, emitter_mass=_mass(), window_nm=(121.0, 122.0)
    )
    assert any(
        "still absorbing" in a for a in result.transmission.provenance.assumptions
    )

def test_a_wide_enough_window_does_not_claim_edge_absorption():
    result = absorb(_hydrogen(), 1e18, emitter_mass=_mass())
    assert not any(
        "still absorbing" in a for a in result.transmission.provenance.assumptions
    )
    assert result.flux_closure == pytest.approx(1.0, abs=0.02)

def test_degenerate_lines_keep_their_own_oscillator_strengths():
    result = absorb(_hydrogen(), 1e20, emitter_mass=_mass())
    assert len(result.lines) == len(_hydrogen().lines)

    balmer_alpha = [d for d in result.lines if abs(d.wavelength_nm - 656.4696) < 1e-3]
    assert len(balmer_alpha) == 3
    assert len({round(d.oscillator_strength, 6) for d in balmer_alpha}) == 3
    assert len({d.label for d in balmer_alpha}) == 3

    for d in result.lines:
        assert d.thin_width_nm == pytest.approx(
            SIGMA_INTEGRAL / sc.c
            * d.lower_column_m2
            * d.oscillator_strength
            * (d.wavelength_nm * 1e-9) ** 2
            * 1e9,
            rel=1e-9,
        )

def test_a_p_lower_level_and_an_s_lower_level_differ_in_column():
    result = absorb(_hydrogen(), 1e20, emitter_mass=_mass())
    balmer = {
        d.label: d for d in result.lines if abs(d.wavelength_nm - 656.4696) < 1e-3
    }
    from_2s, from_2p = balmer["3p->2s"], balmer["3d->2p"]
    assert from_2p.lower_column_m2 == pytest.approx(3.0 * from_2s.lower_column_m2, rel=1e-6)

def test_absorption_is_labelled_approximation():
    result = absorb(_hydrogen(), 1e19, emitter_mass=_mass())
    for field in (result.transmission, result.optical_depth):
        assert field.provenance.fidelity is Fidelity.APPROXIMATION
    assert result.column_density.provenance.fidelity is Fidelity.COUNTERFACTUAL

def test_the_missing_pieces_are_named():
    text = " ".join(absorb(_hydrogen(), 1e19, emitter_mass=_mass())
                    .transmission.provenance.assumptions)
    assert "stimulated emission" in text
    assert "continuous opacity" in text
    assert "never reverses" in text

def test_absorption_refuses_a_list_with_no_populations():
    with pytest.raises(ValueError, match="lower-level population"):
        absorb(_hydrogen(thermal=None), 1e19, emitter_mass=_mass())

def test_absorption_refuses_a_negative_column():
    with pytest.raises(ValueError, match="column density"):
        absorb(_hydrogen(), -1.0, emitter_mass=_mass())

def _tau_unity_half_width(result) -> float:
    grid, tau = result.optical_depth.grid, result.optical_depth.values
    above = grid[tau >= 1.0]
    return float(above[-1] - above[0]) / 2.0

@pytest.mark.parametrize("column", [1e22, 1e23, 1e24, 1e25])
def test_the_window_rule_predicts_where_the_line_stops_absorbing(column):
    line_list = _only(_hydrogen(), 121.567)
    line = line_list.lines[0]
    lam = line.wavelength.value
    rates = level_decay_rates(line_list.lines)
    gamma = natural_gamma_nm(
        rates.get((line.n_upper, line.l_upper, line.j_upper), 0.0)
        + rates.get((line.n_lower, line.l_lower, line.j_lower), 0.0),
        lam,
    )
    sigma = doppler_sigma_nm(lam, HOT.temperature_k, _mass().value)

    result = absorb(line_list, column, emitter_mass=_mass())
    detail = result.lines[0]
    lorentz = math.sqrt(detail.thin_width_nm * gamma / math.pi)
    doppler = sigma * math.sqrt(2.0 * math.log(detail.tau_centre))

    assert _tau_unity_half_width(result) == pytest.approx(
        max(lorentz, doppler), rel=0.10
    )

def test_the_damping_wing_overtakes_the_doppler_core():
    narrow = _tau_unity_half_width(
        absorb(_only(_hydrogen(), 121.567), 1e22, emitter_mass=_mass())
    )
    wide = _tau_unity_half_width(
        absorb(_only(_hydrogen(), 121.567), 1e25, emitter_mass=_mass())
    )
    assert wide > 10.0 * narrow
