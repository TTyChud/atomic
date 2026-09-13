
import pytest

from atomic.analytic.hydrogen import energy
from atomic.analytic.stark import stark_sublevels
from atomic.constants import E0_V_PER_M
from atomic.provenance import Fidelity

_AU_PER_MVM = 1e6 / E0_V_PER_M

def _field_au(mvm: float) -> float:
    return mvm * _AU_PER_MVM

def test_zero_field_recovers_bohr():
    for n in (1, 2, 3):
        want = energy(n).value
        for s in stark_sublevels(n, field_mv_per_m=0.0):
            assert s.energy.value == pytest.approx(want, rel=1e-12)

def test_sublevel_count_is_n_squared():
    assert len(stark_sublevels(1, field_mv_per_m=5.0)) == 1
    assert len(stark_sublevels(2, field_mv_per_m=5.0)) == 4
    assert len(stark_sublevels(3, field_mv_per_m=5.0)) == 9
    assert len(stark_sublevels(4, field_mv_per_m=5.0)) == 16

def test_parabolic_constraint():
    for n in (1, 2, 3, 4):
        for s in stark_sublevels(n, field_mv_per_m=1.0):
            assert s.n1 + s.n2 + abs(s.m) + 1 == n
            assert s.k == s.n1 - s.n2

def test_low_field_slope_is_linear_stark():
    mvm = 1e-3
    f = _field_au(mvm)
    n = 3
    e0 = {(s.n1, s.n2, s.m): s.energy.value for s in stark_sublevels(n, field_mv_per_m=0.0)}
    for s in stark_sublevels(n, field_mv_per_m=mvm):
        slope = (s.energy.value - e0[(s.n1, s.n2, s.m)]) / f
        assert slope == pytest.approx(1.5 * n * s.k, rel=1e-4, abs=5e-6)

def test_linear_fan_is_traceless():
    n = 4
    assert sum(s.k for s in stark_sublevels(n, field_mv_per_m=1.0)) == 0

def test_ground_state_polarizability():
    n = 1
    mvm = 10.0
    f = _field_au(mvm)
    e0 = energy(1).value
    s = stark_sublevels(n, field_mv_per_m=mvm)[0]
    assert s.k == 0
    coeff = (s.energy.value - e0) / (f * f)
    assert coeff == pytest.approx(-9.0 / 4.0, rel=1e-6)

def test_pm_m_degeneracy():
    subs = {(s.n1, s.n2, s.m): s.energy.value for s in stark_sublevels(3, field_mv_per_m=7.0)}
    for (n1, n2, m), val in subs.items():
        assert subs[(n1, n2, -m)] == pytest.approx(val, rel=1e-12)

def test_z_scaling_halves_linear_shift():
    mvm = 1.0
    n = 2
    def linshift(Z):
        s = next(s for s in stark_sublevels(n, Z=Z, field_mv_per_m=mvm) if s.k == 1)
        return s.energy.value - energy(n, Z=Z).value
    small = 1e-4
    fs = _field_au(small)
    def lin(Z):
        s = next(s for s in stark_sublevels(n, Z=Z, field_mv_per_m=small) if s.k == 1)
        return (s.energy.value - energy(n, Z=Z).value) / fs
    assert lin(2) == pytest.approx(lin(1) / 2.0, rel=1e-3)

def test_provenance_tier_and_error():
    s = stark_sublevels(3, field_mv_per_m=10.0)[0]
    assert s.energy.provenance.fidelity is Fidelity.APPROXIMATION
    assert s.energy.provenance.error_estimate is not None
    def err(mvm):
        return next(
            x for x in stark_sublevels(3, field_mv_per_m=mvm) if x.k == 2
        ).energy.provenance.error_estimate
    assert err(20.0) > err(10.0) > 0.0

def _quad_coeff(Z=1, mu=1.0):
    mvm = 100.0
    f = _field_au(mvm)
    s = stark_sublevels(1, Z=Z, mu_ratio=mu, field_mv_per_m=mvm)[0]
    return (s.energy.value - energy(1, Z=Z, mu_ratio=mu).value) / (f * f)

def test_quadratic_scales_as_z4():
    assert _quad_coeff(Z=2) == pytest.approx(_quad_coeff(Z=1) / 16.0, rel=1e-6)

def test_quadratic_scales_as_mu3():
    assert _quad_coeff(mu=0.5) == pytest.approx(_quad_coeff(mu=1.0) / 0.5**3, rel=1e-6)
    assert _quad_coeff(mu=1.0) == pytest.approx(-9.0 / 4.0, rel=1e-6)

def test_negative_field_rejected():
    with pytest.raises(ValueError):
        stark_sublevels(2, field_mv_per_m=-1.0)
