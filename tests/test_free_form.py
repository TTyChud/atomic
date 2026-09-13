import numpy as np

from atomic.numerics.force_law import force_law_levels, free_form_levels
from atomic.provenance import Fidelity


def test_coulomb_recovers_hydrogen():
    res = free_form_levels("-1/r", l=0, system="h", n_states=3)
    assert res.expression == "-1/r"
    assert res.preset_key == "custom"
    energies = [lvl.energy.value for lvl in res.counterfactual]
    np.testing.assert_allclose(energies[:3], [-0.5, -0.125, -1 / 18], rtol=2e-3)
    assert all(lvl.trusted for lvl in res.counterfactual[:3])
    assert res.counterfactual[0].energy.provenance.fidelity is Fidelity.NUMERICAL

def test_custom_matches_powerlaw_preset():
    a = free_form_levels("-1/r", l=0, system="h", n_states=3)
    b = force_law_levels("powerlaw", {"p": 1.0}, l=0, system="h", n_states=3)
    ea = [c.energy.value for c in a.counterfactual]
    eb = [c.energy.value for c in b.counterfactual]
    np.testing.assert_allclose(ea, eb, rtol=1e-3)

def test_oscillator_recovers_levels():
    res = free_form_levels("0.5*r**2", l=0, system="h", n_states=3)
    e = sorted(lvl.energy.value for lvl in res.counterfactual)
    np.testing.assert_allclose(e[:3], [1.5, 3.5, 5.5], rtol=5e-3)

def test_fall_to_center_flagged_untrusted():
    res = free_form_levels("-1/r**3", l=0, system="h", n_states=2)
    assert any(not lvl.trusted for lvl in res.counterfactual) or res.bound_count == 0

def test_purely_positive_has_no_trusted_bound_states():
    res = free_form_levels("exp(-r)", l=0, system="h", n_states=2)
    assert res.bound_count == 0
