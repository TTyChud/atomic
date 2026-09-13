from scipy.constants import physical_constants

from atomic.constants import HARTREE_EV, FundamentalConstants


def test_hartree_ev_matches_codata():

    assert abs(HARTREE_EV - 27.211386) < 1e-6

def test_derived_alpha_matches_published_value():
    c = FundamentalConstants.codata()
    published = physical_constants["fine-structure constant"][0]
    assert abs(c.alpha - published) / published < 1e-9

def test_derived_bohr_radius_matches_published_value():
    c = FundamentalConstants.codata()
    published = physical_constants["Bohr radius"][0]
    assert abs(c.bohr_radius - published) / published < 1e-9

def test_derived_hartree_matches_published_value():
    c = FundamentalConstants.codata()
    published = physical_constants["Hartree energy"][0]
    assert abs(c.hartree_energy - published) / published < 1e-9

def test_counterfactual_universe_rescales():

    real = FundamentalConstants.codata()
    weird = FundamentalConstants(
        hbar=real.hbar, e=2 * real.e, m_e=real.m_e, eps0=real.eps0, c=real.c
    )
    assert abs(weird.alpha / real.alpha - 4.0) < 1e-12
    assert abs(weird.bohr_radius / real.bohr_radius - 0.25) < 1e-12

def test_alpha_matches_codata():
    from atomic.constants import ALPHA

    assert abs(ALPHA - 0.0072973525643) < 1e-11
