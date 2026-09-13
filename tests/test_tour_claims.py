
import math

import pytest

from atomic.tour_claims import CLAIM_KINDS, iter_claims, load_tours, resolve_claim


class TestResolverKinds:

    def test_energy_is_the_bohr_formula_in_ev(self):
        got = resolve_claim({"of": "energy_eV", "system": "h", "n": 1})
        assert got == pytest.approx(-13.5983, abs=1e-3)
        assert resolve_claim({"of": "energy_eV", "system": "h", "n": 2}) == pytest.approx(
            -3.3996, abs=1e-3
        )

    def test_energy_scales_with_reduced_mass(self):
        h = resolve_claim({"of": "energy_eV", "system": "h", "n": 1})
        d = resolve_claim({"of": "energy_eV", "system": "d", "n": 1})
        assert d < h
        assert abs(d - h) == pytest.approx(0.0037, abs=5e-4)

    def test_mean_radius_is_the_closed_form_in_pm(self):
        got = resolve_claim({"of": "mean_r_pm", "system": "h", "n": 1, "l": 0})
        assert got == pytest.approx(79.4198, abs=1e-2)

    def test_mean_radius_depends_on_l_not_only_n(self):
        s = resolve_claim({"of": "mean_r_pm", "system": "h", "n": 2, "l": 0})
        p = resolve_claim({"of": "mean_r_pm", "system": "h", "n": 2, "l": 1})
        assert s > p

    def test_wavelength_is_lyman_alpha(self):
        got = resolve_claim({"of": "wavelength_nm", "system": "h", "n_upper": 2, "n_lower": 1})
        assert got == pytest.approx(121.567, abs=0.01)

    def test_wavelength_is_h_alpha(self):
        got = resolve_claim({"of": "wavelength_nm", "system": "h", "n_upper": 3, "n_lower": 2})
        assert got == pytest.approx(656.47, abs=0.05)

    def test_ionization_energy_of_helium(self):
        got = resolve_claim({"of": "ionization_eV", "system": "he", "model": "hf"})
        assert got == pytest.approx(24.98, abs=0.3)

    def test_every_declared_kind_resolves(self):
        assert set(CLAIM_KINDS) == {
            "energy_eV",
            "mean_r_pm",
            "wavelength_nm",
            "ionization_eV",
        }

    def test_unknown_kind_raises_rather_than_returning_zero(self):
        with pytest.raises(ValueError, match="unknown claim kind"):
            resolve_claim({"of": "spin_of_the_universe", "system": "h"})

    def test_missing_input_raises_rather_than_defaulting(self):
        with pytest.raises(KeyError):
            resolve_claim({"of": "wavelength_nm", "system": "h", "n_upper": 3})

class TestTourContent:
    def test_tours_load(self):
        tours = load_tours()
        assert tours, "no tour JSON found; check the path in load_tours"

    def test_every_claim_holds(self):
        checked = 0
        for tour_id, step_id, claim in iter_claims():
            got = resolve_claim(claim)
            assert math.isfinite(got), f"{tour_id}/{step_id}: {claim['of']} is not finite"
            assert got == pytest.approx(claim["is"], abs=claim["tol"]), (
                f"{tour_id}/{step_id} claims {claim['of']} = {claim['is']} "
                f"+/- {claim['tol']}, engine says {got:.6g}. "
                f"Either the prose is now wrong or the engine changed."
            )
            checked += 1
        assert checked > 0, "no claims checked; the tours declare none"
