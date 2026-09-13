import dataclasses

import numpy as np
import pytest

from atomic.provenance import Fidelity, Field, Provenance, Quantity


def test_fidelity_has_exactly_the_five_spec_tiers():
    assert {f.value for f in Fidelity} == {
        "exact",
        "numerical",
        "approximation",
        "counterfactual",
        "visual_liberty",
    }

def test_quantity_carries_full_provenance():
    p = Provenance(
        fidelity=Fidelity.EXACT,
        method="closed-form Bohr formula",
        assumptions=("non-relativistic", "point nucleus"),
    )
    q = Quantity(value=-0.5, unit="hartree", label="E_1", provenance=p)
    assert q.value == -0.5
    assert q.unit == "hartree"
    assert q.provenance.fidelity is Fidelity.EXACT
    assert "point nucleus" in q.provenance.assumptions
    assert q.provenance.error_estimate is None

def test_provenance_and_quantity_are_immutable():
    p = Provenance(fidelity=Fidelity.NUMERICAL, method="fd")
    q = Quantity(value=1.0, unit="bohr", label="r", provenance=p)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.method = "changed"
    with pytest.raises(dataclasses.FrozenInstanceError):
        q.value = 2.0

def _prov():
    return Provenance(fidelity=Fidelity.EXACT, method="closed-form test fixture")

def test_field_carries_values_grid_and_provenance():
    r = np.linspace(0.1, 10.0, 50)
    f = Field(
        values=np.exp(-r),
        grid=r,
        unit="bohr^-3/2",
        grid_unit="bohr",
        label="R_1,0",
        provenance=_prov(),
    )
    assert f.values.shape == (50,)
    assert f.grid_unit == "bohr"
    assert f.provenance.fidelity is Fidelity.EXACT

def test_field_accepts_stacked_values_with_matching_last_axis():
    r = np.linspace(0.0, 1.0, 20)
    f = Field(
        values=np.zeros((3, 20)), grid=r, unit="dimensionless", grid_unit="bohr",
        label="u_k", provenance=_prov(),
    )
    assert f.values.shape == (3, 20)

def test_field_rejects_mismatched_grid_length():
    with pytest.raises(ValueError, match="grid"):
        Field(
            values=np.zeros(4), grid=np.zeros(5), unit="dimensionless", grid_unit="bohr",
            label="bad", provenance=_prov(),
        )

def test_field_accepts_unsorted_sample_points():
    sample = np.array([2.0, 0.5, 2.0, 1.0])
    f = Field(
        values=np.exp(-sample), grid=sample, unit="bohr^-3/2", grid_unit="bohr",
        label="R_1,0", provenance=_prov(),
    )
    assert f.values.shape == (4,)

def test_field_rejects_non_1d_grid():
    with pytest.raises(ValueError, match="1-D"):
        Field(
            values=np.zeros(4), grid=np.zeros((2, 2)), unit="dimensionless", grid_unit="bohr",
            label="bad", provenance=_prov(),
        )
