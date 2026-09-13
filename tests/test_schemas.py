import json

import numpy as np
import pytest

from atomic.provenance import Fidelity, Field, Provenance, Quantity
from atomic.server.schemas import FieldModel, ProvenanceModel, QuantityModel


def _prov():
    return Provenance(
        fidelity=Fidelity.NUMERICAL,
        method="test method",
        assumptions=("a1", "a2"),
        error_estimate=1e-6,
        refinement="refine",
    )

def test_provenance_round_trip():
    model = ProvenanceModel.from_provenance(_prov())
    data = json.loads(model.model_dump_json())
    assert data["fidelity"] == "numerical"
    assert data["assumptions"] == ["a1", "a2"]
    assert data["error_estimate"] == pytest.approx(1e-6)
    assert data["refinement"] == "refine"

def test_quantity_serializes_with_provenance():
    q = Quantity(value=-0.5, unit="hartree", label="E_1", provenance=_prov())
    data = json.loads(QuantityModel.from_quantity(q).model_dump_json())
    assert data["value"] == -0.5
    assert data["unit"] == "hartree"
    assert data["provenance"]["method"] == "test method"

def test_field_serializes_values_and_grid():
    r = np.linspace(0.0, 1.0, 5)
    f = Field(values=2.0 * r, grid=r, unit="u", grid_unit="bohr", label="f", provenance=_prov())
    data = json.loads(FieldModel.from_field(f).model_dump_json())
    assert data["grid"] == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert data["values"] == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])

def test_field_model_rejects_stacked_values():
    r = np.linspace(0.0, 1.0, 4)
    f = Field(values=np.zeros((2, 4)), grid=r, unit="dimensionless", grid_unit="bohr",
              label="u", provenance=_prov())
    with pytest.raises(ValueError, match="1-D"):
        FieldModel.from_field(f)

def test_every_fidelity_value_is_representable():
    for fid in Fidelity:
        p = Provenance(fidelity=fid, method="m")
        assert ProvenanceModel.from_provenance(p).fidelity == fid.value
