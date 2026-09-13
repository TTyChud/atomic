from dataclasses import dataclass, field
from enum import Enum
from numbers import Real

import numpy as np


class Fidelity(Enum):
    EXACT = "exact"
    NUMERICAL = "numerical"
    APPROXIMATION = "approximation"
    COUNTERFACTUAL = "counterfactual"
    VISUAL_LIBERTY = "visual_liberty"

def _require_nonempty_str(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value

def _require_finite_nonnegative_or_none(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number or None, got {value!r}")
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if v < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return v

@dataclass(frozen=True)
class Provenance:
    fidelity: Fidelity
    method: str
    assumptions: tuple[str, ...] = field(default=())
    error_estimate: float | None = None
    refinement: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.fidelity, Fidelity):
            raise ValueError(
                f"fidelity must be a Fidelity member, got {self.fidelity!r}"
            )
        _require_nonempty_str("method", self.method)
        if not isinstance(self.assumptions, (tuple, list)):
            raise ValueError(
                f"assumptions must be a tuple of strings, got {type(self.assumptions).__name__}"
            )
        for a in self.assumptions:
            _require_nonempty_str("assumptions entry", a)
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(
            self, "error_estimate",
            _require_finite_nonnegative_or_none("error_estimate", self.error_estimate),
        )
        if self.refinement is not None:
            _require_nonempty_str("refinement", self.refinement)

@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str
    label: str
    provenance: Provenance

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, Real):
            raise ValueError(f"value must be a real number, got {self.value!r}")
        v = float(self.value)
        if np.isnan(v):
            raise ValueError("value must not be NaN (use inf for unbounded, never silence)")

        _require_nonempty_str("unit", self.unit)
        _require_nonempty_str("label", self.label)
        if not isinstance(self.provenance, Provenance):
            raise ValueError(
                f"provenance must be a Provenance, got {type(self.provenance).__name__}"
            )
        object.__setattr__(self, "value", v)

@dataclass(frozen=True)
class Field:

    values: np.ndarray
    grid: np.ndarray
    unit: str
    grid_unit: str
    label: str
    provenance: Provenance

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, Provenance):
            raise ValueError(
                f"provenance must be a Provenance, got {type(self.provenance).__name__}"
            )
        _require_nonempty_str("unit", self.unit)
        _require_nonempty_str("grid_unit", self.grid_unit)
        _require_nonempty_str("label", self.label)
        values = np.asarray(self.values, dtype=float)
        grid = np.asarray(self.grid, dtype=float)
        if grid.ndim != 1:
            raise ValueError(f"grid must be 1-D, got shape {grid.shape}")
        if grid.size == 0:
            raise ValueError("grid must not be empty")
        if not np.all(np.isfinite(grid)):
            raise ValueError("grid must be finite (no NaN or inf)")
        if values.shape == ():
            raise ValueError("values must be an array, got a scalar")
        if values.shape[-1] != grid.shape[0]:
            raise ValueError(
                f"values last axis ({values.shape[-1]}) must match "
                f"grid length ({grid.shape[0]})"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("values must be finite (no NaN or inf)")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "grid", grid)
