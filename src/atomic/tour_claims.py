
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from atomic.analytic.hydrogen import energy, mean_radius
from atomic.atoms import ATOM_KEYS, atom_for_key, aufbau_configuration
from atomic.constants import BOHR_RADIUS_PM, HARTREE_EV
from atomic.hf_atom import hf_valence_ionization_energy, solve_hartree_fock
from atomic.spectra import transition_lines
from atomic.systems import get_system

CLAIM_KINDS: tuple[str, ...] = (
    "energy_eV",
    "mean_r_pm",
    "wavelength_nm",
    "ionization_eV",
)

_TOUR_DIR = Path(__file__).resolve().parents[2] / "web" / "src" / "tours"

def load_tours() -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(_TOUR_DIR.glob("*.json"))]

def iter_claims() -> Iterator[tuple[str, str, dict[str, Any]]]:
    inherit = (
        "system",
        "n",
        "l",
        "m",
        "model",
        "fineStructure",
        "dirac",
        "exchange",
        "pauli",
    )
    for tour in load_tours():
        for step in tour["steps"]:
            state = step.get("state", {})
            for claim in step.get("claims", []):
                merged = {k: state[k] for k in inherit if k in state}
                merged.update(claim)
                yield tour["id"], step["id"], merged

def _system_of(claim: dict[str, Any]):
    return get_system(claim.get("system", "h"))

def _energy_ev(claim: dict[str, Any]) -> float:
    system = _system_of(claim)
    q = energy(claim["n"], Z=system.Z, mu_ratio=system.mu_ratio.value)
    return q.value * HARTREE_EV

def _mean_r_pm(claim: dict[str, Any]) -> float:
    system = _system_of(claim)
    q = mean_radius(claim["n"], claim["l"], Z=system.Z, mu_ratio=system.mu_ratio.value)
    return q.value * BOHR_RADIUS_PM

def _wavelength_nm(claim: dict[str, Any]) -> float:
    system = _system_of(claim)
    n_up = claim["n_upper"]
    n_lo = claim["n_lower"]
    lines = transition_lines(system, n_max=max(n_up, n_lo), fine_structure=False)
    for line in lines.lines:
        if line.n_upper == n_up and line.n_lower == n_lo:
            return line.wavelength.value
    raise ValueError(f"no {n_up} -> {n_lo} line in {system.key}")

def _ionization_ev(claim: dict[str, Any]) -> float:
    key = claim.get("system", "h")
    if key not in ATOM_KEYS:
        raise ValueError(f"ionization_eV needs a many-electron atom, got {key!r}")
    z = atom_for_key(key).z
    pauli = claim.get("pauli", True)
    exchange = claim.get("exchange", True)
    config = aufbau_configuration(z, pauli=pauli)
    result = solve_hartree_fock(z, z, config, exchange, pauli)
    return hf_valence_ionization_energy(result).value * HARTREE_EV

_RESOLVERS = {
    "energy_eV": _energy_ev,
    "mean_r_pm": _mean_r_pm,
    "wavelength_nm": _wavelength_nm,
    "ionization_eV": _ionization_ev,
}

def resolve_claim(claim: dict[str, Any]) -> float:
    kind = claim.get("of")
    if kind not in _RESOLVERS:
        raise ValueError(f"unknown claim kind {kind!r}; known: {', '.join(CLAIM_KINDS)}")
    return _RESOLVERS[kind](claim)
