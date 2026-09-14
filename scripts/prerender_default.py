#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from atomic.atoms import atom_for_key
from atomic.sampling import sample_screened_density
from atomic.screened_atom import evaluate_screened_state

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / "web" / "public" / "prerender" / "default"

DEFAULT_SYSTEM = "c"
DEFAULT_N = 6
DEFAULT_L = 2
DEFAULT_M = 0
DEFAULT_BASIS = "complex"


def _prov(p) -> dict:
    return {
        "fidelity": p.fidelity.value,
        "method": p.method,
        "assumptions": list(p.assumptions),
        "error_estimate": p.error_estimate,
        "refinement": p.refinement,
    }


def prerender(dest: Path, count: int, seed: int) -> None:
    element = atom_for_key(DEFAULT_SYSTEM)
    cloud = sample_screened_density(
        element.z, element.z, DEFAULT_N, DEFAULT_L, DEFAULT_M, count,
        seed=seed, basis=DEFAULT_BASIS,
    )
    psi = evaluate_screened_state(
        element.z, element.z, DEFAULT_N, DEFAULT_L, DEFAULT_M,
        cloud.positions.astype(np.float64), basis=DEFAULT_BASIS,
    )
    density = (np.abs(psi.values) ** 2).astype(np.float32)
    phase = np.angle(psi.values).astype(np.float32)
    positions = cloud.positions.astype(np.float32)

    psi_prov = _prov(psi.provenance)
    meta = {
        "kind": "sample",
        "count": int(positions.shape[0]),
        "dtype": "float32",
        "layout": "xyz-interleaved",
        "unit": "bohr",
        "n": cloud.n,
        "l": cloud.l,
        "m": cloud.m,
        "basis": cloud.basis,
        "system": DEFAULT_SYSTEM,
        "model": "screened",
        "seed": seed,
        "provenance": _prov(cloud.provenance),
        "channels": [
            {"name": "positions", "dtype": "float32", "unit": "bohr",
             "provenance": _prov(cloud.provenance)},
            {"name": "density", "dtype": "float32", "unit": "bohr^-3",
             "provenance": psi_prov},
            {"name": "phase", "dtype": "float32", "unit": "rad",
             "provenance": psi_prov},
        ],
    }

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    positions.tofile(dest / "positions.bin")
    density.tofile(dest / "density.bin")
    phase.tofile(dest / "phase.bin")
    total = (
        positions.nbytes + density.nbytes + phase.nbytes
    )
    print(f"prerendered {meta['count']} points ({total / 1e6:.1f} MB) -> {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()
    prerender(args.dest, args.count, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
