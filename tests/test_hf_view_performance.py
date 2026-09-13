
import time

import pytest

from atomic.atoms import aufbau_configuration, parse_config
from atomic.hf_atom import solve_hartree_fock
from atomic.isosurface import hf_isosurface
from atomic.plane import hf_plane_grid
from atomic.sampling import sample_hf_density


def test_four_views_share_one_solve():
    solve_hartree_fock.cache_clear()
    config = aufbau_configuration(10)

    sample_hf_density(10, 10, 2, 1, 0, 2_000, config=config)
    hf_plane_grid(10, 10, 2, 1, 0, resolution=32, config=config)
    hf_isosurface(10, 10, 2, 1, 0, resolution=48, config=config)

    info = solve_hartree_fock.cache_info()
    assert info.misses == 1, f"expected one solve, got {info.misses}"
    assert info.hits > 0

def test_default_config_shares_the_explicit_one():
    solve_hartree_fock.cache_clear()
    sample_hf_density(10, 10, 2, 1, 0, 1_000)
    sample_hf_density(10, 10, 2, 1, 0, 1_000, config=aufbau_configuration(10))
    assert solve_hartree_fock.cache_info().misses == 1

def test_the_counterfactual_key_space_fits_the_cache():
    solve_hartree_fock.cache_clear()
    ground = aufbau_configuration(10)
    excited = parse_config("1s2 2s2 2p5 3s1")
    collapsed = aufbau_configuration(10, pauli=False)

    keys = [
        (ground, True, True),
        (ground, False, True),
        (collapsed, False, False),
        (excited, True, True),
        (excited, False, True),
    ]
    for config, exchange, pauli in keys:
        solve_hartree_fock(10, 10, config, exchange, pauli)
    first_misses = solve_hartree_fock.cache_info().misses

    for config, exchange, pauli in keys:
        solve_hartree_fock(10, 10, config, exchange, pauli)
    assert solve_hartree_fock.cache_info().misses == first_misses

@pytest.mark.parametrize("resolution", [96])
def test_isosurface_budget(resolution):
    solve_hartree_fock(10, 10, aufbau_configuration(10))
    t0 = time.monotonic()
    surf = hf_isosurface(10, 10, 2, 1, 0, resolution=resolution)
    elapsed = time.monotonic() - t0
    assert surf.vertices.shape[0] > 0
    assert elapsed < 60.0, f"96^3 Hartree-Fock isosurface took {elapsed:.1f}s"
    print(f"\nHF isosurface {resolution}^3, warm solve: {elapsed:.2f}s")
