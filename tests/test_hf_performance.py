
import time

import pytest

from atomic.atoms import aufbau_configuration
from atomic.hf_atom import hf_mesh, solve_hartree_fock

_WALL_CLOCK_CEILING = 60.0

_COARSE_SCF_CEILING = 25

_FINE_SCF_CEILING = 15

_ARGON_POINT_CEILING = 6000

def _solve_cold(z: int):
    assert hasattr(solve_hartree_fock, "__wrapped__"), (
        "solve_hartree_fock is no longer memoized; test_repeated_solves_hit_"
        "the_cache should have caught this first"
    )
    return solve_hartree_fock.__wrapped__(z, z, aufbau_configuration(z))

@pytest.mark.parametrize("symbol,z", [("Ar", 18), ("Cl", 17)])
def test_a_cold_solve_stays_within_the_budget(symbol, z):
    start = time.perf_counter()
    _solve_cold(z)
    elapsed = time.perf_counter() - start
    assert elapsed < _WALL_CLOCK_CEILING, (
        f"{symbol} took {elapsed:.1f}s against a ceiling of "
        f"{_WALL_CLOCK_CEILING:.0f}s; profile before raising this"
    )

def test_repeated_solves_hit_the_cache():
    solve_hartree_fock(10, 10, aufbau_configuration(10))
    start = time.perf_counter()
    solve_hartree_fock(10, 10, aufbau_configuration(10))
    assert time.perf_counter() - start < 0.1

def test_the_configuration_argument_does_not_defeat_the_cache():
    a = aufbau_configuration(10)
    b = aufbau_configuration(10)
    assert a is not b
    assert hash(a) == hash(b)
    before = solve_hartree_fock.cache_info()
    solve_hartree_fock(10, 10, a)
    solve_hartree_fock(10, 10, b)
    after = solve_hartree_fock.cache_info()
    assert after.hits - before.hits >= 1

@pytest.mark.parametrize("symbol,z", [("Li", 3), ("Ne", 10), ("S", 16), ("Ar", 18)])
def test_scf_iteration_counts_stay_bounded(symbol, z):
    result = _solve_cold(z)
    assert result.coarse_iterations < _COARSE_SCF_CEILING, (
        f"{symbol} took {result.coarse_iterations} coarse SCF iterations; "
        f"check the mixing parameter in numerics.hartree_fock.scf"
    )
    assert result.iterations < _FINE_SCF_CEILING, (
        f"{symbol} took {result.iterations} fine SCF iterations"
    )

def test_the_coarse_solve_is_the_expensive_one():
    result = _solve_cold(18)
    assert result.coarse_iterations > result.iterations

@pytest.mark.parametrize("refinement,ceiling", [(1, _ARGON_POINT_CEILING // 2),
                                                (2, _ARGON_POINT_CEILING)])
def test_argon_runs_on_a_small_mesh(refinement, ceiling):
    config = aufbau_configuration(18)
    n_top = max(n for (n, _), _ in config)
    mesh = hf_mesh(18, 18, n_top, refinement=refinement)
    assert len(mesh.r) < ceiling, (
        f"argon's refinement-{refinement} mesh has {len(mesh.r)} points; "
        f"a uniform grid of comparable accuracy needed 72000 and about an hour"
    )
