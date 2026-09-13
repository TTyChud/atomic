
import pytest

from atomic.hf_reference import HF_REFERENCE, load_hf_reference

BENCHMARK_ATOMS = [("He", 2), ("Be", 4), ("Ne", 10), ("Mg", 12), ("Ar", 18)]
SYMBOLS = [symbol for symbol, _ in BENCHMARK_ATOMS]

def test_metadata_is_present_and_dated():
    assert HF_REFERENCE["citation"]
    assert HF_REFERENCE["retrieved"]
    assert HF_REFERENCE["units"] == "hartree"

def test_transcription_source_is_disclosed():
    assert HF_REFERENCE["transcribed_from"]

@pytest.mark.parametrize("symbol,z", BENCHMARK_ATOMS)
def test_every_benchmark_atom_has_an_entry(symbol, z):
    entry = load_hf_reference(symbol)
    assert entry["z"] == z

@pytest.mark.parametrize("symbol", SYMBOLS)
def test_neutral_atoms_have_z_electrons(symbol):
    entry = load_hf_reference(symbol)
    assert entry["n_electrons"] == entry["z"]

@pytest.mark.parametrize("symbol", SYMBOLS)
def test_energies_are_bound(symbol):
    entry = load_hf_reference(symbol)
    if entry["total_energy_hartree"] is None:
        pytest.skip("reference energies not yet transcribed from the source")
    assert entry["total_energy_hartree"] < 0.0

def test_energies_decrease_with_z():
    energies = [load_hf_reference(s)["total_energy_hartree"] for s in SYMBOLS]
    if any(e is None for e in energies):
        pytest.skip("reference energies not yet transcribed from the source")
    assert all(a > b for a, b in zip(energies[:-1], energies[1:], strict=True))

@pytest.mark.parametrize("symbol", SYMBOLS)
def test_energy_is_within_the_hydrogenic_bracket(symbol):
    entry = load_hf_reference(symbol)
    energy = entry["total_energy_hartree"]
    if energy is None:
        pytest.skip("reference energies not yet transcribed from the source")

    z = entry["z"]
    shells = {"He": [2], "Be": [2, 2], "Ne": [2, 8], "Mg": [2, 8, 2],
              "Ar": [2, 8, 8]}[symbol]
    hydrogenic = -sum(
        count * z**2 / (2 * n**2) for n, count in enumerate(shells, start=1)
    )
    assert hydrogenic < energy < -(z**2) / 2

def test_unknown_symbol_raises():
    with pytest.raises(KeyError):
        load_hf_reference("Xx")

def test_unknown_symbol_error_names_what_is_available():
    with pytest.raises(KeyError, match="He"):
        load_hf_reference("Xx")
