from math import comb

import pytest

from atomic.atoms import (
    ATOM_KEYS,
    atom_for_key,
    aufbau_configuration,
    element_by_symbol,
    format_config,
    is_atom_key,
    is_ground,
    is_single_term,
    open_subshells,
    parse_config,
    subshell_capacity,
    subshell_term_count,
    subshell_terms,
    total_electrons,
    validate_config,
)


def test_subshell_capacity():
    assert subshell_capacity(0) == 2
    assert subshell_capacity(1) == 6
    assert subshell_capacity(2) == 10

@pytest.mark.parametrize("z,expected", [
    (1, "1s1"),
    (2, "1s2"),
    (6, "1s2 2s2 2p2"),
    (10, "1s2 2s2 2p6"),
    (11, "1s2 2s2 2p6 3s1"),
    (18, "1s2 2s2 2p6 3s2 3p6"),
])
def test_aufbau_matches_known_configs(z, expected):
    assert format_config(aufbau_configuration(z)) == expected

def test_config_roundtrip_and_count():
    cfg = parse_config("1s2 2s2 2p1")
    assert total_electrons(cfg) == 5
    assert format_config(cfg) == "1s2 2s2 2p1"

def test_is_ground():
    assert is_ground(aufbau_configuration(11)) is True
    assert is_ground(parse_config("1s2 2s2 2p6 3p1")) is False

def test_validate_rejects_overfill_and_bad_shell():
    with pytest.raises(ValueError, match="capacity"):
        validate_config(parse_config("1s3"))
    with pytest.raises(ValueError, match="n must be"):
        validate_config(((( 1, 1), 1),))

@pytest.mark.parametrize("l,q,terms", [
    (0, 1, 1),
    (0, 2, 1),
    (1, 1, 1),
    (1, 2, 3),
    (1, 3, 3),
    (1, 4, 3),
    (1, 5, 1),
    (1, 6, 1),
    (2, 1, 1),
    (2, 2, 5),
    (2, 3, 8),
    (2, 5, 16),
])
def test_subshell_term_counts_match_the_textbook_tables(l, q, terms):
    assert subshell_term_count(l, q) == terms

@pytest.mark.parametrize(
    "l,q",
    [(1, q) for q in range(7)] + [(2, q) for q in range(11)] + [(3, 3), (3, 7)],
)
def test_terms_account_for_every_determinant(l, q):
    degeneracy = sum(
        (2 * big_l + 1) * (twice_s + 1) for big_l, twice_s in subshell_terms(l, q)
    )
    assert degeneracy == comb(subshell_capacity(l), q)

def test_terms_name_the_expected_states_for_p2():
    assert subshell_terms(1, 2) == ((1, 2), (2, 0), (0, 0))

def test_term_enumeration_rejects_an_impossible_occupancy():
    with pytest.raises(ValueError, match="out of range"):
        subshell_terms(1, 7)

@pytest.mark.parametrize("z,single", [
    (2, True),
    (3, True),
    (5, True),
    (6, False),
    (7, False),
    (8, False),
    (9, True),
    (10, True),
    (16, False),
    (17, True),
    (18, True),
])
def test_single_term_configurations(z, single):
    assert is_single_term(aufbau_configuration(z)) is single

def test_two_open_subshells_never_span_a_single_term():
    cfg = parse_config("1s2 2s1 2p3")
    assert len(open_subshells(cfg)) == 2
    assert is_single_term(cfg) is False

def test_open_subshells_ignores_full_ones():
    assert open_subshells(aufbau_configuration(10)) == ()
    assert open_subshells(aufbau_configuration(6)) == (((2, 1), 2),)

def test_atom_keys_cover_he_to_ar():

    assert ATOM_KEYS[0] == "he" and ATOM_KEYS[-1] == "ar"
    assert len(ATOM_KEYS) == 17
    assert is_atom_key("na") and not is_atom_key("h")
    assert is_atom_key("s") and is_atom_key("cl")
    assert atom_for_key("na").z == 11 and element_by_symbol("Na").z == 11
