import itertools
from collections import Counter
from dataclasses import dataclass

SUBSHELL_LABELS = "spdfgh"

Subshell = tuple[int, int]
Configuration = tuple[tuple[Subshell, int], ...]

def subshell_capacity(l: int) -> int:
    return 2 * (2 * l + 1)

def _madelung_order() -> list[Subshell]:
    shells = [(n, l) for n in range(1, 8) for l in range(n)]
    shells.sort(key=lambda nl: (nl[0] + nl[1], nl[0]))
    return shells

_MADELUNG = _madelung_order()

def aufbau_configuration(n_electrons: int, pauli: bool = True) -> Configuration:
    if n_electrons < 1:
        raise ValueError(f"n_electrons must be >= 1, got {n_electrons}")
    if not pauli:
        return (((1, 0), n_electrons),)
    remaining = n_electrons
    out: list[tuple[Subshell, int]] = []
    for n, l in _MADELUNG:
        if remaining <= 0:
            break
        fill = min(subshell_capacity(l), remaining)
        out.append(((n, l), fill))
        remaining -= fill
    if remaining > 0:
        raise ValueError(f"{n_electrons} electrons exceeds supported shells")
    return tuple(out)

def format_config(config: Configuration) -> str:
    return " ".join(f"{n}{SUBSHELL_LABELS[l]}{occ}" for (n, l), occ in config)

def parse_config(text: str) -> Configuration:
    out: list[tuple[Subshell, int]] = []
    for tok in text.split():
        n = int(tok[0])
        l = SUBSHELL_LABELS.index(tok[1])
        occ = int(tok[2:])
        out.append(((n, l), occ))
    return tuple(out)

def total_electrons(config: Configuration) -> int:
    return sum(occ for _, occ in config)

def is_ground(config: Configuration, pauli: bool = True) -> bool:
    return config == aufbau_configuration(total_electrons(config), pauli)

def validate_config(config: Configuration, pauli: bool = True) -> None:
    for (n, l), occ in config:
        if n <= l:
            raise ValueError(f"n must be > l for a real subshell, got n={n}, l={l}")
        if occ < 0:
            raise ValueError(f"occupancy must be >= 0, got {occ}")
        if pauli and occ > subshell_capacity(l):
            raise ValueError(
                f"occupancy {occ} exceeds capacity {subshell_capacity(l)} for l={l}"
            )

def open_subshells(config: Configuration) -> Configuration:
    return tuple(
        (nl, occ) for nl, occ in config if 0 < occ < subshell_capacity(nl[1])
    )

def _microstate_census(l: int, q: int) -> Counter[tuple[int, int]]:
    spin_orbitals = [(ml, ms) for ml in range(-l, l + 1) for ms in (1, -1)]
    census: Counter[tuple[int, int]] = Counter()
    for pick in itertools.combinations(spin_orbitals, q):
        census[(sum(ml for ml, _ in pick), sum(ms for _, ms in pick))] += 1
    return census

def subshell_terms(l: int, q: int) -> tuple[tuple[int, int], ...]:
    if not 0 <= q <= subshell_capacity(l):
        raise ValueError(f"occupancy {q} out of range for l={l}")
    census = _microstate_census(l, q)
    found: list[tuple[int, int]] = []
    while census:
        twice_s = max(key[1] for key in census)
        big_l = max(key[0] for key in census if key[1] == twice_s)
        found.append((big_l, twice_s))
        for ml in range(-big_l, big_l + 1):
            for twice_ms in range(-twice_s, twice_s + 1, 2):
                if census[(ml, twice_ms)] < 1:
                    raise AssertionError(
                        f"term (L={big_l}, 2S={twice_s}) of l={l} q={q} claims a "
                        f"microstate (M_L={ml}, 2*M_S={twice_ms}) the census "
                        f"does not hold; the peel is wrong"
                    )
                census[(ml, twice_ms)] -= 1
                if census[(ml, twice_ms)] == 0:
                    del census[(ml, twice_ms)]
    return tuple(found)

def subshell_term_count(l: int, q: int) -> int:
    return len(subshell_terms(l, q))

def is_single_term(config: Configuration) -> bool:
    open_shells = open_subshells(config)
    if len(open_shells) > 1:
        return False
    return all(subshell_term_count(l, q) == 1 for (_, l), q in open_shells)

@dataclass(frozen=True)
class Element:
    z: int
    symbol: str
    name: str

    mass_u: float

ELEMENTS: tuple[Element, ...] = (
    Element(1, "H", "Hydrogen", 1.008), Element(2, "He", "Helium", 4.002602),
    Element(3, "Li", "Lithium", 6.94), Element(4, "Be", "Beryllium", 9.012183),
    Element(5, "B", "Boron", 10.81), Element(6, "C", "Carbon", 12.011),
    Element(7, "N", "Nitrogen", 14.007), Element(8, "O", "Oxygen", 15.999),
    Element(9, "F", "Fluorine", 18.998403), Element(10, "Ne", "Neon", 20.1797),
    Element(11, "Na", "Sodium", 22.989769), Element(12, "Mg", "Magnesium", 24.305),
    Element(13, "Al", "Aluminium", 26.981538), Element(14, "Si", "Silicon", 28.085),
    Element(15, "P", "Phosphorus", 30.973762), Element(16, "S", "Sulfur", 32.06),
    Element(17, "Cl", "Chlorine", 35.45), Element(18, "Ar", "Argon", 39.95),
)

_BY_SYMBOL = {e.symbol: e for e in ELEMENTS}
_BY_Z = {e.z: e for e in ELEMENTS}

NO_GSZ_PARAMETERS: frozenset[int] = frozenset({16, 17})

ATOM_KEYS: tuple[str, ...] = tuple(
    e.symbol.lower() for e in ELEMENTS if e.z >= 2
)

GSZ_ATOM_KEYS: tuple[str, ...] = tuple(
    k for k in ATOM_KEYS if _BY_SYMBOL[k.capitalize()].z not in NO_GSZ_PARAMETERS
)

def has_gsz_parameters(z: int) -> bool:
    return z not in NO_GSZ_PARAMETERS

def element_by_symbol(sym: str) -> Element:
    return _BY_SYMBOL[sym]

def element_by_z(z: int) -> Element:
    return _BY_Z[z]

def is_atom_key(key: str) -> bool:
    return key in ATOM_KEYS

def atom_for_key(key: str) -> Element:
    if not is_atom_key(key):
        raise KeyError(f"unknown atom key {key!r}; known: {ATOM_KEYS}")
    return _BY_SYMBOL[key.capitalize()]
