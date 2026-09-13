from atomic.provenance import Fidelity, Provenance, Quantity

_PROV = Provenance(
    fidelity=Fidelity.EXACT,
    method="3-D isotropic harmonic oscillator closed form E = omega(2k + l + 3/2)",
    assumptions=("Hartree atomic units, with hbar = 1",),
)

def oscillator_energy(k: int, l: int, omega: float) -> Quantity:
    if k < 0:
        raise ValueError(f"radial index k must be >= 0, got {k}")
    if l < 0:
        raise ValueError(f"orbital quantum number l must be >= 0, got {l}")
    if not omega > 0:
        raise ValueError(f"omega must be positive, got {omega}")
    return Quantity(
        value=omega * (2 * k + l + 1.5),
        unit="hartree",
        label=f"E_osc[k={k}, l={l}]",
        provenance=_PROV,
    )

def oscillator_levels(omega: float, l: int, n_states: int) -> tuple[Quantity, ...]:
    if n_states < 1:
        raise ValueError(f"n_states must be >= 1, got {n_states}")
    return tuple(oscillator_energy(k, l, omega) for k in range(n_states))
