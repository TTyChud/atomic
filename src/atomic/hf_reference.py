
import json
from importlib import resources

_FILENAME = "hf_reference_energies.json"

HF_REFERENCE = json.loads(
    resources.files("atomic.data").joinpath(_FILENAME).read_text(encoding="utf-8")
)

__all__ = ["HF_REFERENCE", "load_hf_reference"]

def load_hf_reference(symbol: str) -> dict:
    try:
        return HF_REFERENCE["values"][symbol]
    except KeyError as exc:
        raise KeyError(
            f"no vendored Hartree-Fock reference energy for {symbol!r}; "
            f"available: {sorted(HF_REFERENCE['values'])}"
        ) from exc
