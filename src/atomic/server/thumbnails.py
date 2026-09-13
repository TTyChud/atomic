import io
from functools import lru_cache

import matplotlib
from matplotlib import image as mpl_image

from atomic.atoms import atom_for_key, is_atom_key, parse_config
from atomic.plane import hf_plane_grid, plane_grid, screened_plane_grid
from atomic.systems import get_system

matplotlib.use("Agg")

GAMMA = 0.5


@lru_cache(maxsize=512)
def render_thumbnail(
    n: int,
    l: int,
    m: int,
    system: str,
    basis: str,
    size: int,
    model: str = "gsz",
    config: str | None = None,
    exchange: bool = True,
    pauli: bool = True,
) -> bytes:
    if is_atom_key(system):
        element = atom_for_key(system)
        if model == "hf":
            cfg = parse_config(config) if config else None
            pg = hf_plane_grid(
                element.z, element.z, n, l, m,
                quantity="density", basis=basis, resolution=size,
                config=cfg, exchange=exchange, pauli=pauli,
            )
        else:
            pg = screened_plane_grid(
                element.z, element.z, n, l, m,
                quantity="density", basis=basis, resolution=size,
            )
    else:
        sys_ = get_system(system)
        pg = plane_grid(
            n, l, m, quantity="density", basis=basis,
            Z=sys_.Z, mu_ratio=sys_.mu_ratio.value, resolution=size,
        )
    rho = pg.values
    vmax = float(rho.max())
    t = (rho / vmax) ** GAMMA if vmax > 0.0 else rho
    buf = io.BytesIO()
    mpl_image.imsave(buf, t[::-1], cmap="inferno", vmin=0.0, vmax=1.0, format="png")
    return buf.getvalue()
