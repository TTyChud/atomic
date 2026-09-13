import pytest
from fastapi.testclient import TestClient

from atomic.server.app import create_app
from atomic.server.thumbnails import render_thumbnail

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture()
def client():
    with TestClient(create_app()) as c:
        yield c


def test_render_thumbnail_returns_png_bytes():
    render_thumbnail.cache_clear()
    png = render_thumbnail(2, 1, 0, "h", "complex", 64)
    assert isinstance(png, bytes)
    assert png[:8] == PNG_MAGIC


def test_render_thumbnail_caches():
    render_thumbnail.cache_clear()
    render_thumbnail(1, 0, 0, "h", "complex", 48)
    render_thumbnail(1, 0, 0, "h", "complex", 48)
    assert render_thumbnail.cache_info().hits == 1


def test_thumbnail_endpoint(client):
    r = client.get("/api/thumbnail/2/1/0?size=48")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert "max-age" in r.headers["cache-control"]
    assert r.content[:8] == PNG_MAGIC


def test_thumbnail_states_differ(client):
    a = client.get("/api/thumbnail/1/0/0?size=48").content
    b = client.get("/api/thumbnail/2/1/0?size=48").content
    assert a != b


def test_thumbnail_validation(client):
    assert client.get("/api/thumbnail/2/1/0?size=10").status_code == 422
    assert client.get("/api/thumbnail/2/1/0?size=999").status_code == 422
    assert client.get("/api/thumbnail/2/1/0?basis=cartoon").status_code == 422
    assert client.get("/api/thumbnail/1/1/0").status_code == 422
    assert client.get("/api/thumbnail/2/1/0?system=unobtainium").status_code == 422


def test_screened_thumbnail_returns_png(client):
    r = client.get("/api/thumbnail/2/1/0?system=c&size=48")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == PNG_MAGIC


def test_hf_thumbnail_returns_png(client):
    r = client.get("/api/thumbnail/2/1/0?system=c&model=hf&size=48")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == PNG_MAGIC


def test_screened_and_hf_thumbnails_differ(client):
    a = client.get("/api/thumbnail/2/1/0?system=c&size=48").content
    b = client.get("/api/thumbnail/2/1/0?system=c&model=hf&size=48").content
    assert a != b


def test_hf_thumbnail_refuses_an_empty_subshell(client):
    r = client.get("/api/thumbnail/3/2/0?system=c&model=hf&size=48")
    assert r.status_code == 422
    assert "not occupied" in r.json()["detail"]


def test_sulfur_has_no_gsz_thumbnail_but_hf_is_fine(client):
    assert client.get("/api/thumbnail/1/0/0?system=s&size=48").status_code == 400
    r = client.get("/api/thumbnail/1/0/0?system=s&model=hf&size=48")
    assert r.status_code == 200
    assert r.content[:8] == PNG_MAGIC


def test_thumbnail_rejects_an_unknown_model(client):
    assert client.get("/api/thumbnail/2/1/0?system=c&model=wat").status_code == 422


def test_inferno_encoding_matches_matplotlib_pixel_for_pixel():
    """The stdlib encoder must reproduce matplotlib's rendering exactly.

    This is the evidence that dropping matplotlib from the runtime cost
    nothing visually. matplotlib stays a dev-only dependency for this test.
    """
    import io

    import numpy as np
    import pytest
    from matplotlib import image as mpl_image
    from PIL import Image

    from atomic.plane import plane_grid
    from atomic.server.png import encode_png, inferno_rgb

    pytest.importorskip("matplotlib")

    pg = plane_grid(2, 1, 0, quantity="density", basis="complex",
                    Z=1, mu_ratio=1.0, resolution=64)
    rho = pg.values
    vmax = float(rho.max())
    t = (rho / vmax) ** 0.5 if vmax > 0.0 else rho

    buf = io.BytesIO()
    mpl_image.imsave(buf, t[::-1], cmap="inferno", vmin=0.0, vmax=1.0, format="png")
    expected = np.asarray(Image.open(buf).convert("RGB"))

    flipped = t[::-1]
    produced = np.asarray(Image.open(
        io.BytesIO(encode_png(inferno_rgb(flipped.tolist()),
                              flipped.shape[1], flipped.shape[0]))
    ).convert("RGB"))

    assert produced.shape == expected.shape
    assert np.array_equal(produced, expected)
