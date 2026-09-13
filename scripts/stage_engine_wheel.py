#!/usr/bin/env python3
"""Stage the in-browser engine runtime.

Builds the `atomic` wheel, copies the Pyodide distribution from web/node_modules,
and writes a manifest naming the wheel, into:

    web/public/atomic-engine/
        atomic-<version>-py3-none-any.whl
        pyodide/            (runtime: pyodide.asm.wasm, packages, ...)
        manifest.json

`web/dist` copies public/ verbatim, so the dev server and the deployed UI
serve /atomic-engine/... identically. The engine worker fetches the wheel
from the manifest, installs it with micropip, and runs the real package.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
STAGE = WEB / "public" / "atomic-engine"
PYODIDE_NM = WEB / "node_modules" / "pyodide"

# Runtime files the browser actually requests. The npm package also ships
# node-only entry points and maps for them; the engine worker loads the .mjs
# directly, so only the core wasm, lock file, and stdlib are needed.
PYODIDE_KEEP = [
    "pyodide.asm.mjs",
    "pyodide.asm.wasm",
    "pyodide-lock.json",
    "python_stdlib.zip",
]

# Python packages the engine imports, resolved to their full dependency
# closure from pyodide-lock.json and downloaded next to the runtime so
# loadPackage never touches a third-party CDN at runtime. matplotlib is
# deliberately absent: thumbnails are encoded by the stdlib PNG writer in
# atomic/server/png.py, which keeps ~9 MB (matplotlib + Pillow + fonttools +
# contourpy + kiwisolver + cycler + pyparsing) out of the visitor download.
PYODIDE_PKGS = ["numpy", "scipy", "fastapi", "micropip"]


def _lock() -> dict:
    return json.loads((PYODIDE_NM / "pyodide-lock.json").read_text())


def _pyodide_version() -> str:
    return json.loads((PYODIDE_NM / "package.json").read_text())["version"]


def wheel_closure(lock: dict, names: list[str]) -> list[str]:
    """file_names for `names` and everything they depend on, per the lock."""
    packages = lock["packages"]

    def key_for(name: str) -> str | None:
        for candidate in (name, name.replace("-", "_"), name.replace("_", "-")):
            if candidate in packages:
                return candidate
        return None

    files: list[str] = []
    stack = list(names)
    seen: set[str] = set()
    while stack:
        name = stack.pop()
        key = key_for(name)
        if key is None:
            print(f"warning: {name} not in pyodide-lock.json", file=sys.stderr)
            continue
        if key in seen:
            continue
        seen.add(key)
        entry = packages[key]
        file_name = entry.get("file_name")
        if file_name:
            files.append(file_name)
        stack.extend(entry.get("depends", []))
    return sorted(set(files))


def download_wheels(lock: dict) -> int:
    files = wheel_closure(lock, PYODIDE_PKGS)
    base = f"https://cdn.jsdelivr.net/pyodide/v{_pyodide_version()}/full/"
    total = 0
    for file_name in files:
        dest = STAGE / "pyodide" / file_name
        if dest.exists():
            total += dest.stat().st_size
            continue
        print(f"  fetching {file_name}")
        with urllib.request.urlopen(base + file_name) as resp, open(dest, "wb") as out:
            total += out.write(resp.read())
    print(f"  {len(files)} package wheels, {total / 1e6:.1f} MB")
    return 0


def build_wheel() -> Path:
    """Build the wheel and copy it into the staging dir (returns the copy)."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp)],
            cwd=ROOT,
            check=True,
        )
        wheel = next(Path(tmp).glob("atomic-*.whl"))
        for old in STAGE.glob("atomic-*.whl"):
            old.unlink()
        dest = STAGE / wheel.name
        shutil.copy2(wheel, dest)
        return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyodide-only",
        action="store_true",
        help="skip the wheel build; refresh the runtime files only",
    )
    args = parser.parse_args()

    if not PYODIDE_NM.is_dir():
        print("pyodide not installed; run: cd web && npm install", file=sys.stderr)
        return 1

    STAGE.mkdir(parents=True, exist_ok=True)
    pydir = STAGE / "pyodide"
    if pydir.exists():
        shutil.rmtree(pydir)
    pydir.mkdir()

    for name in PYODIDE_KEEP:
        src = PYODIDE_NM / name
        if not src.exists():
            print(f"warning: pyodide package missing {name}", file=sys.stderr)
            continue
        shutil.copy2(src, pydir / name)

    lock = _lock()
    download_wheels(lock)

    wheel_name = None
    if not args.pyodide_only:
        wheel_name = build_wheel().name

    manifest_path = STAGE / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if wheel_name:
        manifest["wheel"] = wheel_name
    manifest.setdefault("wheel", None)
    manifest["pyodide"] = "pyodide/"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"staged engine runtime in {STAGE}")
    print(f"  wheel: {manifest['wheel']}")
    return 0 if manifest["wheel"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
