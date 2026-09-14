
import { bytesToBase64 } from "./engineCodec";

interface Pyodide {
  runPythonAsync: (code: string) => Promise<unknown>;
  loadPackage: (names: string[]) => Promise<void>;
}

interface AsgiResult {
  status: number;
  content_type: string;
  body_b64: string;
}

function q(s: string): string {
  return JSON.stringify(s);
}

const STAGED_ROOT = new URL("/atomic-engine/", import.meta.url).href;

const BOOT_PY = `
from atomic.server.app import create_app
from atomic.server.engine_shim import InlineExecutor, install_inline_threadpool, dispatch
install_inline_threadpool()
_app = create_app()
_app.state.executor = InlineExecutor()
`;

let booting: Promise<void> | null = null;
let pyodide: Pyodide | null = null;

function phase(p: "runtime" | "engine"): void {
  postMessage({ type: "boot-phase", phase: p });
}

async function installWheel(py: Pyodide): Promise<void> {
  const manifest = (await (await fetch(`${STAGED_ROOT}manifest.json`)).json()) as {
    wheel: string;
  };
  const wheelUrl = new URL(`/atomic-engine/${manifest.wheel}`, self.location.origin).href;
  await py.runPythonAsync(
    `import micropip; await micropip.install(${q(wheelUrl)})`,
  );
}

async function boot(): Promise<void> {
  phase("runtime");
  const loadPyodide = (await import("pyodide")).loadPyodide;
  const py = (await loadPyodide({
    indexURL: `${STAGED_ROOT}pyodide/`,
    stdout: (s: string) => console.log("[engine]", s),
    stderr: (s: string) => console.warn("[engine]", s),
  })) as unknown as Pyodide;

  phase("engine");

  await py.loadPackage(["numpy", "scipy", "fastapi", "micropip"]);
  await installWheel(py);

  await py.runPythonAsync(BOOT_PY);

  pyodide = py;
  const version = (await py.runPythonAsync("import atomic; atomic.__version__")) as string;
  postMessage({ type: "ready", version: String(version) });
}

async function getPyodide(): Promise<Pyodide> {
  if (!booting) booting = boot();
  return booting.then(() => pyodide as Pyodide);
}

async function dispatch(
  py: Pyodide,
  method: string,
  url: string,
  body: string | null,
): Promise<AsgiResult> {
  const bodyB64 =
    body !== null ? bytesToBase64(new TextEncoder().encode(body)) : null;
  const resultJson = (await py.runPythonAsync(
    `await dispatch(_app, ${q(method)}, ${q(url)}, ${bodyB64 === null ? "None" : q(bodyB64)})`,
  )) as string;
  return JSON.parse(resultJson) as AsgiResult;
}

self.onmessage = async (ev: MessageEvent) => {
  const msg = ev.data as {
    type: string;
    id?: number;
    method?: string;
    url?: string;
    body?: string | null;
  };
  if (msg.type === "boot") {
    try {
      await getPyodide();
    } catch (e) {
      booting = null;
      postMessage({ type: "error", error: String(e) });
    }
    return;
  }
  if (msg.type === "request") {
    try {
      const py = await getPyodide();
      const result = await dispatch(
        py,
        msg.method ?? "GET",
        msg.url ?? "/",
        msg.body ?? null,
      );
      postMessage({ type: "result", id: msg.id, result });
    } catch (e) {
      postMessage({ type: "error", id: msg.id, error: String(e) });
    }
  }
};
