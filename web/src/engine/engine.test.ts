import { describe, expect, it } from "vitest";
import { EngineBridge, engineMode, type EngineBoot } from "./engine";

describe("engineMode", () => {
  it("runs on-device for same-origin localhost", () => {
    expect(engineMode("localhost", "")).toBe("local");
    expect(engineMode("127.0.0.1", "")).toBe("local");
  });

  it("runs on-device on a hosted static UI with no API base", () => {
    // The staged runtime ships with the UI, so a static host runs the
    // physics in the visitor's browser — no backend involved.
    expect(engineMode("atomic.vercel.app", "")).toBe("local");
  });

  it("uses the network when a split-deploy base is configured", () => {
    expect(engineMode("localhost", "https://atomic-api.fly.dev")).toBe("remote");
    expect(engineMode("atomic.vercel.app", "https://api.example.com")).toBe(
      "remote",
    );
  });
});

/** Minimal Worker double: records posts, lets tests emit worker messages. */
class FakeWorker {
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onerror: (() => void) | null = null;
  readonly sent: unknown[] = [];

  postMessage(data: unknown): void {
    this.sent.push(data);
  }

  emit(data: unknown): void {
    this.onmessage?.({ data });
  }
}

function bootingBridge(worker: FakeWorker): EngineBridge {
  const bridge = new EngineBridge(() => worker as unknown as Worker);
  void bridge.start();
  return bridge;
}

describe("EngineBridge", () => {
  it("publishes boot phases from the worker and lands on ready", () => {
    const worker = new FakeWorker();
    const seen: EngineBoot[] = [];
    const bridge = new EngineBridge(() => worker as unknown as Worker);
    bridge.onChange((s) => seen.push(s));
    void bridge.start();

    worker.emit({ type: "boot-phase", phase: "runtime" });
    worker.emit({ type: "boot-phase", phase: "engine" });
    worker.emit({ type: "ready", version: "0.1.0" });

    expect(seen.map((s) => s.state)).toEqual([
      "idle",
      "loading",
      "loading",
      "ready",
    ]);
    expect(bridge.current).toEqual({ state: "ready", version: "0.1.0" });
    expect(bridge.ready).toBe(true);
  });

  it("posts the boot message exactly once for repeated start calls", () => {
    const worker = new FakeWorker();
    const bridge = bootingBridge(worker);
    void bridge.start();
    void bridge.start();
    expect(worker.sent).toEqual([{ type: "boot" }]);
  });

  it("correlates a result message back to its pending request", async () => {
    const worker = new FakeWorker();
    const bridge = bootingBridge(worker);
    worker.emit({ type: "ready", version: "0.1.0" });

    const pending = bridge.request("/api/systems");
    await flush();
    const posted = worker.sent.find((m) => (m as { type: string }).type === "request") as {
      id: number;
      method: string;
      url: string;
    };
    expect(posted.method).toBe("GET");
    expect(posted.url).toBe("/api/systems");

    worker.emit({
      type: "result",
      id: posted.id,
      result: {
        status: 200,
        content_type: "application/json",
        body_b64: btoa(JSON.stringify({ ok: true })),
      },
    });
    const res = await pending;
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: true });
  });

  it("rejects the pending request when the worker reports an error for it", async () => {
    const worker = new FakeWorker();
    const bridge = bootingBridge(worker);
    worker.emit({ type: "ready", version: "0.1.0" });

    const pending = bridge.request("/api/jobs/x/meta");
    await flush();
    const posted = worker.sent.at(-1) as { id: number };
    worker.emit({ type: "error", id: posted.id, error: "boom" });
    await expect(pending).rejects.toThrow("boom");
    // The failure is request-scoped: the engine itself stays ready.
    expect(bridge.ready).toBe(true);
  });

  it("treats an id-less error as a boot failure and allows a retry", async () => {
    const worker = new FakeWorker();
    const bridge = new EngineBridge(() => worker as unknown as Worker);
    const started = bridge.start();

    worker.emit({ type: "error", error: "pyodide exploded" });
    await expect(started).rejects.toThrow("pyodide exploded");
    expect(bridge.current).toEqual({ state: "error", error: "pyodide exploded" });

    // The failed boot is forgotten, so a retry re-posts the boot message.
    const retried = bridge.start();
    worker.emit({ type: "ready", version: "0.1.0" });
    await expect(retried).resolves.toBeUndefined();
    const boots = worker.sent.filter((m) => (m as { type: string }).type === "boot");
    expect(boots).toHaveLength(2);
  });

  it("ignores results for unknown ids instead of crashing", () => {
    const worker = new FakeWorker();
    bootingBridge(worker);
    expect(() =>
      worker.emit({
        type: "result",
        id: 9999,
        result: { status: 200, content_type: "application/json", body_b64: "" },
      }),
    ).not.toThrow();
  });
});

/** Flush the microtask chain so an awaited `bridge.request` posts its message. */
const flush = (): Promise<void> => new Promise((r) => setTimeout(r, 0));
