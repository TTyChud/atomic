/**
 * Main-thread bridge to the in-browser engine.
 *
 * The engine is a Web Worker running the *real* Python package via Pyodide —
 * the same `atomic` package, the same FastAPI app, the same schemas a server
 * deploy would run. Nothing is stored, nothing is shared between users: the
 * physics happens on the device, so the UI can be served as pure static files
 * from anywhere (localhost, a hosted static host) with no backend at all.
 *
 * A server is still reachable when one exists: `VITE_API_BASE` at build time
 * (split deploy) or `?engine=server` at runtime point every engine call at
 * the network instead. `engineMode` encodes that policy.
 */

import { API_BASE } from "../lib/apiBase";
import { asgiResponse, type AsgiResult } from "./engineCodec";

export type EngineMode = "local" | "remote";

export type EngineBoot =
  | { state: "idle" }
  | { state: "loading"; phase: "runtime" | "engine" }
  | { state: "ready"; version: string }
  | { state: "error"; error: string };

/**
 * Device mode unless a server is explicitly configured via a split-deploy API
 * base. The UI ships the staged engine runtime alongside itself, so any host
 * — localhost or a public static host — can run the physics on-device.
 */
export function engineMode(_hostname: string, apiBase: string): EngineMode {
  return apiBase === "" ? "local" : "remote";
}

/** Hosts where an `atomic serve` process may actually be listening. */
const SERVER_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]", "0.0.0.0"]);

/** True when falling back to a same-origin server could plausibly work. */
export function hasLocalServer(hostname: string): boolean {
  return SERVER_HOSTS.has(hostname);
}

/**
 * Effective mode, honoring a ?engine=server|device URL override. `server`
 * lets a locally-running `atomic serve` answer instead of booting the
 * in-browser runtime; `device` forces the engine anywhere.
 */
export function currentEngineMode(): EngineMode {
  const param = new URLSearchParams(
    globalThis.location?.search ?? "",
  ).get("engine");
  if (param === "device") return "local";
  if (param === "server") return "remote";
  return engineMode(globalThis.location?.hostname ?? "", API_BASE);
}

interface Pending {
  resolve: (value: Response) => void;
  reject: (reason: Error) => void;
}

export class EngineBridge {
  private worker: Worker | null = null;
  private boot: Promise<void> | null = null;
  private state: EngineBoot = { state: "idle" };
  private listeners = new Set<(s: EngineBoot) => void>();
  private pending = new Map<number, Pending>();
  private nextId = 1;

  /**
   * `makeWorker` is injectable so tests can drive the protocol without a
   * real Worker (which the node test runner cannot construct).
   */
  constructor(
    private makeWorker: () => Worker = () =>
      new Worker(new URL("./engineWorker.ts", import.meta.url), {
        type: "module",
      }),
  ) {}

  /** Subscribe to boot-state changes; fires immediately with the current state. */
  onChange(fn: (s: EngineBoot) => void): () => void {
    this.listeners.add(fn);
    fn(this.state);
    return () => {
      this.listeners.delete(fn);
    };
  }

  get current(): EngineBoot {
    return this.state;
  }

  /** True once the engine worker is booted and answering. */
  get ready(): boolean {
    return this.state.state === "ready";
  }

  private ensureWorker(): Worker {
    if (this.worker) return this.worker;
    const w = this.makeWorker();
    w.onmessage = (ev: MessageEvent) => this.receive(ev.data);
    w.onerror = () => {
      // Forget the boot so a crashed worker can be retried later.
      this.boot = null;
      this.setState({ state: "error", error: "engine worker crashed" });
    };
    this.worker = w;
    return w;
  }

  /** Boot the runtime and the engine package (idempotent). */
  start(): Promise<void> {
    if (this.boot) return this.boot;
    // A previous failed attempt leaves a terminal state behind; reset so the
    // subscriber below cannot fire on stale state (it would call `off` before
    // assignment) and so listeners see a fresh loading cycle on retry.
    if (this.state.state === "error") this.setState({ state: "idle" });
    const w = this.ensureWorker();
    this.boot = new Promise<void>((resolve, reject) => {
      let off: (() => void) | undefined;
      off = this.onChange((s) => {
        if (s.state === "ready") {
          off?.();
          resolve();
        } else if (s.state === "error") {
          off?.();
          reject(new Error(s.error));
        }
      });
      w.postMessage({ type: "boot" });
    });
    return this.boot;
  }

  /** Perform an HTTP request against the in-browser engine. */
  async request(url: string, init?: RequestInit): Promise<Response> {
    await this.start();
    const id = this.nextId++;
    const body = typeof init?.body === "string" ? init.body : null;
    return new Promise<Response>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ensureWorker().postMessage({
        type: "request",
        id,
        method: init?.method ?? "GET",
        url,
        body,
      });
    });
  }

  private setState(next: EngineBoot): void {
    this.state = next;
    for (const fn of this.listeners) fn(next);
  }

  private receive(msg: Record<string, unknown>): void {
    const type = msg.type as string;
    if (type === "boot-phase") {
      this.setState({ state: "loading", phase: msg.phase as "runtime" | "engine" });
      return;
    }
    if (type === "ready") {
      this.setState({ state: "ready", version: String(msg.version ?? "") });
      return;
    }
    if (type === "error") {
      const id = msg.id as number | undefined;
      const error = new Error(String(msg.error ?? "engine error"));
      if (id !== undefined && this.pending.has(id)) {
        this.pending.get(id)!.reject(error);
        this.pending.delete(id);
      } else {
        this.setState({ state: "error", error: error.message });
        this.boot = null;
      }
      return;
    }
    if (type === "result") {
      const id = msg.id as number;
      const entry = this.pending.get(id);
      if (!entry) return;
      this.pending.delete(id);
      entry.resolve(asgiResponse(msg.result as AsgiResult));
    }
  }
}

export const engine = new EngineBridge();
