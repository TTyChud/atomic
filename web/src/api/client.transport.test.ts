import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/**
 * Tests for the single transport every engine call goes through.
 *
 * `client.ts` routes relative URLs to the in-browser engine on localhost
 * (honoring the ?engine= override) and to plain fetch everywhere else.
 * The routing decision runs at call time against `globalThis.location`, so
 * these tests stub the location and the engine bridge, then re-import the
 * module fresh.
 */

const engineRequest = vi.fn();

vi.mock("../engine/engine", () => ({
  engine: {
    request: (...args: unknown[]) => engineRequest(...args),
  },
  currentEngineMode: vi.fn(() => "local"),
  engineMode: vi.fn(() => "local"),
}));

async function loadClient() {
  vi.resetModules();
  return await import("./client");
}

function setLocation(url: string | undefined): void {
  if (url === undefined) {
    vi.unstubAllGlobals();
    return;
  }
  vi.stubGlobal("location", new URL(url) as unknown as Location);
}

beforeEach(() => {
  engineRequest.mockReset();
  vi.unstubAllGlobals();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("transport routing", () => {
  it("routes relative calls to the device engine on localhost", async () => {
    setLocation("http://localhost:5173/");
    const { getSystems } = await loadClient();
    engineRequest.mockResolvedValue(new Response('{"systems":[]}'));

    await getSystems();

    expect(engineRequest).toHaveBeenCalledWith(
      "/api/systems",
      undefined,
    );
  });

  it("?engine=server forces the network even on localhost", async () => {
    setLocation("http://localhost:5173/?engine=server");
    const fetchSpy = vi.fn().mockResolvedValue(new Response("{}", { status: 200 }));
    vi.stubGlobal("fetch", fetchSpy);
    const { getSystems } = await loadClient();
    const { currentEngineMode } = await import("../engine/engine");
    (currentEngineMode as ReturnType<typeof vi.fn>).mockReturnValue("remote");

    await getSystems();

    expect(engineRequest).not.toHaveBeenCalled();
    expect(fetchSpy).toHaveBeenCalledWith(
      "/api/systems",
      undefined,
    );
  });

  it("?engine=device forces the engine even on a deployed origin", async () => {
    setLocation("https://atomic.vercel.app/?engine=device");
    const { getSystems } = await loadClient();
    const { currentEngineMode } = await import("../engine/engine");
    (currentEngineMode as ReturnType<typeof vi.fn>).mockReturnValue("local");
    engineRequest.mockResolvedValue(new Response("{}"));

    await getSystems();

    expect(engineRequest).toHaveBeenCalled();
  });

  it("routes absolute URLs over the network even in device mode", async () => {
    setLocation("http://localhost:5173/");
    const fetchSpy = vi.fn().mockResolvedValue(new Response("", { status: 200 }));
    vi.stubGlobal("fetch", fetchSpy);
    const { fetchThumbnail } = await loadClient();
    const { currentEngineMode } = await import("../engine/engine");
    (currentEngineMode as ReturnType<typeof vi.fn>).mockReturnValue("local");

    const blob = await fetchThumbnail("https://api.example.com/api/thumbnail/2/1/0");

    expect(engineRequest).not.toHaveBeenCalled();
    expect(fetchSpy).toHaveBeenCalledWith(
      "https://api.example.com/api/thumbnail/2/1/0",
      undefined,
    );
    expect((await blob.arrayBuffer()).byteLength).toBe(0);
  });

  it("carries job meta through the same routing (split-deploy regression)", async () => {
    setLocation("https://atomic.vercel.app/");
    const fetchSpy = vi.fn().mockResolvedValue(
      new Response('{"kind":"sample"}', { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchSpy);
    const { getJobMeta } = await loadClient();
    const { currentEngineMode } = await import("../engine/engine");
    (currentEngineMode as ReturnType<typeof vi.fn>).mockReturnValue("remote");

    const meta = await getJobMeta("abc123");

    expect(meta.kind).toBe("sample");
    expect(fetchSpy).toHaveBeenCalledWith("/api/jobs/abc123/meta", undefined);
  });

  it("surfaces HTTP errors with the URL and status", async () => {
    setLocation("https://atomic.vercel.app/");
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("{}", { status: 404 })),
    );
    const { getJobMeta } = await loadClient();
    const { currentEngineMode } = await import("../engine/engine");
    (currentEngineMode as ReturnType<typeof vi.fn>).mockReturnValue("remote");

    await expect(getJobMeta("nope")).rejects.toThrow(/HTTP 404/);
  });
});
