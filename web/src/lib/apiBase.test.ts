import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/**
 * apiBase reads VITE_API_BASE once at module load, so each case re-imports
 * the module with a fresh env stub.
 */
async function loadWith(env: Record<string, string | undefined>) {
  vi.resetModules();
  vi.stubEnv("VITE_API_BASE", env.VITE_API_BASE ?? "");
  return await import("./apiBase");
}

describe("apiBase", () => {
  beforeEach(() => {
    vi.unstubAllEnvs();
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("stays same-origin when no base is configured", async () => {
    const { API_BASE, apiUrl } = await loadWith({});
    expect(API_BASE).toBe("");
    expect(apiUrl("/api/systems")).toBe("/api/systems");
  });

  it("prefixes the configured engine for split deploys", async () => {
    const { API_BASE, apiUrl } = await loadWith({
      VITE_API_BASE: "https://atomic-api.fly.dev",
    });
    expect(API_BASE).toBe("https://atomic-api.fly.dev");
    expect(apiUrl("/api/health")).toBe("https://atomic-api.fly.dev/api/health");
  });

  it("strips a trailing slash from the configured base", async () => {
    const { apiUrl } = await loadWith({
      VITE_API_BASE: "https://atomic-api.fly.dev/",
    });
    expect(apiUrl("/api/systems")).toBe(
      "https://atomic-api.fly.dev/api/systems",
    );
  });

  it("prefixes job routes for split deploys (the getJobMeta regression)", async () => {
    const { apiUrl } = await loadWith({
      VITE_API_BASE: "https://atomic-api.fly.dev",
    });
    expect(apiUrl("/api/jobs/abc/meta")).toBe(
      "https://atomic-api.fly.dev/api/jobs/abc/meta",
    );
  });
});
