import { afterEach, describe, expect, it, vi } from "vitest";
import { getLevels, getRadial, getState } from "./client";

let lastUrl = "";

function stubFetch() {
  lastUrl = "";
  // Pin the network path (?engine=server) — this suite is about URL shapes,
  // not transport selection.
  vi.stubGlobal(
    "location",
    new URL("https://atomic.test/?engine=server") as unknown as Location,
  );
  vi.stubGlobal("fetch", (url: string) => {
    lastUrl = url;
    return Promise.resolve({ ok: true, json: () => Promise.resolve({}) } as Response);
  });
}

afterEach(() => vi.unstubAllGlobals());

const CALLS: [string, () => Promise<unknown>][] = [
  ["getState", () => getState(2, 1, 0, "he+")],
  ["getRadial", () => getRadial(2, 1, "he+")],
  ["getLevels", () => getLevels("he+", 6, false)],
];

describe("a system key containing '+' survives the trip to the server", () => {
  it.each(CALLS)("%s encodes it", async (_name, call) => {
    stubFetch();
    await call();
    expect(lastUrl).toContain("system=he%2B");
    expect(lastUrl).not.toContain("system=he+");
  });

  it("leaves a key that needs no encoding alone", () => {
    stubFetch();
    void getState(3, 1, -1, "mu-h");
    expect(lastUrl).toContain("system=mu-h");
  });
});
