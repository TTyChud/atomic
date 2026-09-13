import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const engineModeValue = { current: "remote" as "local" | "remote" };
const fetchThumbnailMock = vi.fn();

vi.mock("../engine/engine", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../engine/engine")>();
  return {
    ...actual,
    currentEngineMode: vi.fn(() => engineModeValue.current),
  };
});

vi.mock("./client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./client")>();
  return {
    ...actual,
    fetchThumbnail: (...args: unknown[]) => fetchThumbnailMock(...args),
  };
});

async function loadThumbnail() {
  vi.resetModules();
  return await import("./thumbnail");
}

beforeEach(() => {
  fetchThumbnailMock.mockReset();
  engineModeValue.current = "remote";
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("thumbnailMode", () => {
  it("serves URLs on a deployed origin and blobs on the device engine", async () => {
    const { thumbnailMode } = await loadThumbnail();
    expect(thumbnailMode()).toBe("url");
    engineModeValue.current = "local";
    expect(thumbnailMode()).toBe("blob");
  });
});

describe("thumbnailSrc", () => {
  it("passes the plain URL through in url mode", async () => {
    const { thumbnailSrc } = await loadThumbnail();
    const src = await thumbnailSrc(2, 1, 0, "he+", "complex", 96);
    expect(src).toBe("/api/thumbnail/2/1/0?system=he%2B&basis=complex&size=96");
    expect(fetchThumbnailMock).not.toHaveBeenCalled();
  });

  it("fetches bytes and mints an object URL in blob mode", async () => {
    engineModeValue.current = "local";
    const bytes = new Uint8Array([1, 2, 3]);
    fetchThumbnailMock.mockResolvedValue(new Blob([bytes]));
    const createObjectURL = vi.fn(() => "blob:mock-1");
    vi.stubGlobal("URL", Object.assign(Object(URL), { createObjectURL }));

    const { thumbnailSrc } = await loadThumbnail();
    const src = await thumbnailSrc(2, 1, 0, "he+", "complex", 96);

    expect(src).toBe("blob:mock-1");
    expect(fetchThumbnailMock).toHaveBeenCalledWith(
      "/api/thumbnail/2/1/0?system=he%2B&basis=complex&size=96",
    );
    expect(createObjectURL).toHaveBeenCalledTimes(1);
  });

  it("caches per URL so repeated gallery renders do not refetch", async () => {
    engineModeValue.current = "local";
    fetchThumbnailMock.mockResolvedValue(new Blob([new Uint8Array([9])]));
    vi.stubGlobal("URL", Object.assign(Object(URL), { createObjectURL: () => "blob:cached" }));

    const { thumbnailSrc } = await loadThumbnail();
    const a = await thumbnailSrc(3, 2, 1, "c", "real", 96);
    const b = await thumbnailSrc(3, 2, 1, "c", "real", 96);
    expect(a).toBe(b);
    expect(fetchThumbnailMock).toHaveBeenCalledTimes(1);

    await thumbnailSrc(3, 1, 0, "c", "real", 96);
    expect(fetchThumbnailMock).toHaveBeenCalledTimes(2);
  });
});
