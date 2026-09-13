import { describe, expect, it } from "vitest";
import {
  asgiResponse,
  base64ToBytes,
  bytesToBase64,
} from "./engineCodec";

describe("base64 helpers", () => {
  it("round-trips binary payloads", () => {
    const bytes = new Uint8Array([0, 1, 2, 250, 251, 252, 253, 254, 255]);
    expect(base64ToBytes(bytesToBase64(bytes))).toEqual(bytes);
  });
});

describe("asgiResponse", () => {
  it("rebuilds a JSON response with its true status", async () => {
    const body = bytesToBase64(new TextEncoder().encode('{"detail":"nope"}'));
    const res = asgiResponse({ status: 404, content_type: "application/json", body_b64: body });
    expect(res.status).toBe(404);
    expect(res.headers.get("Content-Type")).toBe("application/json");
    expect(await res.json()).toEqual({ detail: "nope" });
  });

  it("rebuilds a binary channel response", async () => {
    const floats = new Float32Array([1.5, -2.25, 3.125]);
    const b64 = bytesToBase64(new Uint8Array(floats.buffer));
    const res = asgiResponse({
      status: 200,
      content_type: "application/octet-stream",
      body_b64: b64,
    });
    const buf = await res.arrayBuffer();
    expect(new Float32Array(buf)).toEqual(floats);
  });
});
