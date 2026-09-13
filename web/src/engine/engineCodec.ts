/**
 * Codec for the in-browser engine worker.
 *
 * The worker drives the real Python ASGI app and answers with
 * `{status, content_type, body_b64}` — one shape for JSON routes and binary
 * job channels alike, so status codes (404, 409, 422) survive the boundary.
 * Only the base64 translation lives here.
 */

export function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

export function base64ToBytes(b64: string): Uint8Array {
  const binary = atob(b64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}

export interface AsgiResult {
  status: number;
  content_type: string;
  body_b64: string;
}

/** Rebuild the exact HTTP Response the ASGI app produced. */
export function asgiResponse(msg: AsgiResult): Response {
  const bytes = base64ToBytes(msg.body_b64);
  return new Response(bytes.slice().buffer, {
    status: msg.status,
    headers: { "Content-Type": msg.content_type },
  });
}
