

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

export function asgiResponse(msg: AsgiResult): Response {
  const bytes = base64ToBytes(msg.body_b64);
  return new Response(bytes.slice().buffer, {
    status: msg.status,
    headers: { "Content-Type": msg.content_type },
  });
}
