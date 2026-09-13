/**
 * Where the physics engine lives.
 *
 * Same-origin by default: every URL is relative to wherever the UI itself is
 * served from (the local `atomic serve`, or the Fly container that mounts the
 * UI). A split deploy — static UI on Vercel, API on Fly — sets
 * `VITE_API_BASE` at build time, e.g. `https://atomic-api.fly.dev`:
 *
 *     cd web && VITE_API_BASE=https://atomic-api.fly.dev npm run build
 *
 * Trailing slashes are stripped so either spelling works.
 */
const RAW = (import.meta.env?.VITE_API_BASE as string | undefined) ?? "";

function stripTrailingSlash(url: string): string {
  return url.endsWith("/") ? url.slice(0, -1) : url;
}

export const API_BASE = stripTrailingSlash(RAW);

/** Join the API base with an absolute path ("/api/..." or "/ws/..."). */
export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}
