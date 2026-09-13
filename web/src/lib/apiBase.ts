
const RAW = (import.meta.env?.VITE_API_BASE as string | undefined) ?? "";

function stripTrailingSlash(url: string): string {
  return url.endsWith("/") ? url.slice(0, -1) : url;
}

export const API_BASE = stripTrailingSlash(RAW);

export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}
