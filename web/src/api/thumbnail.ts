

import { currentEngineMode } from "../engine/engine";
import { fetchThumbnail, thumbnailUrl } from "./client";

const cache = new Map<string, string>();

export function thumbnailMode(): "url" | "blob" {
  return currentEngineMode() === "local" ? "blob" : "url";
}

export async function thumbnailSrc(
  n: number,
  l: number,
  m: number,
  system: string,
  basis: string,
  size: number,
  params?: {
    model?: "gsz" | "hf";
    config?: string | null;
    exchange?: boolean;
    pauli?: boolean;
  },
): Promise<string> {
  const url = thumbnailUrl(n, l, m, system, basis as "complex" | "real", size, params);
  if (thumbnailMode() === "url") return url;

  const hit = cache.get(url);
  if (hit) return hit;
  const blob = await fetchThumbnail(url);
  const objectUrl = URL.createObjectURL(blob);
  cache.set(url, objectUrl);
  return objectUrl;
}
