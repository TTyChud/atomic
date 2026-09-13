/**
 * Thumbnails through the standard transport.
 *
 * `thumbnailUrl()` produces a URL for `<img src>`; over the network that is
 * exactly right. In device mode the engine lives in a Worker, which `<img>`
 * cannot reach — so the gallery fetches the bytes through the same
 * `request()` transport and renders object URLs instead. Results are cached
 * per URL for the page's lifetime.
 */

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
