import { useCallback, useState } from "react";

export const MIN_PLOT_WIDTH = 240;

export function plotWidth(measured: number, min: number = MIN_PLOT_WIDTH): number {
  return Math.max(min, Math.round(measured));
}

export function plotFit(
  measured: number,
  min: number = MIN_PLOT_WIDTH,
): { width: number; compact: boolean } {
  const width = plotWidth(measured);
  return { width, compact: width < min };
}

export function plotHeight(
  width: number,
  ratio: number,
  min: number,
  max: number,
): number {
  return Math.round(Math.min(max, Math.max(min, width * ratio)));
}

export function usePlotWidth(min: number = MIN_PLOT_WIDTH): {
  width: number;
  compact: boolean;
  ref: (el: HTMLElement | null) => void;
} {
  const [measured, setMeasured] = useState(min);
  const ref = useCallback((el: HTMLElement | null) => {
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => {
      setMeasured(entry.contentRect.width);
    });
    observer.observe(el);
    setMeasured(el.clientWidth - paddingX(el));
    return () => observer.disconnect();
  }, []);
  return { ...plotFit(measured, min), ref };
}

function paddingX(el: HTMLElement): number {
  const s = getComputedStyle(el);
  return parseFloat(s.paddingLeft) + parseFloat(s.paddingRight);
}
