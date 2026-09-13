import { useEffect, useRef } from "react";
import { useAppStore } from "../state/store";
import { VIEW_HINTS } from "../lib/viewHints";
import { VIEW_OPTIONS } from "./Controls";

export const SHORT: Record<string, string> = {
  cloud: "3D cloud",
  plane: "2D slice",
  radial: "Radial",
  levels: "Levels",
  spectrum: "Spectrum",
  whatif: "What-If: constants",
  forcelaw: "What-If: force law",
};

export function ViewTabs() {
  const view = useAppStore((s) => s.view);
  const setView = useAppStore((s) => s.setView);
  const live = useRef<HTMLButtonElement | null>(null);
  useEffect(() => {
    live.current?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [view]);
  return (
    <nav className="view-tabs" data-tour="view-list" aria-label="view">
      {VIEW_OPTIONS.map((v) => (
        <button
          key={v.value}
          ref={view === v.value ? live : undefined}
          type="button"
          className={`view-tab${view === v.value ? " view-tab-on" : ""}`}
          aria-pressed={view === v.value}
          aria-label={v.label}
          title={VIEW_HINTS[v.value]?.what}
          onClick={() => setView(v.value)}
        >
          {SHORT[v.value] ?? v.label}
        </button>
      ))}
    </nav>
  );
}
