import { useEffect, useState } from "react";
import { currentEngineMode, engine, type EngineBoot } from "../engine/engine";
import type { ViewMode } from "../lib/urlState";
import { isNarrow, useViewport } from "../lib/viewport";
import { VIEW_HINTS } from "../lib/viewHints";
import { useAppStore } from "../state/store";
import { Shortcuts } from "./Shortcuts";
import { TourMenu } from "./TourMenu";

const TABS: { value: ViewMode; label: string }[] = [
  { value: "cloud", label: "Cloud" },
  { value: "plane", label: "Plane" },
  { value: "radial", label: "Radial" },
  { value: "levels", label: "Levels" },
  { value: "spectrum", label: "Spectrum" },
  { value: "whatif", label: "What-If" },
  { value: "forcelaw", label: "Force Law" },
];

function CopyLink({ narrow }: { narrow: boolean }) {
  const [said, setSaid] = useState<"idle" | "copied" | "failed">("idle");
  useEffect(() => {
    if (said === "idle") return;
    const t = setTimeout(() => setSaid("idle"), 1600);
    return () => clearTimeout(t);
  }, [said]);
  return (
    <button
      className="topbar-btn"
      type="button"
      title="Copy a link to this exact state"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(window.location.href);
          setSaid("copied");
        } catch {
          setSaid("failed");
        }
      }}
    >
      {said === "idle" ? (narrow ? "link" : "copy link") : said === "copied" ? "copied" : "failed"}
    </button>
  );
}

const ENGINE_LABEL: Record<EngineBoot["state"], string> = {
  idle: "device engine",
  loading: "device engine · loading",
  ready: "device engine",
  error: "device engine · error",
};

function EngineBadge() {
  const [boot, setBoot] = useState<EngineBoot>(engine.current);
  useEffect(() => engine.onChange(setBoot), []);
  if (currentEngineMode() === "remote") return null;
  const label = ENGINE_LABEL[boot.state];
  const title =
    boot.state === "ready"
      ? `Physics computed on this device by atomic ${boot.version} (Python via WebAssembly). No server, nothing stored.`
      : boot.state === "loading"
        ? "Booting the Python runtime on this device…"
        : boot.state === "error"
          ? "The on-device engine failed to start; reload to retry."
          : "The physics engine will run on this device.";
  return (
    <span
      className={
        boot.state === "ready" ? "engine-badge engine-ready" : "engine-badge"
      }
      title={title}
    >
      {label}
    </span>
  );
}

export function TopBar() {
  const { view, setView, n, l, m, system } = useAppStore();
  const { width } = useViewport();
  return (
    <header className="topbar">
      <span className="brand">atomic</span>
      <span className="crumb">
        {system} · n={n} l={l} m={m}
      </span>
      <nav className="tabs">
        {TABS.map((t) => (
          <button
            key={t.value}
            className={view === t.value ? "tab tab-active" : "tab"}
            title={VIEW_HINTS[t.value]?.what}
            onClick={() => setView(t.value)}
          >
            {t.label}
          </button>
        ))}
      </nav>
      <EngineBadge />
      <CopyLink narrow={isNarrow(width)} />
      <Shortcuts />
      <TourMenu />
    </header>
  );
}
