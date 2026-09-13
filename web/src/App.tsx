import { Suspense, lazy } from "react";
import { useAppStore } from "./state/store";
import { Controls } from "./components/Controls";
import { GalleryStrip } from "./components/GalleryStrip";
import { InfoPanel } from "./components/InfoPanel";
import { LevelsView } from "./components/LevelsView";
import { MobileSheet } from "./components/MobileSheet";
import { PlaneView } from "./components/PlaneView";
import { RadialView } from "./components/RadialView";
import { SpectrumView } from "./components/SpectrumView";
import { TopBar } from "./components/TopBar";
import { TourInvite } from "./components/TourInvite";
import { TourPanel } from "./components/TourPanel";
import { TourSpotlight } from "./components/TourSpotlight";
import { ViewTabs } from "./components/ViewTabs";
import { WhatIfView } from "./components/WhatIfView";
import { ForceLawView } from "./components/ForceLawView";
import { snapHeight } from "./lib/sheet";
import { isNarrow, useViewport } from "./lib/viewport";
import type { CSSProperties } from "react";

const CloudView = lazy(() =>
  import("./components/CloudView").then((m) => ({ default: m.CloudView })),
);

function Stage() {
  const view = useAppStore((s) => s.view);
  return (
    <>
      {view === "cloud" && (
        <Suspense fallback={<p className="hint-block">Loading the 3-D stage…</p>}>
          <CloudView />
        </Suspense>
      )}
      {view === "plane" && <PlaneView />}
      {view === "radial" && <RadialView />}
      {view === "levels" && <LevelsView />}
      {view === "spectrum" && <SpectrumView />}
      {view === "whatif" && <WhatIfView />}
      {view === "forcelaw" && <ForceLawView />}
    </>
  );
}

function DesktopShell() {
  return (
    <div className="app">
      <TopBar />
      <TourInvite />
      <div className="app-body">
        <aside className="app-side">
          <Controls />
          <InfoPanel />
        </aside>
        <main className="app-stage">
          <Stage />
          <GalleryStrip />
          <TourPanel />
        </main>
      </div>
      <TourSpotlight />
    </div>
  );
}

function MobileShell({ height }: { height: number }) {
  const sheet = useAppStore((s) => s.sheet);
  return (
    <div
      className="app app-mobile"
      style={{ "--sheet-h": `${snapHeight(sheet, height)}px` } as CSSProperties}
    >
      <TopBar />
      <ViewTabs />
      <TourInvite />
      <main className="mobile-stage">
        <Stage />
        <GalleryStrip />
      </main>
      <TourPanel />
      <MobileSheet viewportHeight={height} />
      <TourSpotlight />
    </div>
  );
}

export default function App() {
  const { width, height } = useViewport();
  return isNarrow(width) ? <MobileShell height={height} /> : <DesktopShell />;
}
