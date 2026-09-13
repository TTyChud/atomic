import { useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import { stateLabel } from "../lib/quantum";
import { nextSnap, PEEK_HEIGHT, snapHeight } from "../lib/sheet";
import { useAppStore } from "../state/store";
import { Controls } from "./Controls";
import { InfoPanel } from "./InfoPanel";

export function MobileSheet({ viewportHeight }: { viewportHeight: number }) {
  const { n, l, m, system, systems, sheet, setSheet } = useAppStore();
  const sys = systems.find((s) => s.key === system);
  const [drag, setDrag] = useState<number | null>(null);
  const track = useRef<{ id: number; y0: number; y: number; t: number; v: number } | null>(
    null,
  );

  const resting = snapHeight(sheet, viewportHeight);
  const height = Math.min(
    Math.max(resting - (drag ?? 0), PEEK_HEIGHT),
    snapHeight("full", viewportHeight),
  );

  const onPointerDown = (e: ReactPointerEvent<HTMLElement>) => {
    track.current = { id: e.pointerId, y0: e.clientY, y: e.clientY, t: e.timeStamp, v: 0 };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: ReactPointerEvent<HTMLElement>) => {
    const t = track.current;
    if (!t || t.id !== e.pointerId) return;
    const dt = e.timeStamp - t.t;
    if (dt > 0) t.v = (e.clientY - t.y) / dt;
    t.y = e.clientY;
    t.t = e.timeStamp;
    setDrag(e.clientY - t.y0);
  };
  const onPointerUp = (e: ReactPointerEvent<HTMLElement>) => {
    const t = track.current;
    if (!t || t.id !== e.pointerId) return;
    track.current = null;
    setDrag(null);
    const moved = Math.abs(e.clientY - t.y0) > 4;
    if (!moved) {
      setSheet(sheet === "collapsed" ? "half" : "collapsed");
      return;
    }
    setSheet(nextSnap(sheet, e.clientY - t.y0, t.v, viewportHeight));
  };

  return (
    <section
      className={`mobile-sheet mobile-sheet-${sheet}${drag === null ? "" : " mobile-sheet-dragging"}`}
      style={{ height }}
      aria-label="controls and readouts"
    >
      <header
        className="mobile-sheet-handle"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      >
        <span className="mobile-sheet-grip" aria-hidden="true" />
        <span className="mobile-sheet-peek">
          {sys ? sys.name : system} · {stateLabel(n, l, m)}
        </span>
        <button
          type="button"
          className="mobile-sheet-toggle"
          aria-expanded={sheet !== "collapsed"}
          onClick={() => setSheet(sheet === "collapsed" ? "half" : "collapsed")}
        >
          {sheet === "collapsed" ? "controls" : "hide"}
        </button>
      </header>
      <div className="mobile-sheet-body" inert={sheet === "collapsed"}>
        <Controls />
        <InfoPanel />
      </div>
    </section>
  );
}
