import { useEffect, useState } from "react";
import { thumbnailUrl } from "../api/client";
import type { Basis } from "../api/client";
import { galleryStates } from "../lib/gallery";
import { subshellAvailable } from "../lib/hfModel";
import { THUMBNAIL_LIBERTY } from "../lib/liberties";
import { stateLabel } from "../lib/quantum";
import { systemKind } from "../lib/systemKind";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

function Thumb({ src, label }: { src: string; label: string }) {
  const [broken, setBroken] = useState(false);
  useEffect(() => setBroken(false), [src]);
  if (broken) return <span className="thumb-blank" aria-hidden="true" />;
  return (
    <img
      src={src}
      alt={label}
      width={56}
      height={56}
      loading="lazy"
      onError={() => setBroken(true)}
    />
  );
}

export function GalleryStrip() {
  const {
    n, l, m, system, systems, basis, model, config, exchange, pauli,
    setQuantumNumbers, hfLevels,
  } = useAppStore();
  const hasThumbnails = systemKind(systems, system) !== null;
  return (
    <div className="gallery">
      <div className="gallery-head">
        <span>n = {n} states</span>
        {hasThumbnails && <Badge provenance={THUMBNAIL_LIBERTY} />}
      </div>
      <div className="gallery-scroll">
        {galleryStates(n).map((s) => {
          const active = s.l === l && s.m === m;
          const reachable = subshellAvailable(hfLevels, model, s.n, s.l);
          return (
            <button
              key={`${s.l},${s.m}`}
              type="button"
              className={active ? "thumb thumb-active" : "thumb"}
              disabled={!reachable}
              title={
                reachable
                  ? stateLabel(s.n, s.l, s.m)
                  : `${stateLabel(s.n, s.l, s.m)}: empty in this configuration, ` +
                    `so the Hartree-Fock solve has no orbital for it`
              }
              onClick={() => setQuantumNumbers(s.n, s.l, s.m)}
            >
              {hasThumbnails && reachable ? (
                <Thumb
                  src={thumbnailUrl(s.n, s.l, s.m, system, basis as Basis, 96, {
                    model, config, exchange, pauli,
                  })}
                  label={stateLabel(s.n, s.l, s.m)}
                />
              ) : (
                <span className="thumb-blank" aria-hidden="true" />
              )}
              <span>{stateLabel(s.n, s.l, s.m)}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
