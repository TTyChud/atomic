import { useEffect, useState } from "react";
import { thumbnailSrc } from "../api/thumbnail";
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

function GalleryThumb(props: {
  n: number;
  l: number;
  m: number;
  system: string;
  basis: Basis;
  size: number;
  model: "gsz" | "hf";
  config: string | null;
  exchange: boolean;
  pauli: boolean;
  label: string;
}) {
  const [src, setSrc] = useState<string | null>(null);
  const key = `${props.n}/${props.l}/${props.m}/${props.system}/${props.basis}/${props.model}/${props.config}/${props.exchange}/${props.pauli}`;
  useEffect(() => {
    let alive = true;
    setSrc(null);
    thumbnailSrc(
      props.n, props.l, props.m, props.system, props.basis, props.size,
      { model: props.model, config: props.config, exchange: props.exchange, pauli: props.pauli },
    )
      .then((s) => {
        if (alive) setSrc(s);
      })
      .catch(() => {
        if (alive) setSrc("");
      });
    return () => {
      alive = false;
    };
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps
  if (src === null || src === "") return <span className="thumb-blank" aria-hidden="true" />;
  return <Thumb src={src} label={props.label} />;
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
                <GalleryThumb
                  n={s.n} l={s.l} m={s.m}
                  system={system} basis={basis as Basis} size={96}
                  model={model} config={config} exchange={exchange} pauli={pauli}
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
