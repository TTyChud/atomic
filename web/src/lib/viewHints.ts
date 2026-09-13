import type { ViewMode } from "../lib/urlState";

export interface ViewHint {

  what: string;
}

export const VIEW_HINTS: Record<ViewMode, ViewHint> = {
  cloud: {
    what: "the electron's probable positions in 3-D, point by point",
  },
  plane: {
    what: "a slice through the atom, like a textbook figure but computed",
  },
  radial: {
    what: "how far the electron sits from the nucleus, on average and in detail",
  },
  levels: {
    what: "every energy the atom is allowed to have, as a ladder",
  },
  spectrum: {
    what: "the light the atom emits, checked against measured wavelengths",
  },
  whatif: {
    what: "change the constants of nature and watch what refuses to move",
  },
  forcelaw: {
    what: "replace the Coulomb pull with another force and re-solve the atom",
  },
};
