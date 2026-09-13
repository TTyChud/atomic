import type { ViewMode } from "../lib/urlState";

export interface ViewHint {
  /** One line under the tab name: what am I looking at? */
  what: string;
}

/**
 * Per-view one-liners shown as tooltips (and on the mobile tabs' title
 * attribute). They answer the first question a newcomer has — "what is this
 * tab?" — without opening the tour or the intro panel.
 */
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
