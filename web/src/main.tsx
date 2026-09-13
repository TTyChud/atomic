import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { COUNT_CHOICES } from "./components/Controls";
import { currentUrlState, isNewPlace, parseAppUrl, serializeAppUrl } from "./lib/urlState";
import { isNarrow } from "./lib/viewport";
import { isScreened } from "./lib/systemKind";
import { useAppStore } from "./state/store";
import "@fontsource-variable/space-grotesk";
import "@fontsource-variable/jetbrains-mono";
import "./index.css";

if (isNarrow(window.innerWidth)) useAppStore.setState({ count: COUNT_CHOICES[0] });

const opening = parseAppUrl(window.location.search);
const { ghost: ghostOn, ...rest } = opening;
useAppStore.setState(rest);
if (ghostOn) useAppStore.setState({ ghostOn: true });

if (opening.tour) {
  const id = opening.tour;
  const step = opening.step ?? 0;
  queueMicrotask(() => useAppStore.getState().startTour(id, step));
}

{
  const settled = useAppStore.getState();
  void settled.loadSystems().then(() => {
    const s = useAppStore.getState();
    if (!isScreened(s.systems, s.system)) void s.loadStateInfo();
  });
  if (settled.view === "cloud") void settled.sample();
  else if (settled.view === "plane") void settled.loadPlane();
  else if (settled.view === "radial") void settled.loadRadial();
  else if (settled.view === "whatif") {
    void settled.loadWhatIf();
    void settled.loadGhost();
  } else if (settled.view === "forcelaw") void settled.loadForceLaw();
  else void settled.loadLevels();
}

let lastUrl = {
  ...currentUrlState(useAppStore.getState()),
  tour: opening.tour ?? null,
  step: opening.step ?? 0,
  ghost: opening.ghost ?? false,
};

useAppStore.subscribe((s) => {
  const now = { ...currentUrlState(s), tour: s.tourId, step: s.stepIndex, ghost: s.ghostOn };
  const next = window.location.pathname + serializeAppUrl(now);
  if (next !== window.location.pathname + window.location.search) {
    if (isNewPlace(lastUrl, now)) window.history.pushState(null, "", next);
    else window.history.replaceState(null, "", next);
  }
  lastUrl = now;
});

window.addEventListener("popstate", () => {
  useAppStore.getState().applyUrl(parseAppUrl(window.location.search));
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
