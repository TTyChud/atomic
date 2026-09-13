/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the physics engine for split deploys (e.g. Vercel → Fly). */
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
