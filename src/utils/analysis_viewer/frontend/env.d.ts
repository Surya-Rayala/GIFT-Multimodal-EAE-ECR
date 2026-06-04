/// <reference types="vite/client" />

declare module '*.vue' {
  import type { DefineComponent } from 'vue';
  const component: DefineComponent<{}, {}, any>;
  export default component;
}

interface ImportMetaEnv {
  readonly VITE_SIDECAR_PORT?: string;
  // Set by the Python launcher (src/utils/analysis_viewer/__main__.py) when a
  // session path is supplied on the CLI; the frontend auto-loads it on mount.
  readonly VITE_INITIAL_SESSION?: string;
}

interface Window {
  // Injected by the Tauri shell at runtime once the Python sidecar reports its bound port.
  __SIDECAR_PORT__?: number;
}
