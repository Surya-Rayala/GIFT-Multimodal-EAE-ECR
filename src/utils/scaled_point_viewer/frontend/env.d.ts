/// <reference types="vite/client" />

declare global {
  interface Window {
    __SIDECAR_PORT__?: number;
    __TAURI__?: unknown;
    __TAURI_INTERNALS__?: unknown;
  }
}

interface ImportMetaEnv {
  readonly VITE_SIDECAR_PORT?: string;
  readonly VITE_INITIAL_PROJECT_DIR?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

export {};
