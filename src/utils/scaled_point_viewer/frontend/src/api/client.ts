// API client for the Python sidecar.
//
// Port resolution priority:
//   1. Tauri runtime  -> invoke('get_sidecar_port')
//   2. window.__SIDECAR_PORT__ (Tauri eval injection)
//   3. import.meta.env.VITE_SIDECAR_PORT
//   4. 8765 default (matches manual sidecar smoke tests)

import type { SaveRequest, SaveResponse } from '@/types/models';

let cachedPort: number | null = null;

function isTauri(): boolean {
  return (
    typeof window !== 'undefined' &&
    ('__TAURI_INTERNALS__' in window || '__TAURI__' in window)
  );
}

export async function initSidecarPort(): Promise<number> {
  if (cachedPort != null) return cachedPort;

  if (isTauri()) {
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      const port = await invoke<number | null>('get_sidecar_port');
      if (typeof port === 'number' && port > 0) {
        cachedPort = port;
        return port;
      }
    } catch {
      // fall through
    }
  }

  if (typeof window !== 'undefined' && typeof window.__SIDECAR_PORT__ === 'number') {
    cachedPort = window.__SIDECAR_PORT__;
    return cachedPort;
  }

  const env = import.meta.env.VITE_SIDECAR_PORT;
  cachedPort = env ? Number(env) : 8765;
  return cachedPort;
}

export function sidecarBaseUrl(): string {
  const port = cachedPort ?? 8765;
  return `http://127.0.0.1:${port}`;
}

async function postJson<T>(path: string, payload: unknown): Promise<T> {
  const url = `${sidecarBaseUrl()}${path}`;
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText} — ${body || url}`);
  }
  return (await res.json()) as T;
}

export function imageUrl(absPath: string): string {
  return `${sidecarBaseUrl()}/image?path=${encodeURIComponent(absPath)}`;
}

export function saveScaledPoints(req: SaveRequest): Promise<SaveResponse> {
  return postJson<SaveResponse>('/save', req);
}
