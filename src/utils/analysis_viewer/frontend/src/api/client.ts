// Typed fetch helpers for the FastAPI sidecar.
//
// Port resolution priority:
//   1. Tauri runtime  → invoke('get_sidecar_port') (the Rust shell knows the
//      port it bound the Python sidecar to)
//   2. window.__SIDECAR_PORT__ (best-effort eval injection from Tauri)
//   3. import.meta.env.VITE_SIDECAR_PORT (Vite dev override)
//   4. 8765 default (matches the port used by the manual sidecar smoke tests)

import type { AnalysisSession, CompareResult, RunSummary } from '@/types/models';

let cachedPort: number | null = null;

function isTauri(): boolean {
  // Tauri 2 exposes both `__TAURI_INTERNALS__` and `__TAURI__`. Either is fine.
  return (
    typeof window !== 'undefined' &&
    ('__TAURI_INTERNALS__' in window || '__TAURI__' in window)
  );
}

/**
 * Resolves the sidecar port asynchronously and caches it. Call once at app
 * startup before any other API helper, since the rest are sync.
 */
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
      // fall through to other resolvers
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
  // Once `initSidecarPort` has resolved, this is sync. Before then it falls
  // back to 8765, which is the dev-default and still likely to work in the
  // common case where the user runs the sidecar manually.
  const port = cachedPort ?? 8765;
  return `http://127.0.0.1:${port}`;
}

async function getJson<T>(path: string): Promise<T> {
  const url = `${sidecarBaseUrl()}${path}`;
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText} — ${body || url}`);
  }
  return (await res.json()) as T;
}

export function fetchSession(jsonPath: string): Promise<AnalysisSession> {
  return getJson<AnalysisSession>(`/session?path=${encodeURIComponent(jsonPath)}`);
}

export function fetchHealth(): Promise<{ ok: boolean; service: string }> {
  return getJson(`/health`);
}

export function videoUrl(videoPath: string, sessionPath: string): string {
  const params = new URLSearchParams({ path: videoPath, session: sessionPath });
  return `${sidecarBaseUrl()}/video?${params.toString()}`;
}

export function imageUrl(imagePath: string, sessionPath: string): string {
  const params = new URLSearchParams({ path: imagePath, session: sessionPath });
  return `${sidecarBaseUrl()}/image?${params.toString()}`;
}

export function fetchRuns(outputsRoot: string): Promise<RunSummary[]> {
  return getJson<RunSummary[]>(`/runs?outputs_root=${encodeURIComponent(outputsRoot)}`);
}

export function fetchComparison(
  currentRun: string,
  otherRun: string,
  metricId: string,
  sessionPath: string,
): Promise<CompareResult> {
  const params = new URLSearchParams({
    current_run: currentRun,
    other_run: otherRun,
    metric_id: metricId,
    session: sessionPath,
  });
  return getJson<CompareResult>(`/compare?${params.toString()}`);
}
