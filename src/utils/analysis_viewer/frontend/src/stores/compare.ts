import { defineStore } from 'pinia';
import { ref } from 'vue';

import { fetchComparison, fetchRuns } from '@/api/client';
import type { CompareResult, RunSummary } from '@/types/models';

import { useSessionStore } from './session';

export const useCompareStore = defineStore('compare', () => {
  const outputsRoot = ref<string | null>(null);
  const currentRunPath = ref<string | null>(null);
  const currentRunId = ref<string | null>(null);
  const runs = ref<RunSummary[]>([]);

  const selectedOtherRunPath = ref<string | null>(null);
  const selectedMetricId = ref<string | null>(null);

  const result = ref<CompareResult | null>(null);
  const resultKey = ref<string | null>(null);
  const loading = ref(false);
  const error = ref<string | null>(null);

  // Layout of multi-chart results: 'stack' (default — one below the other,
  // each at full pane width) or 'row' (side-by-side, equal columns).
  // Persisted across metric switches in the same session so the user only
  // has to choose once.
  const vizLayout = ref<'stack' | 'row'>('stack');
  function setVizLayout(v: 'stack' | 'row'): void {
    vizLayout.value = v;
  }

  function _key(): string {
    return `${currentRunPath.value ?? ''}|${selectedOtherRunPath.value ?? ''}|${selectedMetricId.value ?? ''}`;
  }

  function initFromEnv(): void {
    const session = useSessionStore();

    // env vars (set when launched via `python -m src.utils.analysis_viewer
    // <run_folder>`) win over derived state, but only on first call.
    if (!outputsRoot.value) {
      const envRoot = import.meta.env.VITE_OUTPUTS_ROOT as string | undefined;
      if (envRoot) outputsRoot.value = envRoot;
    }
    if (!currentRunPath.value) {
      const envRun = import.meta.env.VITE_INITIAL_RUN_DIR as string | undefined;
      if (envRun) currentRunPath.value = envRun;
    }

    // Re-derive from the currently-loaded session every time. This handles
    // the case where the user launched the viewer with no preset path and
    // then opened a run folder via the in-app dialog.
    const info = session.session?.run_info;
    if (info?.run_id) currentRunId.value = info.run_id;
    const runDir = session.session?.run_dir;
    if (runDir) {
      currentRunPath.value = runDir;
      if (!outputsRoot.value) {
        // Run folder is a subdir of the outputs tree; parent dir is the root.
        const trimmed = runDir.replace(/[\\/]+$/, '');
        const lastSep = Math.max(trimmed.lastIndexOf('/'), trimmed.lastIndexOf('\\'));
        if (lastSep > 0) outputsRoot.value = trimmed.slice(0, lastSep);
      }
    }
  }

  async function refreshRuns(): Promise<void> {
    if (!outputsRoot.value) {
      runs.value = [];
      return;
    }
    try {
      loading.value = true;
      runs.value = await fetchRuns(outputsRoot.value);
    } catch (e) {
      error.value = (e as Error).message;
      runs.value = [];
    } finally {
      loading.value = false;
    }
  }

  async function setOutputsRoot(path: string): Promise<void> {
    outputsRoot.value = path || null;
    selectedOtherRunPath.value = null;
    result.value = null;
    resultKey.value = null;
    await refreshRuns();
  }

  function selectOther(path: string | null): void {
    selectedOtherRunPath.value = path;
    result.value = null;
    resultKey.value = null;
  }

  function selectMetric(metricId: string | null): void {
    // Re-clicking the same metric is a no-op. Vue's ref equality silently
    // suppresses watchers when the value is unchanged, so unconditionally
    // wiping `result` here would blank the rendered comparison without
    // ever triggering a re-fetch.
    if (metricId === selectedMetricId.value) return;
    selectedMetricId.value = metricId;
    result.value = null;
    resultKey.value = null;
  }

  async function fetchComparisonResult(): Promise<void> {
    const cur = currentRunPath.value;
    const other = selectedOtherRunPath.value;
    const metric = selectedMetricId.value;
    if (!cur || !other || !metric) return;
    const session = useSessionStore();
    const sessionPath = session.session?.session_json_path;
    if (!sessionPath) {
      error.value = 'No active session — open a run first.';
      return;
    }
    const key = _key();
    if (resultKey.value === key && result.value) return;
    try {
      loading.value = true;
      error.value = null;
      const payload = await fetchComparison(cur, other, metric, sessionPath);
      result.value = payload;
      resultKey.value = key;
    } catch (e) {
      error.value = (e as Error).message;
      result.value = null;
    } finally {
      loading.value = false;
    }
  }

  return {
    outputsRoot,
    currentRunPath,
    currentRunId,
    runs,
    selectedOtherRunPath,
    selectedMetricId,
    result,
    loading,
    error,
    vizLayout,
    initFromEnv,
    refreshRuns,
    setOutputsRoot,
    selectOther,
    selectMetric,
    fetchComparisonResult,
    setVizLayout,
  };
});
