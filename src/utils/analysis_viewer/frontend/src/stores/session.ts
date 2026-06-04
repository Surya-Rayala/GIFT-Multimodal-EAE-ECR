import { defineStore } from 'pinia';
import { computed, ref } from 'vue';
import type {
  AnalysisSession,
  FlagRecord,
  MetricRecord,
  TimelineItem,
} from '@/types/models';
import { fetchSession } from '@/api/client';

export const useSessionStore = defineStore('session', () => {
  const session = ref<AnalysisSession | null>(null);
  const loading = ref(false);
  const error = ref<string | null>(null);

  const itemById = computed(() => {
    const map = new Map<string, TimelineItem>();
    if (!session.value) return map;
    for (const item of session.value.timeline.items) {
      map.set(item.item_id, item);
    }
    return map;
  });

  const metricById = computed(() => {
    const map = new Map<string, MetricRecord>();
    if (!session.value) return map;
    for (const m of session.value.metrics) {
      map.set(m.metric_id, m);
    }
    return map;
  });

  const flagById = computed(() => {
    const map = new Map<string, FlagRecord>();
    if (!session.value) return map;
    for (const f of session.value.flags) {
      map.set(f.flag_id, f);
    }
    return map;
  });

  async function load(jsonPath: string): Promise<void> {
    loading.value = true;
    error.value = null;
    try {
      session.value = await fetchSession(jsonPath);
    } catch (e: unknown) {
      session.value = null;
      error.value = e instanceof Error ? e.message : String(e);
    } finally {
      loading.value = false;
    }
  }

  function clear(): void {
    session.value = null;
    error.value = null;
  }

  return {
    session,
    loading,
    error,
    itemById,
    metricById,
    flagById,
    load,
    clear,
  };
});
