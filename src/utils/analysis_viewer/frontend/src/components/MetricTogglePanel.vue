<template>
  <ul class="metric-toggle-panel">
    <li
      v-for="m in metrics"
      :key="m.metric_id"
      class="row"
      :class="{ selected: ui.selectedMetricId === m.metric_id }"
      @click="onRowClick(m.metric_id)"
    >
      <button
        class="toggle"
        type="button"
        :class="{ on: ui.isMetricVisible(m.metric_id) }"
        :aria-pressed="ui.isMetricVisible(m.metric_id)"
        :aria-label="`Toggle ${m.label} on timeline`"
        :title="ui.isMetricVisible(m.metric_id) ? 'Hide on timeline' : 'Show on timeline'"
        @click.stop="
          ui.setMetricVisibility(m.metric_id, !ui.isMetricVisible(m.metric_id))
        "
      >
        <span class="dot" />
      </button>
      <span class="swatch" :style="{ background: metricColor(m.metric_id) }" />
      <span class="label">{{ m.label }}</span>
      <span class="score">
        {{ formatScore(m.score) }}
      </span>
    </li>
  </ul>
</template>

<script setup lang="ts">
import { useSessionStore } from '@/stores/session';
import { useUIStore } from '@/stores/ui';
import { metricColor } from '@/theme/tokens';
import { formatScore } from '@/utils/formatters';
import { computed } from 'vue';

const session = useSessionStore();
const ui = useUIStore();

const metrics = computed(() => session.session?.metrics ?? []);

function onRowClick(metricId: string): void {
  ui.selectedMetricId = ui.selectedMetricId === metricId ? null : metricId;
  ui.selectedItemId = null;
  ui.selectedFlagId = null;
}
</script>

<style scoped>
.metric-toggle-panel {
  list-style: none;
  padding: 0;
  margin: 0;
}
.row {
  display: grid;
  grid-template-columns: auto auto 1fr auto;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  margin: 0 -8px;
  border-radius: 4px;
  cursor: pointer;
  user-select: none;
}
.row:hover {
  background: var(--color-border);
}
.row.selected {
  background: var(--color-accent-bg);
  outline: 1px solid rgba(59, 130, 246, 0.6);
}
.toggle {
  position: relative;
  width: 30px;
  height: 16px;
  padding: 0;
  margin: 0;
  border: 1px solid var(--color-border);
  border-radius: 999px;
  background: var(--color-bg-elev);
  cursor: pointer;
  flex-shrink: 0;
  transition: background-color 0.12s ease, border-color 0.12s ease;
}
.toggle .dot {
  position: absolute;
  top: 1px;
  left: 1px;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: var(--color-muted);
  transition: transform 0.12s ease, background-color 0.12s ease;
}
.toggle:hover {
  border-color: var(--color-text-secondary, var(--color-muted));
}
.toggle.on {
  background: var(--color-accent);
  border-color: var(--color-accent);
}
.toggle.on .dot {
  background: var(--color-bg);
  transform: translateX(14px);
}
.toggle:focus-visible {
  outline: 2px solid var(--color-accent);
  outline-offset: 1px;
}
.swatch {
  width: 10px;
  height: 10px;
  border-radius: 2px;
  display: inline-block;
}
.label {
  flex: 1;
  font-size: 0.92em;
}
.score {
  font-variant-numeric: tabular-nums;
  padding: 2px 7px;
  border-radius: 3px;
  background: var(--color-border);
  font-size: 0.78em;
  font-weight: 600;
}
</style>
