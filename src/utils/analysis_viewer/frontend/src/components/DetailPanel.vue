<template>
  <div class="detail-panel">
    <div class="tabs">
      <button
        v-for="t in tabs"
        :key="t.id"
        :class="{ active: activeTab === t.id }"
        @click="activeTab = t.id"
      >
        {{ t.label }}
      </button>
    </div>

    <div class="tab-body">
      <!-- ============================== Summary ============================== -->
      <section v-if="activeTab === 'summary'" class="summary">
        <template v-if="selection">
          <header class="selection-header">
            <span class="kind-pill" :style="{ background: pillColor }">{{ pillLabel }}</span>
            <h4>{{ selection.title }}</h4>
            <p v-if="selection.subtitle" class="muted small">{{ selection.subtitle }}</p>
          </header>

          <p v-if="selection.message" class="message">{{ selection.message }}</p>

          <table v-if="selection.rows.length" class="kv">
            <tbody>
              <tr v-for="row in selection.rows" :key="row.label">
                <td class="key">{{ row.label }}</td>
                <td class="value">{{ row.value }}</td>
              </tr>
            </tbody>
          </table>

          <div v-if="selection.linkedItem" class="linked">
            <span class="muted small">Linked item:</span>
            <button class="link-btn" @click="selectItem(selection.linkedItem)">
              {{ session.itemById.get(selection.linkedItem)?.label ?? selection.linkedItem }}
            </button>
          </div>
        </template>
        <SessionOverview v-else />
      </section>

      <!-- ============================== Metrics ============================== -->
      <section v-if="activeTab === 'metrics'" class="metrics-tab">
        <table v-if="session.session" class="metrics-table">
          <thead>
            <tr>
              <th></th>
              <th>Metric</th>
              <th>Score</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="m in session.session.metrics"
              :key="m.metric_id"
              :class="{ selected: ui.selectedMetricId === m.metric_id }"
              @click="onMetricRowClick(m.metric_id)"
            >
              <td><span class="swatch" :style="{ background: metricColor(m.metric_id) }" /></td>
              <td>{{ m.label }}</td>
              <td class="score-cell">{{ formatScore(m.score) }}</td>
            </tr>
          </tbody>
        </table>
      </section>

      <!-- ================================ Raw ================================ -->
      <section v-if="activeTab === 'raw'" class="raw-tab">
        <pre>{{ rawJson }}</pre>
      </section>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useSessionStore } from '@/stores/session';
import { useUIStore } from '@/stores/ui';
import { metricColor, SEVERITY_COLOR } from '@/theme/tokens';
import {
  formatFrame,
  formatObjectRows,
  formatScore,
  formatTime,
  type SummaryRow,
} from '@/utils/formatters';
import SessionOverview from './SessionOverview.vue';

const session = useSessionStore();
const ui = useUIStore();

type TabId = 'summary' | 'metrics' | 'raw';
const tabs: { id: TabId; label: string }[] = [
  { id: 'summary', label: 'Summary' },
  { id: 'metrics', label: 'Metrics' },
  { id: 'raw', label: 'Raw' },
];
const activeTab = ref<TabId>('summary');

type SelectionView = {
  title: string;
  subtitle: string | null;
  pillLabel: string;
  pillColor: string;
  message: string | null;
  rows: SummaryRow[];
  linkedItem: string | null;
  raw: unknown;
};

const selection = computed<SelectionView | null>(() => {
  // Priority: flag > item > metric. Only one can be active at a time per the
  // click handlers, but be defensive.
  if (ui.selectedFlagId) {
    const f = session.flagById.get(ui.selectedFlagId);
    if (f) {
      const rows: SummaryRow[] = [
        { label: 'Type', value: f.type },
        { label: 'Severity', value: f.severity },
        { label: 'Frame', value: formatFrame(f.frame) },
        { label: 'Time', value: formatTime(f.time_sec) },
      ];
      if (f.start_frame != null) {
        rows.push({ label: 'Start frame', value: formatFrame(f.start_frame) });
      }
      if (f.end_frame != null) {
        rows.push({ label: 'End frame', value: formatFrame(f.end_frame) });
      }
      return {
        title: f.title,
        subtitle: `Flag · ${f.metric_id}`,
        pillLabel: 'Flag',
        pillColor: SEVERITY_COLOR[f.severity] ?? SEVERITY_COLOR.warning,
        message: f.message,
        rows,
        linkedItem: f.linked_item_id,
        raw: f,
      };
    }
  }

  if (ui.selectedItemId) {
    const item = session.itemById.get(ui.selectedItemId);
    if (item) {
      const rows: SummaryRow[] = [];
      if ('frame' in item) {
        rows.push({ label: 'Frame', value: formatFrame(item.frame) });
        rows.push({ label: 'Time', value: formatTime(item.time_sec) });
      } else {
        rows.push({ label: 'Start frame', value: formatFrame(item.start_frame) });
        rows.push({ label: 'End frame', value: formatFrame(item.end_frame) });
        rows.push({ label: 'Start time', value: formatTime(item.start_time_sec) });
        rows.push({ label: 'End time', value: formatTime(item.end_time_sec) });
      }
      rows.push(
        ...formatObjectRows(item.data as unknown as Record<string, unknown>),
      );
      return {
        title: item.label,
        subtitle: `${item.kind} · ${item.metric_id ?? 'baseline'}`,
        pillLabel: item.kind,
        pillColor: metricColor(item.metric_id),
        message: null,
        rows,
        linkedItem: null,
        raw: item,
      };
    }
  }

  if (ui.selectedMetricId) {
    const metric = session.metricById.get(ui.selectedMetricId);
    if (metric) {
      const rows: SummaryRow[] = [
        { label: 'Score', value: formatScore(metric.score) },
        ...formatObjectRows(metric.summary as unknown as Record<string, unknown>),
      ];
      return {
        title: metric.label,
        subtitle: `Metric · ${metric.metric_id}`,
        pillLabel: 'Metric',
        pillColor: metricColor(metric.metric_id),
        message: null,
        rows,
        linkedItem: null,
        raw: metric,
      };
    }
  }

  return null;
});

const pillLabel = computed(() => selection.value?.pillLabel ?? '');
const pillColor = computed(() => selection.value?.pillColor ?? 'transparent');

const rawJson = computed(() => {
  if (selection.value) return JSON.stringify(selection.value.raw, null, 2);
  if (session.session) return JSON.stringify(session.session, null, 2);
  return '';
});

function onMetricRowClick(metricId: string): void {
  ui.selectedMetricId = ui.selectedMetricId === metricId ? null : metricId;
  ui.selectedItemId = null;
  ui.selectedFlagId = null;
  activeTab.value = 'summary';
}

function selectItem(itemId: string): void {
  ui.selectedItemId = itemId;
  ui.selectedFlagId = null;
  ui.selectedMetricId = null;
  activeTab.value = 'summary';
}

// Whenever the user picks something elsewhere (timeline click, left panel
// click), make sure we're showing the Summary tab so it's actually visible.
watch(
  () => [ui.selectedItemId, ui.selectedFlagId, ui.selectedMetricId] as const,
  ([item, flag, metric]) => {
    if (item || flag || metric) activeTab.value = 'summary';
  },
);

</script>

<style scoped>
.detail-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
}
.tabs {
  display: flex;
  gap: 2px;
  border-bottom: 1px solid var(--color-border);
  padding: 0 4px;
  flex-shrink: 0;
}
.tabs button {
  background: transparent;
  color: var(--color-muted);
  border: none;
  padding: 8px 14px;
  font-size: 0.85em;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
}
.tabs button.active {
  color: var(--color-text);
  border-bottom-color: var(--color-accent);
}
.tabs button:hover:not(.active) {
  color: var(--color-text);
}

.tab-body {
  flex: 1;
  overflow: auto;
  padding: 12px 14px;
}

/* ----- Summary tab ----- */
.summary {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.selection-header h4 {
  margin: 4px 0 0;
  font-size: 1em;
  font-weight: 600;
}
.kind-pill {
  display: inline-block;
  font-size: 0.7em;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding: 2px 8px;
  border-radius: var(--radius-sm);
  font-weight: 700;
  color: var(--color-bg);
}
.message {
  background: var(--color-bg-elev);
  padding: 8px 10px;
  border-radius: 4px;
  margin: 0;
  border-left: 3px solid var(--color-accent);
  font-size: 0.92em;
}
.kv {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88em;
}
.kv td {
  padding: 4px 0;
  border-bottom: 1px dashed var(--color-border);
  vertical-align: top;
}
.kv td.key {
  color: var(--color-muted);
  width: 38%;
  padding-right: 12px;
}
.kv td.value {
  font-variant-numeric: tabular-nums;
  word-break: break-word;
}
.linked {
  display: flex;
  align-items: center;
  gap: 8px;
  padding-top: 4px;
}
.link-btn {
  background: var(--color-bg-elev);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  padding: 3px 10px;
  border-radius: 3px;
  font-size: 0.85em;
  cursor: pointer;
}
.link-btn:hover {
  background: var(--color-border);
}

/* ----- Metrics tab ----- */
.metrics-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9em;
}
.metrics-table thead th {
  text-align: left;
  font-weight: 500;
  font-size: 0.78em;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--color-muted);
  padding: 6px 4px;
  border-bottom: 1px solid var(--color-border);
}
.metrics-table tbody tr {
  cursor: pointer;
}
.metrics-table tbody tr:hover {
  background: var(--color-bg-elev);
}
.metrics-table tbody tr.selected {
  background: var(--color-accent-bg);
}
.metrics-table tbody td {
  padding: 8px 4px;
  border-bottom: 1px solid var(--color-border);
}
.swatch {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 2px;
}
.score-cell {
  font-variant-numeric: tabular-nums;
}
/* ----- Raw tab ----- */
.raw-tab pre {
  background: var(--color-bg-deep);
  border: 1px solid var(--color-border);
  padding: 10px;
  border-radius: var(--radius-md);
  font-size: 0.78em;
  margin: 0;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  overflow: auto;
  white-space: pre;
}

.muted {
  color: var(--color-muted);
}
.small {
  font-size: 0.85em;
}
</style>
