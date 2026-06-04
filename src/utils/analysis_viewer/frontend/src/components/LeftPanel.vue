<template>
  <div class="left-panel">
    <header>
      <div class="title-row">
        <h3>Analysis Viewer</h3>
        <button
          v-if="canOpenDialog"
          class="open-btn"
          :disabled="opening"
          @click="onOpenClick"
        >
          {{ opening ? 'Opening…' : 'Open…' }}
        </button>
      </div>
      <p v-if="session.session" class="muted small">
        {{ session.session.session.video_basename }}
      </p>
      <!-- Browser-only fallback: no native file dialog, so accept a typed path. -->
      <form
        v-if="!canOpenDialog"
        class="browser-load"
        @submit.prevent="onBrowserLoad"
      >
        <input
          v-model="browserPath"
          type="text"
          placeholder="path/to/run_folder"
          class="path-input"
          spellcheck="false"
        />
        <button type="submit" class="open-btn">Load</button>
      </form>
    </header>

    <section v-if="session.loading">
      <p class="muted">Loading session…</p>
    </section>

    <section v-else-if="session.error" class="error">
      <strong>Failed to load session</strong>
      <p>{{ session.error }}</p>
    </section>

    <template v-else-if="session.session">
      <CompareSetupPanel v-if="ui.viewMode === 'compare'" />
      <template v-else>
        <section>
          <h4>Metrics</h4>
          <MetricTogglePanel />
        </section>

        <section>
          <h4>Flags</h4>
          <ul v-if="session.session.flags.length" class="flag-list">
            <li
              v-for="f in session.session.flags"
              :key="f.flag_id"
              class="flag-row"
              :class="{ selected: ui.selectedFlagId === f.flag_id }"
              @click="onFlagClick(f.flag_id, f.frame)"
            >
              <span class="severity" :class="f.severity">{{ f.severity }}</span>
              <span class="title">{{ f.title }}</span>
            </li>
          </ul>
          <p v-else class="muted small">No flags raised.</p>
        </section>
      </template>
    </template>

    <section v-else>
      <p class="muted">
        No run loaded. Click <strong>Open…</strong> and pick a run folder, or
        launch with
        <code>python -m src.utils.analysis_viewer &lt;run_folder&gt;</code>.
      </p>
    </section>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref, watch } from 'vue';
import { useSessionStore } from '@/stores/session';
import { usePlaybackStore } from '@/stores/playback';
import { useUIStore } from '@/stores/ui';
import CompareSetupPanel from './CompareSetupPanel.vue';
import MetricTogglePanel from './MetricTogglePanel.vue';

const session = useSessionStore();
const playback = usePlaybackStore();
const ui = useUIStore();

// The native file picker requires the Tauri runtime. In the browser the
// dialog button is hidden and the typed-path form below takes over instead.
const canOpenDialog =
  typeof window !== 'undefined' &&
  ('__TAURI_INTERNALS__' in window || '__TAURI__' in window);

const opening = ref(false);

// Browser-only path input. Auto-fills with the URL's ?session= or with the
// last successfully-loaded path so the next reload only needs one click.
const RECENT_KEY = 'Vanderbilt-GIFT.AnalysisViewer.recentSession';
const browserPath = ref<string>('');

onMounted(() => {
  const params = new URLSearchParams(window.location.search);
  const fromUrl = params.get('session');
  const fromStorage = (() => {
    try {
      return localStorage.getItem(RECENT_KEY) ?? '';
    } catch {
      return '';
    }
  })();
  browserPath.value = fromUrl || fromStorage;
});

watch(
  () => session.session?.session_json_path,
  (path) => {
    if (!path) return;
    try {
      localStorage.setItem(RECENT_KEY, path);
    } catch {
      // localStorage may be disabled — silent fail
    }
  },
);

async function onBrowserLoad(): Promise<void> {
  const path = browserPath.value.trim();
  if (!path) return;
  await session.load(path);
}

async function onOpenClick(): Promise<void> {
  if (opening.value) return;
  opening.value = true;
  try {
    const { open } = await import('@tauri-apps/plugin-dialog');
    const selection = await open({
      title: 'Open Run Folder',
      multiple: false,
      directory: true,
    });
    if (typeof selection === 'string' && selection) {
      await session.load(selection);
    }
  } catch (err) {
    console.error('[analysis-viewer] folder dialog failed:', err);
  } finally {
    opening.value = false;
  }
}

function onFlagClick(flagId: string, frame: number | null): void {
  ui.selectedFlagId = flagId;
  ui.selectedItemId = null;
  ui.selectedMetricId = null;
  if (frame != null) playback.requestSeek(frame);
}
</script>

<style scoped>
.left-panel {
  padding: 14px 14px 24px;
  height: 100%;
  overflow: auto;
}
header {
  margin-bottom: 12px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--color-border);
}
h3 {
  margin: 0;
  font-size: 1.05em;
  font-weight: 600;
}
.title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}
.open-btn {
  background: var(--color-bg-elev);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  padding: 4px 10px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.82em;
}
.open-btn:hover:not(:disabled) {
  background: var(--color-border);
}
.open-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
.browser-load {
  display: flex;
  gap: 6px;
  margin-top: 8px;
  align-items: stretch;
}
.path-input {
  flex: 1;
  background: var(--color-bg);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  padding: 4px 8px;
  font-size: 0.82em;
  min-width: 0;
}
.path-input:focus {
  outline: none;
  border-color: var(--color-accent);
}
h4 {
  margin: 16px 0 8px;
  font-size: 0.78em;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--color-muted);
}
.muted {
  color: var(--color-muted);
}
.small {
  font-size: 0.85em;
}
.error {
  background: var(--color-danger-bg);
  border: 1px solid var(--color-border-strong);
  border-radius: var(--radius-md);
  padding: var(--space-sm) 10px;
  color: var(--color-danger-text);
}
.error strong {
  display: block;
  margin-bottom: 4px;
}
.error p {
  margin: 0;
  font-size: 0.85em;
  word-break: break-word;
}
ul {
  list-style: none;
  padding: 0;
  margin: 0;
}
.flag-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 6px;
  margin: 0 -6px;
  font-size: 0.88em;
  border-radius: 4px;
  cursor: pointer;
}
.flag-row:hover {
  background: var(--color-border);
}
.flag-row.selected {
  background: var(--color-accent-bg);
  outline: 1px solid rgba(59, 130, 246, 0.6);
}
.flag-row .severity {
  font-size: 0.72em;
  text-transform: uppercase;
  padding: 1px 6px;
  border-radius: 3px;
  font-weight: 600;
  letter-spacing: 0.04em;
}
.flag-row .severity.warning {
  background: var(--color-warning-bg);
  color: var(--color-warning);
}
.flag-row .title {
  flex: 1;
  word-break: break-word;
}
code {
  background: var(--color-border);
  padding: 1px 5px;
  border-radius: 3px;
  font-size: 0.85em;
}
</style>
