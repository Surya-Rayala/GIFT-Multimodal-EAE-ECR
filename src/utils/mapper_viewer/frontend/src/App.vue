<template>
  <div class="app">
    <header class="topbar">
      <h1>Mapper</h1>
      <p class="muted">{{ subtitle }}</p>
    </header>

    <!-- Completion view replaces the entire wizard once Finish is hit. -->
    <section v-if="ui.showComplete" class="complete">
      <div class="complete-card">
        <div class="check">✓</div>
        <h2>All done</h2>
        <p class="muted">
          Files for project <strong>{{ project.projectName || 'project' }}</strong> saved to:
        </p>
        <p class="path-line">{{ project.outputDir || '(no output folder set)' }}</p>

        <ul class="saved-list">
          <li v-for="row in savedRows" :key="row.label" :class="{ unsaved: !row.saved }">
            <span class="saved-icon" aria-hidden="true">{{ row.saved ? '✓' : '–' }}</span>
            <span class="saved-label">{{ row.label }}</span>
            <span class="saved-file" :title="row.filename">{{ row.saved ? row.filename : 'not saved' }}</span>
          </li>
        </ul>

        <p class="muted small">
          What would you like to do?
        </p>
        <div class="complete-actions">
          <button class="secondary" @click="onStartOver">Start a new project</button>
          <button class="primary" @click="closeWindow">Close mapper</button>
        </div>
      </div>
    </section>

    <!-- Normal wizard chrome — hidden once the completion screen is up. -->
    <template v-else>
    <StepBar />

    <main class="body">
      <SetupStep        v-if="ui.currentStep === 'setup'" />
      <AlignStep        v-else-if="ui.currentStep === 'align'" />
      <EntryZonesStep   v-else-if="ui.currentStep === 'entries'" />
      <PodTargetsStep   v-else-if="ui.currentStep === 'pods'" />
      <RoomOutlineStep  v-else-if="ui.currentStep === 'boundary'" />
    </main>

    <footer class="footer">
      <button
        class="secondary"
        @click="onBack"
        :disabled="!canGoBack"
      >
        ← Back
      </button>

      <p v-if="saveError" class="status error">{{ saveError }}</p>
      <p v-else-if="saveOk" class="status ok">{{ saveOk }}</p>
      <p v-else class="status muted">
        {{ contextualFooterText }}
      </p>

      <button
        v-if="ui.currentStep !== 'setup'"
        class="secondary"
        @click="onSave"
        :disabled="saving || !canSave"
        :title="canSave ? '' : saveBlockedReason"
      >
        {{ saving ? 'Saving…' : 'Save this step' }}
      </button>
      <button
        v-if="ui.currentStep !== 'boundary'"
        class="primary"
        @click="onNext"
        :disabled="!canGoNext"
        :title="canGoNext ? '' : nextBlockedReason"
      >
        Next →
      </button>
      <button
        v-else
        class="primary"
        @click="onFinish"
        :disabled="!ui.saved.boundary"
        :title="ui.saved.boundary ? 'All steps complete' : 'Save the room boundary first.'"
      >
        Finish
      </button>
    </footer>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';

import StepBar from './components/StepBar.vue';
import SetupStep from './components/SetupStep.vue';
import AlignStep from './components/AlignStep.vue';
import EntryZonesStep from './components/EntryZonesStep.vue';
import PodTargetsStep from './components/PodTargetsStep.vue';
import RoomOutlineStep from './components/RoomOutlineStep.vue';

import { saveArtifact } from '@/api/client';
import { useUIStore } from '@/stores/ui';
import { useProjectStore } from '@/stores/project';
import { useMapperStore } from '@/stores/mapper';
import { STEPS, type StepKey } from '@/types/models';

const ui = useUIStore();
const project = useProjectStore();
const mapper = useMapperStore();

const saving = ref(false);
const saveOk = ref('');
const saveError = ref('');

const subtitle = computed(() => {
  const s = STEPS.find((s) => s.key === ui.currentStep);
  return s ? s.subtitle : '';
});

const stepIndex = computed(() => STEPS.findIndex((s) => s.key === ui.currentStep));
const canGoBack = computed(() => stepIndex.value > 0);
const canGoNext = computed(() => stepIndex.value < STEPS.length - 1 && nextBlockedReason.value === '');
const nextBlockedReason = computed(() => {
  if (ui.currentStep === 'setup') {
    if (!project.mapImagePath) return 'Pick a map image first.';
    if (!project.cameraVideoPath && !project.cameraFrameImagePath) return 'Pick a camera reference first.';
    if (!project.outputDir) return 'Pick a save folder first.';
    if (!project.projectName.trim()) return 'Project name cannot be empty.';
  }
  // Steps 2-5 don't block Next — saving is optional per step until Finish.
  return '';
});

const canSave = computed(() => saveBlockedReason.value === '');
const saveBlockedReason = computed(() => {
  if (!project.outputDir) return 'Pick a save folder in Setup first.';
  if (!project.projectName.trim()) return 'Set a project name in Setup first.';
  switch (ui.currentStep) {
    case 'align':
      if (!mapper.pairs.length) return 'Add at least one mapping pair.';
      if (mapper.pairs.some((p) => p.mx == null || p.my == null)) return 'Finish the last pair before saving.';
      return '';
    case 'entries':
      if (!mapper.entryPolygons.length) return 'Confirm at least one polygon.';
      return '';
    case 'pods':
      if (!mapper.podPoints.length) return 'Drop at least one POD.';
      return '';
    case 'boundary':
      if (!mapper.boundaryConfirmed) return 'Confirm the boundary before saving.';
      return '';
    default:
      return '';
  }
});

const contextualFooterText = computed(() => {
  switch (ui.currentStep) {
    case 'setup': return 'Fill in the fields, then press Next.';
    case 'align': return mapper.pairs.length
      ? `${mapper.pairs.length} pair${mapper.pairs.length === 1 ? '' : 's'}. Aim for 4+ well-spread pairs, then Save.`
      : 'Alternate camera ↔ map clicks. Save when you have 4+ pairs.';
    case 'entries': return mapper.entryPolygons.length
      ? `${mapper.entryPolygons.length} polygon${mapper.entryPolygons.length === 1 ? '' : 's'} ready. Save to write the file.`
      : 'Click on the map to drop vertices, then Confirm.';
    case 'pods': return mapper.podPoints.length
      ? `${mapper.podPoints.length} POD${mapper.podPoints.length === 1 ? '' : 's'}. Save to write the file.`
      : 'Click on the map to drop each POD pin.';
    case 'boundary': return mapper.boundaryConfirmed
      ? 'Boundary confirmed. Save to write the file.'
      : 'Click vertices, then Confirm. Need at least 4 points.';
  }
  return '';
});

function onBack(): void {
  if (stepIndex.value <= 0) return;
  ui.goTo(STEPS[stepIndex.value - 1].key as StepKey);
  saveOk.value = ''; saveError.value = '';
}
function onNext(): void {
  if (!canGoNext.value) return;
  ui.goTo(STEPS[stepIndex.value + 1].key as StepKey);
  saveOk.value = ''; saveError.value = '';
}

function onFinish(): void {
  // Warn about any data-producing step that was never saved. The room
  // boundary is implicit (Finish is gated on ui.saved.boundary), so the
  // remaining ones to check are align / entries / pods.
  const unsavedLabels: string[] = [];
  if (!ui.saved.align) unsavedLabels.push('Align Camera to Map');
  if (!ui.saved.entries) unsavedLabels.push('Mark Entry Zones');
  if (!ui.saved.pods) unsavedLabels.push('Mark POD Targets');

  if (unsavedLabels.length) {
    const ok = window.confirm(
      `Some steps weren't saved:\n  • ${unsavedLabels.join('\n  • ')}\n\n` +
      'Finish anyway? Any unsaved work will be lost.',
    );
    if (!ok) return;
  }

  saveOk.value = ''; saveError.value = '';
  ui.showComplete = true;
}

function onStartOver(): void {
  // Reset every in-memory store so the user can map another room without
  // re-launching the app. Source-file paths in the project store are
  // cleared too, so the new project starts from a blank Setup form.
  mapper.reset();
  project.reset();
  ui.reset();           // also sets currentStep = 'setup', showComplete = false
  saveOk.value = '';
  saveError.value = '';
}

async function closeWindow(): Promise<void> {
  try {
    const { getCurrentWindow } = await import('@tauri-apps/api/window');
    await getCurrentWindow().close();
  } catch {
    // Browser fallback (e.g. running via `npm run dev` without Tauri).
    // Browsers won't let scripts close arbitrary tabs; surface a message
    // so the user knows to close manually.
    window.alert('All done. You can close this window now.');
  }
}

// Rows shown on the completion screen. Each row reports a per-step save
// status + the filename that would be / was produced for that step.
const savedRows = computed(() => {
  const name = (project.projectName || 'project').trim() || 'project';
  return [
    { label: 'Camera ↔ Map mapping', saved: ui.saved.align,    filename: `${name}_mapping.txt` },
    { label: 'Entry zones',           saved: ui.saved.entries,  filename: `${name}_entry_polygons.txt` },
    { label: 'POD targets',           saved: ui.saved.pods,     filename: `${name}_POD_points.txt` },
    { label: 'Room boundary',         saved: ui.saved.boundary, filename: `${name}_room_boundary.txt` },
  ];
});

async function onSave(): Promise<void> {
  if (!canSave.value) return;
  saving.value = true;
  saveOk.value = '';
  saveError.value = '';
  try {
    let res;
    switch (ui.currentStep) {
      case 'align':
        res = await saveArtifact({
          output_dir: project.outputDir,
          project_name: project.projectName,
          kind: 'mapping',
          mapping: mapper.pairs
            .filter((p) => p.mx != null && p.my != null)
            .map((p) => ({ fx: p.fx, fy: p.fy, mx: p.mx as number, my: p.my as number })),
        });
        break;
      case 'entries':
        res = await saveArtifact({
          output_dir: project.outputDir,
          project_name: project.projectName,
          kind: 'entry_polygons',
          polygons: mapper.entryPolygons,
        });
        break;
      case 'pods':
        res = await saveArtifact({
          output_dir: project.outputDir,
          project_name: project.projectName,
          kind: 'pod_points',
          points: mapper.podPoints,
        });
        break;
      case 'boundary':
        res = await saveArtifact({
          output_dir: project.outputDir,
          project_name: project.projectName,
          kind: 'room_boundary',
          polygons: [mapper.boundary],
        });
        break;
      default:
        return;
    }
    ui.markSaved(ui.currentStep, true);
    saveOk.value = `Saved to ${res.path}`;
  } catch (e) {
    saveError.value = `Save failed: ${(e as Error).message}`;
  } finally {
    saving.value = false;
  }
}
</script>

<style scoped>
.app {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: var(--color-bg);
  color: var(--color-text);
}

.topbar {
  padding: 12px 16px 8px;
  border-bottom: 1px solid var(--color-border);
  background: var(--color-bg-elev);
}
.topbar h1 {
  margin: 0;
  font-size: 1.15em;
  font-weight: 700;
  letter-spacing: 0.02em;
}
.topbar p {
  margin: 2px 0 0;
  font-size: 0.85em;
}
.muted { color: var(--color-muted); }

.body {
  flex: 1;
  min-height: 0;
  overflow: hidden;
}
.body > * {
  height: 100%;
}

.footer {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 16px;
  background: var(--color-bg-elev);
  border-top: 1px solid var(--color-border);
}
.footer .status {
  flex: 1;
  margin: 0;
  font-size: 0.88em;
}
.status.ok       { color: var(--color-success); }
.status.error    { color: var(--color-danger-text); }
.status.muted    { color: var(--color-muted); }

button.primary, button.secondary {
  border: none;
  padding: 8px 18px;
  border-radius: 5px;
  font-size: 0.9em;
  font-weight: 600;
  cursor: pointer;
}
button.primary {
  background: var(--color-accent);
  color: var(--color-bg);
}
button.primary:hover:not(:disabled) { background: var(--color-accent-soft); color: var(--color-text); }
button.primary:disabled { opacity: 0.5; cursor: not-allowed; }
button.secondary {
  background: var(--color-bg-deep);
  color: var(--color-text);
  border: 1px solid var(--color-border);
}
button.secondary:hover:not(:disabled) { background: var(--color-border); }
button.secondary:disabled { opacity: 0.5; cursor: not-allowed; }

/* ----- Completion screen ----- */
.complete {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 28px;
  overflow: auto;
}
.complete-card {
  width: 100%;
  max-width: 560px;
  background: var(--color-bg-elev);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 28px 28px 24px;
  text-align: center;
}
.complete-card .check {
  width: 56px;
  height: 56px;
  border-radius: 50%;
  background: var(--color-success-bg);
  color: var(--color-success);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.6em;
  font-weight: 700;
  margin: 0 auto 12px;
}
.complete-card h2 {
  margin: 0 0 4px;
  font-size: 1.4em;
  font-weight: 700;
}
.complete-card .path-line {
  margin: 4px 0 18px;
  padding: 6px 10px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.85em;
  background: var(--color-bg-deep);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  word-break: break-all;
}
.saved-list {
  list-style: none;
  margin: 0 0 18px;
  padding: 0;
  text-align: left;
}
.saved-list li {
  display: grid;
  grid-template-columns: 20px 1fr auto;
  gap: 10px;
  align-items: center;
  padding: 6px 4px;
  border-bottom: 1px solid var(--color-border);
  font-size: 0.9em;
}
.saved-list li:last-child { border-bottom: none; }
.saved-list .saved-icon {
  color: var(--color-success);
  font-weight: 700;
  text-align: center;
}
.saved-list li.unsaved .saved-icon { color: var(--color-muted); }
.saved-list li.unsaved .saved-label { color: var(--color-muted); }
.saved-list .saved-file {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.85em;
  color: var(--color-text-secondary);
}
.saved-list li.unsaved .saved-file { color: var(--color-muted); font-style: italic; }
.complete-actions {
  display: flex;
  justify-content: center;
  gap: 12px;
  margin-top: 8px;
}
.small { font-size: 0.85em; }
</style>
