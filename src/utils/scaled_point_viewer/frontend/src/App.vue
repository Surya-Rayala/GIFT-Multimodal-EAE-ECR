<template>
  <div class="app">
    <header class="topbar">
      <h1>Scaled Point Viewer</h1>
      <p class="muted">{{ subtitle }}</p>
    </header>

    <StepBar />

    <main class="body">
      <SetupStep v-if="ui.currentStep === 'setup'" />
      <WallsStep v-else-if="ui.currentStep === 'walls'" />
      <PointsStep v-else-if="ui.currentStep === 'points'" />
    </main>

    <footer class="footer">
      <button class="secondary" @click="onBack" :disabled="!canGoBack">← Back</button>

      <p v-if="saveError" class="status error">{{ saveError }}</p>
      <p v-else-if="saveOk" class="status ok">{{ saveOk }}</p>
      <p v-else class="status muted">{{ footerText }}</p>

      <button
        v-if="ui.currentStep !== 'points'"
        class="primary"
        @click="onNext"
        :disabled="!canGoNext"
        :title="nextBlockedReason"
      >
        Next →
      </button>
      <button
        v-else
        class="primary"
        @click="onSave"
        :disabled="saving || !canSave"
        :title="canSave ? '' : saveBlockedReason"
      >
        {{ saving ? 'Saving…' : 'Save points' }}
      </button>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';

import StepBar from './components/StepBar.vue';
import SetupStep from './components/SetupStep.vue';
import WallsStep from './components/WallsStep.vue';
import PointsStep from './components/PointsStep.vue';

import { saveScaledPoints } from '@/api/client';
import { useUIStore } from '@/stores/ui';
import { useProjectStore } from '@/stores/project';
import { useScalerStore } from '@/stores/scaler';
import { STEPS, type StepKey } from '@/types/models';

const ui = useUIStore();
const project = useProjectStore();
const scaler = useScalerStore();

const saving = ref(false);
const saveOk = ref('');
const saveError = ref('');

const subtitle = computed(() => STEPS.find((s) => s.key === ui.currentStep)?.subtitle ?? '');
const stepIndex = computed(() => STEPS.findIndex((s) => s.key === ui.currentStep));
const canGoBack = computed(() => stepIndex.value > 0);

const nextBlockedReason = computed(() => {
  if (ui.currentStep === 'setup') {
    if (!project.mapImagePath) return 'Pick a map image first.';
    if (!project.outputDir) return 'Pick a save folder first.';
    if (!project.projectName.trim()) return 'Project name cannot be empty.';
  }
  if (ui.currentStep === 'walls' && !scaler.walls.length) return 'Draw at least one wall first.';
  return '';
});
const canGoNext = computed(() => stepIndex.value < STEPS.length - 1 && nextBlockedReason.value === '');

const saveBlockedReason = computed(() => {
  if (!project.mapImagePath) return 'Pick a map image in Setup.';
  if (!project.outputDir) return 'Pick a save folder in Setup.';
  if (!project.projectName.trim()) return 'Set a project name in Setup.';
  if (!scaler.points.length) return 'Add at least one point.';
  return '';
});
const canSave = computed(() => saveBlockedReason.value === '');

const footerText = computed(() => {
  switch (ui.currentStep) {
    case 'setup':
      return 'Fill in the fields, then press Next.';
    case 'walls':
      return scaler.walls.length
        ? `${scaler.walls.length} wall${scaler.walls.length === 1 ? '' : 's'} — scale ${scaler.globalScale.scale.toFixed(1)} px/${project.unit}.`
        : 'Click two points to draw a wall, then type its length.';
    case 'points':
      return scaler.points.length
        ? `${scaler.points.length} point${scaler.points.length === 1 ? '' : 's'}. Save to write the files.`
        : 'Select two walls and type the distances to add a point.';
  }
  return '';
});

function onBack(): void {
  if (stepIndex.value <= 0) return;
  ui.goTo(STEPS[stepIndex.value - 1].key as StepKey);
  saveOk.value = '';
  saveError.value = '';
}
function onNext(): void {
  if (!canGoNext.value) return;
  ui.goTo(STEPS[stepIndex.value + 1].key as StepKey);
  saveOk.value = '';
  saveError.value = '';
}

async function onSave(): Promise<void> {
  if (!canSave.value) return;
  saving.value = true;
  saveOk.value = '';
  saveError.value = '';
  try {
    const res = await saveScaledPoints({
      output_dir: project.outputDir,
      project_name: project.projectName,
      map_image_path: project.mapImagePath,
      points: scaler.points.map((p) => [p.x, p.y] as [number, number]),
      project: { unit: project.unit, ...scaler.snapshot() },
    });
    saveOk.value = `Saved ${res.count} point(s) → ${res.points_path}`;
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
.topbar p { margin: 2px 0 0; font-size: 0.85em; }
.muted { color: var(--color-muted); }

.body { flex: 1; min-height: 0; overflow: hidden; }
.body > * { height: 100%; }

.footer {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 16px;
  background: var(--color-bg-elev);
  border-top: 1px solid var(--color-border);
}
.footer .status { flex: 1; margin: 0; font-size: 0.88em; }
.status.ok { color: var(--color-success); }
.status.error { color: var(--color-danger-text); }
.status.muted { color: var(--color-muted); }

button.primary,
button.secondary {
  border: none;
  padding: 8px 18px;
  border-radius: 5px;
  font-size: 0.9em;
  font-weight: 600;
  cursor: pointer;
}
button.primary { background: var(--color-accent); color: var(--color-bg); }
button.primary:hover:not(:disabled) { background: var(--color-accent-soft); color: var(--color-text); }
button.primary:disabled { opacity: 0.5; cursor: not-allowed; }
button.secondary {
  background: var(--color-bg-deep);
  color: var(--color-text);
  border: 1px solid var(--color-border);
}
button.secondary:hover:not(:disabled) { background: var(--color-border); }
button.secondary:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
