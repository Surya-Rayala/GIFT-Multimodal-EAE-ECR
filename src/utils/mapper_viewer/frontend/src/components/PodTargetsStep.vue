<template>
  <div class="step">
    <HelpCard
      title="Drop a pin on each POD"
      :steps="[
        'Click anywhere on the map to add a Point-Of-Domination.',
        'Drag any pin to fine-tune its position.',
        'Undo removes the last pin. Clear all removes everything.',
      ]"
      why="Each POD is a target the team is expected to reach. The engine compares each entrant's path against the assigned POD to score capture timing and coverage."
    />

    <div class="row">
      <MapperCanvas
        :src="mapSrc"
        title="Map"
        emptyText="Pick a map image in Setup."
        :markers="markers"
        @pointClicked="onClick"
        @pointMoved="onMove"
      />
    </div>

    <div class="actions">
      <button @click="mapper.podUndo()" :disabled="!mapper.podPoints.length">Undo</button>
      <button @click="onClear" :disabled="!mapper.podPoints.length">Clear all</button>
      <span class="count">{{ mapper.podPoints.length }} POD{{ mapper.podPoints.length === 1 ? '' : 's' }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';

import HelpCard from './HelpCard.vue';
import MapperCanvas from './MapperCanvas.vue';

import { imageUrl } from '@/api/client';
import { useMapperStore } from '@/stores/mapper';
import { useProjectStore } from '@/stores/project';
import { POD_COLOR } from '@/utils/palette';

const project = useProjectStore();
const mapper = useMapperStore();

const mapSrc = computed(() => (project.mapImagePath ? imageUrl(project.mapImagePath) : ''));

const markers = computed(() =>
  mapper.podPoints.map((p, i) => ({
    x: p[0],
    y: p[1],
    label: String(i + 1),
    color: POD_COLOR,
    handleId: `pod:${i}`,
  })),
);

function onClick(x: number, y: number): void {
  mapper.podAdd([x, y]);
}
function onMove(id: string, x: number, y: number): void {
  if (!id.startsWith('pod:')) return;
  mapper.podMove(Number(id.slice(4)), [x, y]);
}
function onClear(): void {
  if (window.confirm('Clear all POD points?')) mapper.podClear();
}
</script>

<style scoped>
.step {
  display: flex;
  flex-direction: column;
  height: 100%;
  gap: 10px;
  padding: 14px 16px;
  min-height: 0;
}
.row { flex: 1; min-height: 0; }
.actions {
  display: flex;
  align-items: center;
  gap: 8px;
}
.actions button {
  background: var(--color-bg-elev);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  padding: 5px 12px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.88em;
}
.actions button:hover:not(:disabled) { background: var(--color-border); }
.actions button:disabled { opacity: 0.5; cursor: not-allowed; }
.count { margin-left: auto; color: var(--color-muted); font-size: 0.85em; }
</style>
