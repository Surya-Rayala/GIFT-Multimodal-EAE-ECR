<template>
  <div class="step">
    <HelpCard
      title="Outline the entry zone(s) on the map"
      :steps="[
        'Click on the map to drop the polygon vertices in order (clockwise or counter-clockwise).',
        'Do NOT close the polygon manually — press Confirm and it closes for you.',
        'Press New polygon to start the next entry zone.',
        'Drag any draft point to fine-tune. Undo removes the last point.',
      ]"
      why="The engine treats anyone inside one of these polygons as 'at the door' — this drives the entry-detection logic for every metric."
    />

    <div class="row">
      <MapperCanvas
        :src="mapSrc"
        title="Map"
        emptyText="Pick a map image in Setup."
        :markers="markers"
        :polylines="polylines"
        @pointClicked="onClick"
        @pointMoved="onMove"
      />
    </div>

    <div class="actions">
      <button @click="mapper.entryUndo()" :disabled="!mapper.entryDraft.length">Undo point</button>
      <button @click="onConfirm" :disabled="mapper.entryDraft.length < 3">Confirm polygon</button>
      <button @click="onNew" :disabled="mapper.entryDraft.length < 3">New polygon</button>
      <button @click="onClear" :disabled="!mapper.entryPolygons.length && !mapper.entryDraft.length">Clear all</button>
      <span class="count">
        {{ mapper.entryPolygons.length }} saved
        {{ mapper.entryDraft.length ? `· draft (${mapper.entryDraft.length} pts)` : '' }}
      </span>
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
import { CONFIRMED_COLOR, DRAFT_COLOR } from '@/utils/palette';

const project = useProjectStore();
const mapper = useMapperStore();

const mapSrc = computed(() => (project.mapImagePath ? imageUrl(project.mapImagePath) : ''));

const polylines = computed(() => {
  const out: Array<{ points: [number, number][]; closed: boolean; color: string; width?: number }> = [];
  for (const poly of mapper.entryPolygons) {
    out.push({ points: poly, closed: true, color: CONFIRMED_COLOR, width: 2.5 });
  }
  if (mapper.entryDraft.length) {
    out.push({
      points: mapper.entryDraft,
      closed: false,
      color: DRAFT_COLOR,
      width: 2.5,
    });
  }
  return out;
});

const markers = computed(() =>
  mapper.entryDraft.map((p, i) => ({
    x: p[0],
    y: p[1],
    label: String(i + 1),
    color: DRAFT_COLOR,
    handleId: `draft:${i}`,
  })),
);

function onClick(x: number, y: number): void {
  mapper.entryAddPoint([x, y]);
}
function onMove(id: string, x: number, y: number): void {
  if (!id.startsWith('draft:')) return;
  mapper.entryMoveDraftPoint(Number(id.slice(6)), [x, y]);
}

function onConfirm(): void {
  const ok = mapper.entryConfirm(3);
  if (!ok) window.alert('A polygon needs at least 3 points.');
}
function onNew(): void {
  if (mapper.entryConfirm(3)) {
    // entryConfirm auto-clears the draft; nothing else to do.
  } else {
    window.alert('Finish the current polygon first (≥3 points).');
  }
}
function onClear(): void {
  if (window.confirm('Clear all entry polygons (including the current draft)?')) {
    mapper.entryClear();
  }
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
