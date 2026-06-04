<template>
  <div class="step">
    <HelpCard
      title="Trace the room walls"
      :steps="[
        'Click vertices along the inside of the walls (clockwise or counter-clockwise).',
        'Do NOT close the polygon manually — Confirm closes it for you.',
        'Drag a vertex to fine-tune. Undo removes the last point.',
        'Press Clear if you want to redraw from scratch.',
      ]"
      why="The room outline anchors several metrics: 'stay along wall' measures distance from this boundary, and 'floor coverage' uses it to compute the room's interior area."
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
      <button @click="mapper.boundaryUndo()" :disabled="!mapper.boundary.length || mapper.boundaryConfirmed">Undo</button>
      <button
        @click="onConfirm"
        :disabled="mapper.boundary.length < 4 || mapper.boundaryConfirmed"
      >
        Confirm boundary
      </button>
      <button @click="onClear" :disabled="!mapper.boundary.length">Clear</button>
      <span class="count">
        {{ mapper.boundary.length }} point{{ mapper.boundary.length === 1 ? '' : 's' }}
        {{ mapper.boundaryConfirmed ? '· confirmed' : '' }}
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
  if (!mapper.boundary.length) return [];
  const color = mapper.boundaryConfirmed ? CONFIRMED_COLOR : DRAFT_COLOR;
  return [{
    points: mapper.boundary,
    closed: mapper.boundaryConfirmed,
    color,
    width: 2.5,
  }];
});

const markers = computed(() => {
  if (mapper.boundaryConfirmed) return [];
  return mapper.boundary.map((p, i) => ({
    x: p[0],
    y: p[1],
    label: String(i + 1),
    color: DRAFT_COLOR,
    handleId: `bnd:${i}`,
  }));
});

function onClick(x: number, y: number): void {
  mapper.boundaryAdd([x, y]);
}
function onMove(id: string, x: number, y: number): void {
  if (!id.startsWith('bnd:')) return;
  mapper.boundaryMove(Number(id.slice(4)), [x, y]);
}

function onConfirm(): void {
  const ok = mapper.boundaryConfirm(4);
  if (!ok) window.alert('Boundary needs at least 4 points.');
}
function onClear(): void {
  if (window.confirm('Clear the room boundary?')) mapper.boundaryClear();
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
