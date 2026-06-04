<template>
  <div class="align">
    <HelpCard
      title="Click matching pairs on the camera and the map"
      :steps="[
        'Click a recognizable spot in the camera view (a corner, a doorway).',
        'Click the same spot on the map.',
        'Repeat ' + (mapper.pairs.length < 4 ? 'at least 4 times' : 'with more points if needed') + '. Spread the pairs across the whole scene.',
        'Drag any point to fine-tune. Undo removes the last action.',
      ]"
      why="These pairs compute a homography that lets the engine project tracker positions from the camera image onto the map plane. More well-spread pairs = more accurate projection."
    />

    <div class="status-line" :class="{ awaiting: needsMap }">
      <strong>Next:</strong>
      <span v-if="!camReady || !mapReady">load a map and a camera reference in Setup first.</span>
      <span v-else-if="needsMap">click the MATCHING point on the map.</span>
      <span v-else>click a recognizable spot on the camera.</span>
    </div>

    <div class="canvases">
      <MapperCanvas
        :src="camSrc"
        title="Camera"
        emptyText="Pick a video or frame image in Setup."
        :markers="camMarkers"
        @pointClicked="onCameraClick"
        @pointMoved="onCameraMove"
      />
      <MapperCanvas
        :src="mapSrc"
        title="Map"
        emptyText="Pick a map image in Setup."
        :markers="mapMarkers"
        @pointClicked="onMapClick"
        @pointMoved="onMapMove"
      />
    </div>

    <div class="actions">
      <button @click="mapper.undoPair()" :disabled="!mapper.pairs.length">Undo</button>
      <button @click="onClear" :disabled="!mapper.pairs.length">Clear all</button>
      <span class="pair-count">{{ mapper.pairs.length }} pair{{ mapper.pairs.length === 1 ? '' : 's' }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';

import HelpCard from './HelpCard.vue';
import MapperCanvas from './MapperCanvas.vue';

import { imageUrl, videoFrameUrl } from '@/api/client';
import { useMapperStore } from '@/stores/mapper';
import { useProjectStore } from '@/stores/project';
import { trackPalette } from '@/utils/palette';

const project = useProjectStore();
const mapper = useMapperStore();

const camReady = computed(
  () => !!project.cameraFrameImagePath || !!project.cameraVideoPath,
);
const mapReady = computed(() => !!project.mapImagePath);
const needsMap = computed(() => mapper.expectedClick === 'map');

const camSrc = computed(() => {
  if (project.cameraVideoPath) {
    return videoFrameUrl(project.cameraVideoPath, project.cameraFrameIndex);
  }
  if (project.cameraFrameImagePath) {
    return imageUrl(project.cameraFrameImagePath);
  }
  return '';
});

const mapSrc = computed(() => (project.mapImagePath ? imageUrl(project.mapImagePath) : ''));

const camMarkers = computed(() =>
  mapper.pairs.map((p, i) => ({
    x: p.fx,
    y: p.fy,
    label: String(i + 1),
    color: trackPalette[i % trackPalette.length],
    handleId: `cam:${i}`,
  })),
);

const mapMarkers = computed(() =>
  mapper.pairs
    .map((p, i) => (p.mx != null && p.my != null
      ? {
          x: p.mx,
          y: p.my,
          label: String(i + 1),
          color: trackPalette[i % trackPalette.length],
          handleId: `map:${i}`,
        }
      : null))
    .filter((x): x is NonNullable<typeof x> => x != null),
);

function onCameraClick(x: number, y: number): void {
  if (!camReady.value || !mapReady.value) return;
  if (mapper.expectedClick !== 'camera') return;
  mapper.appendCameraPoint([x, y]);
}

function onMapClick(x: number, y: number): void {
  if (!camReady.value || !mapReady.value) return;
  if (mapper.expectedClick !== 'map') return;
  mapper.appendMapPoint([x, y]);
}

function onCameraMove(id: string, x: number, y: number): void {
  if (!id.startsWith('cam:')) return;
  const idx = Number(id.slice(4));
  mapper.moveCameraPoint(idx, [x, y]);
}

function onMapMove(id: string, x: number, y: number): void {
  if (!id.startsWith('map:')) return;
  const idx = Number(id.slice(4));
  mapper.moveMapPoint(idx, [x, y]);
}

function onClear(): void {
  if (window.confirm('Clear all mapping pairs?')) {
    mapper.clearPairs();
  }
}
</script>

<style scoped>
.align {
  display: flex;
  flex-direction: column;
  height: 100%;
  gap: 10px;
  padding: 14px 16px;
  min-height: 0;
}
.canvases {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
  flex: 1;
  min-height: 0;
}
.status-line {
  font-size: 0.92em;
  padding: 8px 12px;
  background: var(--color-bg-elev);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  border-left: 3px solid var(--color-accent);
}
.status-line.awaiting {
  border-left-color: var(--color-warning);
}
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
.actions button:hover:not(:disabled) {
  background: var(--color-border);
}
.actions button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.pair-count {
  margin-left: auto;
  color: var(--color-muted);
  font-size: 0.85em;
}
</style>
