<template>
  <div class="step">
    <aside class="panel">
      <HelpCard
        title="Place a point from two distances"
        :steps="[
          'Click a wall to set Wall A (blue), then another wall at the same corner for Wall B (green).',
          'Type the point’s straight-line distance to each wall, then Add point.',
          'The orange preview shows where it lands before you commit.',
        ]"
        why="Each distance is the perpendicular (shortest) distance from the point to that wall — measured straight out at a right angle, not diagonally to the corner. Keep every number in the same unit you used for the walls."
      />

      <!-- A / B selection -->
      <div class="sel-card">
        <div class="sel-row">
          <span class="sel-tag a">A</span>
          <span class="sel-val">{{ scaler.selAId != null ? `Wall ${scaler.selAId}` : 'click a wall' }}</span>
          <button v-if="scaler.selAId != null" class="del" @click="scaler.selAId = null">✕</button>
        </div>
        <div class="sel-row">
          <span class="sel-tag b">B</span>
          <span class="sel-val">{{ scaler.selBId != null ? `Wall ${scaler.selBId}` : 'click a wall' }}</span>
          <button v-if="scaler.selBId != null" class="del" @click="scaler.selBId = null">✕</button>
        </div>
      </div>

      <!-- distances -->
      <div class="dist-card">
        <label>
          Distance from Wall A
          <div class="dist-row">
            <input class="num" type="number" min="0" step="any" v-model.number="scaler.draftDistA" />
            <span class="unit">{{ project.unit }}</span>
          </div>
        </label>
        <label>
          Distance from Wall B
          <div class="dist-row">
            <input class="num" type="number" min="0" step="any" v-model.number="scaler.draftDistB" />
            <span class="unit">{{ project.unit }}</span>
          </div>
        </label>

        <p v-if="scaler.ghost.ok" class="ghost-ok">
          Preview: ({{ Math.round(scaler.ghost.point![0]) }}, {{ Math.round(scaler.ghost.point![1]) }})
        </p>
        <p v-else class="ghost-err">{{ scaler.ghost.error }}</p>

        <button class="primary" :disabled="!scaler.ghost.ok" @click="onAdd">Add point</button>
        <p v-if="addError" class="ghost-err">{{ addError }}</p>
      </div>

      <!-- points list -->
      <div class="pt-list">
        <div v-for="(p, i) in scaler.points" :key="p.id" class="pt-item">
          <span class="dot" :style="{ background: POINT_COLOR }" />
          <span class="pid">#{{ i + 1 }}</span>
          <span class="coord">({{ Math.round(p.x) }}, {{ Math.round(p.y) }})</span>
          <button class="del" title="Delete point" @click="scaler.deletePoint(p.id)">✕</button>
        </div>
        <p v-if="!scaler.points.length" class="muted small empty-hint">No points yet.</p>
      </div>
      <button v-if="scaler.points.length" class="ghost clear" @click="onClear">Clear all points</button>
    </aside>

    <div class="canvas-col">
      <ScalerCanvas
        :src="mapSrc"
        title="Map"
        emptyText="Pick a map image in Setup."
        hint="Click a wall to mark A, then B."
        mode="points"
        :walls="wallShapes"
        :points="pointShapes"
        :ghost="ghostPoint"
        :corner="scaler.selectionCorner"
        @wallPicked="onWallPicked"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';

import HelpCard from './HelpCard.vue';
import ScalerCanvas from './ScalerCanvas.vue';

import { imageUrl } from '@/api/client';
import { useProjectStore } from '@/stores/project';
import { useScalerStore } from '@/stores/scaler';
import { POINT_COLOR, WALL_A_COLOR, WALL_B_COLOR, WALL_COLOR } from '@/utils/palette';

const project = useProjectStore();
const scaler = useScalerStore();

const addError = ref('');

const mapSrc = computed(() => (project.mapImagePath ? imageUrl(project.mapImagePath) : ''));

const wallShapes = computed(() =>
  scaler.walls.map((w) => {
    const isA = w.id === scaler.selAId;
    const isB = w.id === scaler.selBId;
    return {
      id: w.id,
      p1: w.p1,
      p2: w.p2,
      color: isA ? WALL_A_COLOR : isB ? WALL_B_COLOR : WALL_COLOR,
      label: isA ? 'A' : isB ? 'B' : undefined,
      width: isA || isB ? 4 : 3,
    };
  }),
);

const pointShapes = computed(() =>
  scaler.points.map((p, i) => ({ x: p.x, y: p.y, color: POINT_COLOR, label: String(i + 1) })),
);

const ghostPoint = computed(() =>
  scaler.ghost.ok && scaler.ghost.point ? { x: scaler.ghost.point[0], y: scaler.ghost.point[1] } : null,
);

function onWallPicked(id: number): void {
  if (scaler.selAId === id) {
    scaler.selAId = null;
    return;
  }
  if (scaler.selBId === id) {
    scaler.selBId = null;
    return;
  }
  if (scaler.selAId == null) scaler.markWallA(id);
  else if (scaler.selBId == null) scaler.markWallB(id);
  else {
    // Both full — start over with this as A.
    scaler.markWallA(id);
    scaler.selBId = null;
  }
}

function onAdd(): void {
  addError.value = '';
  const res = scaler.addPoint();
  if (!res.ok) addError.value = res.error || 'Could not add point.';
}

function onClear(): void {
  if (window.confirm('Clear all points?')) scaler.clearPoints();
}
</script>

<style scoped>
.step {
  display: flex;
  height: 100%;
  gap: 12px;
  padding: 12px 14px;
  min-height: 0;
}
.panel {
  width: 320px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
  overflow: auto;
}
.canvas-col { flex: 1; min-width: 0; }

.sel-card,
.dist-card,
.pt-list {
  background: var(--color-bg-elev);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  padding: 10px 12px;
}
.sel-row { display: flex; align-items: center; gap: 8px; padding: 3px 0; }
.sel-tag {
  width: 22px;
  height: 22px;
  border-radius: 50%;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 0.8em;
  color: #fff;
}
.sel-tag.a { background: var(--color-accent); }
.sel-tag.b { background: var(--color-success); color: var(--color-bg); }
.sel-val { flex: 1; font-size: 0.88em; color: var(--color-text-secondary); }

.dist-card { display: flex; flex-direction: column; gap: 10px; }
.dist-card label { font-size: 0.8em; color: var(--color-muted); display: flex; flex-direction: column; gap: 4px; }
.dist-row { display: flex; align-items: center; gap: 6px; }
.num {
  background: var(--color-bg-deep);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  padding: 6px 8px;
  font-size: 0.9em;
  width: 120px;
}
.unit { color: var(--color-muted); font-size: 0.85em; }
.ghost-ok { margin: 0; font-size: 0.85em; color: var(--color-success); font-family: ui-monospace, Menlo, monospace; }
.ghost-err { margin: 0; font-size: 0.82em; color: var(--color-warning); }

.pt-list { display: flex; flex-direction: column; gap: 4px; }
.pt-item { display: flex; align-items: center; gap: 8px; font-size: 0.85em; padding: 3px 2px; }
.dot { width: 10px; height: 10px; border-radius: 50%; }
.pid { width: 30px; color: var(--color-text-secondary); }
.coord { flex: 1; font-family: ui-monospace, Menlo, monospace; color: var(--color-text-secondary); }
.del {
  background: transparent;
  border: none;
  color: var(--color-muted);
  cursor: pointer;
  font-size: 0.9em;
  padding: 2px 4px;
}
.del:hover { color: var(--color-danger-text); }

.primary {
  background: var(--color-accent);
  color: var(--color-bg);
  border: none;
  padding: 7px 14px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.9em;
  font-weight: 600;
}
.primary:disabled { opacity: 0.5; cursor: not-allowed; }
.ghost {
  background: var(--color-bg-deep);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  padding: 6px 12px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.85em;
}
.ghost:hover { background: var(--color-border); }
.clear { align-self: flex-start; }
.empty-hint { padding: 6px; margin: 0; }
.muted { color: var(--color-muted); }
.small { font-size: 0.85em; }
</style>
