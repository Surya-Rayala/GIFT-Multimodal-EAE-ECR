<template>
  <div class="step">
    <aside class="panel">
      <HelpCard
        title="Trace walls, type their real lengths"
        :steps="[
          'Click two points on the map to draw a wall, then type its real-world length.',
          'Existing endpoints show as dots — click one to reuse a corner exactly.',
          'Draw the walls near each point you want to place (the two that meet at its corner).',
        ]"
        why="Every wall’s length sets the map scale (pixels per real unit). Using several walls — and longer ones — makes that scale accurate, so points stay correct even when a wall is short or cut by a corridor."
      />

      <!-- Pending wall: ask for its length -->
      <div v-if="scaler.pendingWall" class="length-card">
        <p class="length-title">New wall — how long is it really?</p>
        <div class="length-row">
          <input
            ref="lenInput"
            class="num"
            type="number"
            min="0"
            step="any"
            v-model.number="pendingLen"
            @keyup.enter="addWall"
            :placeholder="`length in ${project.unit}`"
          />
          <span class="unit">{{ project.unit }}</span>
        </div>
        <div class="length-actions">
          <button class="primary" @click="addWall" :disabled="!(pendingLen > 0)">Add wall</button>
          <button class="ghost" @click="scaler.cancelPendingWall()">Cancel</button>
        </div>
      </div>

      <!-- Scale readout -->
      <div class="scale-card">
        <span class="scale-label">Map scale</span>
        <span v-if="scaler.globalScale.scale > 0" class="scale-value">
          {{ scaler.globalScale.scale.toFixed(1) }} px / {{ project.unit }}
        </span>
        <span v-else class="scale-value muted">Add a wall to set it</span>
      </div>

      <!-- Wall list -->
      <div class="wall-list">
        <div v-for="w in scaler.walls" :key="w.id" class="wall-item">
          <span class="dot" :style="{ background: WALL_COLOR }" />
          <span class="wid">Wall {{ w.id }}</span>
          <input
            class="num small"
            type="number"
            min="0"
            step="any"
            :value="w.realLength"
            @change="onEditLen(w.id, ($event.target as HTMLInputElement).value)"
          />
          <span class="unit">{{ project.unit }}</span>
          <span class="badge" :class="badgeClass(w.id)">{{ badgeText(w.id) }}</span>
          <button class="del" title="Delete wall" @click="scaler.deleteWall(w.id)">✕</button>
        </div>
        <p v-if="!scaler.walls.length" class="muted small empty-hint">No walls yet.</p>
      </div>

      <button v-if="scaler.walls.length" class="ghost clear" @click="onClearWalls">Clear all walls</button>
    </aside>

    <div class="canvas-col">
      <ScalerCanvas
        :src="mapSrc"
        title="Map"
        emptyText="Pick a map image in Setup."
        hint="Click two points to draw a wall."
        mode="walls"
        :walls="wallShapes"
        :pendingStart="scaler.pendingWallStart"
        :snapPoints="snapPoints"
        @vertexClicked="onVertex"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue';

import HelpCard from './HelpCard.vue';
import ScalerCanvas from './ScalerCanvas.vue';

import { imageUrl } from '@/api/client';
import { useProjectStore } from '@/stores/project';
import { useScalerStore } from '@/stores/scaler';
import type { Vec2 } from '@/utils/scaling';
import { WALL_COLOR } from '@/utils/palette';

const project = useProjectStore();
const scaler = useScalerStore();

const pendingLen = ref<number>(0);
const lenInput = ref<HTMLInputElement>();

const mapSrc = computed(() => (project.mapImagePath ? imageUrl(project.mapImagePath) : ''));

const wallShapes = computed(() =>
  scaler.walls.map((w) => ({
    id: w.id,
    p1: w.p1,
    p2: w.p2,
    color: WALL_COLOR,
    label: `${w.realLength} ${project.unit}`,
  })),
);

const snapPoints = computed<Vec2[]>(() => scaler.walls.flatMap((w) => [w.p1, w.p2]));

// Focus the length field as soon as a wall's second point is placed.
watch(
  () => scaler.pendingWall,
  async (pw) => {
    if (pw) {
      pendingLen.value = 0;
      await nextTick();
      lenInput.value?.focus();
    }
  },
);

function onVertex(x: number, y: number): void {
  scaler.wallClickVertex([x, y]);
}

function addWall(): void {
  if (scaler.confirmWall(pendingLen.value)) pendingLen.value = 0;
}

function onEditLen(id: number, v: string): void {
  const n = Number(v);
  if (Number.isFinite(n) && n > 0) scaler.editWallLength(id, n);
}

function onClearWalls(): void {
  if (window.confirm('Clear all walls? This also removes points that used them.')) scaler.clearWalls();
}

function ratioOf(id: number): number {
  const r = scaler.globalScale.perWall.find((p) => p.id === id);
  return r ? r.ratio : 1;
}
function badgeClass(id: number): string {
  return Math.abs(ratioOf(id) - 1) <= 0.1 ? 'ok' : 'warn';
}
function badgeText(id: number): string {
  const r = ratioOf(id);
  return Math.abs(r - 1) <= 0.1 ? 'consistent' : `${r.toFixed(2)}×`;
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
.canvas-col {
  flex: 1;
  min-width: 0;
}

.length-card,
.scale-card {
  background: var(--color-bg-elev);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  padding: 10px 12px;
}
.length-card { border-left: 3px solid var(--color-accent); }
.length-title { margin: 0 0 8px; font-size: 0.9em; font-weight: 600; }
.length-row { display: flex; align-items: center; gap: 6px; }
.length-actions { display: flex; gap: 8px; margin-top: 8px; }

.scale-card { display: flex; align-items: center; justify-content: space-between; }
.scale-label { font-size: 0.78em; text-transform: uppercase; letter-spacing: 0.06em; color: var(--color-muted); }
.scale-value { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.9em; }

.wall-list {
  background: var(--color-bg-elev);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  padding: 6px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.wall-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 4px;
  font-size: 0.85em;
}
.dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.wid { width: 52px; flex-shrink: 0; color: var(--color-text-secondary); }
.num {
  background: var(--color-bg-deep);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  padding: 5px 7px;
  font-size: 0.9em;
  width: 90px;
}
.num.small { width: 64px; padding: 3px 6px; }
.unit { color: var(--color-muted); font-size: 0.85em; }
.badge {
  margin-left: auto;
  font-size: 0.72em;
  padding: 2px 6px;
  border-radius: 10px;
}
.badge.ok { background: var(--color-success-bg); color: var(--color-success); }
.badge.warn { background: var(--color-warning-bg); color: var(--color-warning); }
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
  padding: 6px 14px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.88em;
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
