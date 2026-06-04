// All editor state for the scaling workflow: walls (with real lengths), the
// global map scale derived from them, the A/B wall selection, the live point
// preview, and the committed reference points.

import { defineStore } from 'pinia';
import { computed, ref } from 'vue';

import {
  computeGlobalScale,
  placeFromWalls,
  sharedCorner,
  type Vec2,
  type Wall,
} from '@/utils/scaling';

export interface ScaledPoint {
  id: number;
  wallAId: number;
  wallBId: number;
  distFromA: number; // perpendicular distance to wall A, real units
  distFromB: number; // perpendicular distance to wall B, real units
  x: number; // computed map pixel
  y: number;
}

export const useScalerStore = defineStore('scaler', () => {
  const walls = ref<Wall[]>([]);
  const points = ref<ScaledPoint[]>([]);

  // Walls step: two-click wall drawing.
  const pendingWallStart = ref<Vec2 | null>(null);
  const pendingWall = ref<{ p1: Vec2; p2: Vec2 } | null>(null);

  // Points step: which two walls are selected, and the in-progress distances.
  const selAId = ref<number | null>(null);
  const selBId = ref<number | null>(null);
  const draftDistA = ref<number>(0);
  const draftDistB = ref<number>(0);

  let wallSeq = 1;
  let pointSeq = 1;

  // --- derived ---------------------------------------------------------
  const globalScale = computed(() => computeGlobalScale(walls.value));

  function getWall(id: number | null): Wall | undefined {
    if (id == null) return undefined;
    return walls.value.find((w) => w.id === id);
  }

  const selectionCorner = computed<Vec2 | null>(() => {
    const a = getWall(selAId.value);
    const b = getWall(selBId.value);
    if (!a || !b) return null;
    return sharedCorner(a, b);
  });

  // Live preview of the point that "Add point" would create.
  const ghost = computed(() => {
    const a = getWall(selAId.value);
    const b = getWall(selBId.value);
    if (!a || !b) return { ok: false as const, error: 'Select wall A and wall B.' };
    return placeFromWalls(a, b, draftDistA.value, draftDistB.value, globalScale.value.scale);
  });

  // --- walls -----------------------------------------------------------
  function wallClickVertex(p: Vec2): void {
    if (pendingWall.value) return; // waiting for length entry
    if (pendingWallStart.value == null) {
      pendingWallStart.value = p;
    } else {
      const start = pendingWallStart.value;
      // Ignore degenerate clicks.
      if (Math.hypot(p[0] - start[0], p[1] - start[1]) < 3) return;
      pendingWall.value = { p1: start, p2: p };
      pendingWallStart.value = null;
    }
  }

  function confirmWall(realLength: number): boolean {
    if (!pendingWall.value || !(realLength > 0)) return false;
    walls.value = [
      ...walls.value,
      { id: wallSeq++, p1: pendingWall.value.p1, p2: pendingWall.value.p2, realLength },
    ];
    pendingWall.value = null;
    recomputeAllPoints();
    return true;
  }

  function cancelPendingWall(): void {
    pendingWall.value = null;
    pendingWallStart.value = null;
  }

  function editWallLength(id: number, realLength: number): void {
    if (!(realLength > 0)) return;
    walls.value = walls.value.map((w) => (w.id === id ? { ...w, realLength } : w));
    recomputeAllPoints();
  }

  function deleteWall(id: number): void {
    walls.value = walls.value.filter((w) => w.id !== id);
    // Drop any points that depended on this wall.
    points.value = points.value.filter((p) => p.wallAId !== id && p.wallBId !== id);
    if (selAId.value === id) selAId.value = null;
    if (selBId.value === id) selBId.value = null;
  }

  function clearWalls(): void {
    walls.value = [];
    points.value = [];
    selAId.value = null;
    selBId.value = null;
    cancelPendingWall();
  }

  // --- selection -------------------------------------------------------
  function markWallA(id: number): void {
    selAId.value = id;
    if (selBId.value === id) selBId.value = null;
  }
  function markWallB(id: number): void {
    selBId.value = id;
    if (selAId.value === id) selAId.value = null;
  }
  function clearSelection(): void {
    selAId.value = null;
    selBId.value = null;
  }

  // --- points ----------------------------------------------------------
  function addPoint(): { ok: boolean; error?: string } {
    const res = ghost.value;
    if (!res.ok || !res.point || selAId.value == null || selBId.value == null) {
      return { ok: false, error: res.ok ? 'Selection incomplete.' : res.error };
    }
    points.value = [
      ...points.value,
      {
        id: pointSeq++,
        wallAId: selAId.value,
        wallBId: selBId.value,
        distFromA: draftDistA.value,
        distFromB: draftDistB.value,
        x: res.point[0],
        y: res.point[1],
      },
    ];
    draftDistA.value = 0;
    draftDistB.value = 0;
    return { ok: true };
  }

  function deletePoint(id: number): void {
    points.value = points.value.filter((p) => p.id !== id);
  }
  function clearPoints(): void {
    points.value = [];
  }

  function recomputeAllPoints(): void {
    const scale = globalScale.value.scale;
    points.value = points.value.map((p) => {
      const a = getWall(p.wallAId);
      const b = getWall(p.wallBId);
      if (!a || !b) return p;
      const res = placeFromWalls(a, b, p.distFromA, p.distFromB, scale);
      return res.ok && res.point ? { ...p, x: res.point[0], y: res.point[1] } : p;
    });
  }

  // --- persistence / reset --------------------------------------------
  function snapshot(): Record<string, unknown> {
    return {
      version: 1,
      scale_px_per_unit: globalScale.value.scale,
      walls: walls.value,
      points: points.value,
    };
  }

  function reset(): void {
    walls.value = [];
    points.value = [];
    selAId.value = null;
    selBId.value = null;
    draftDistA.value = 0;
    draftDistB.value = 0;
    cancelPendingWall();
    wallSeq = 1;
    pointSeq = 1;
  }

  return {
    // state
    walls,
    points,
    pendingWallStart,
    pendingWall,
    selAId,
    selBId,
    draftDistA,
    draftDistB,
    // derived
    globalScale,
    selectionCorner,
    ghost,
    getWall,
    // walls
    wallClickVertex,
    confirmWall,
    cancelPendingWall,
    editWallLength,
    deleteWall,
    clearWalls,
    // selection
    markWallA,
    markWallB,
    clearSelection,
    // points
    addPoint,
    deletePoint,
    clearPoints,
    recomputeAllPoints,
    // misc
    snapshot,
    reset,
  };
});
