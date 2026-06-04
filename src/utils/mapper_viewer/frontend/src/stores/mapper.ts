// Per-step data stores in one place: pairs (homography), entry polygons,
// POD points, and the room boundary polygon. Each step manipulates only its
// own slice. App.vue handles cross-cutting concerns (save, dirty flags).

import { defineStore } from 'pinia';
import { ref } from 'vue';

import type { MappingPair, Point } from '@/types/models';

export const useMapperStore = defineStore('mapper', () => {
  // Step 2 — Align Camera to Map (homography)
  // The "expected click" alternates camera → map → camera → … so we always
  // know which canvas the next click belongs to.
  const pairs = ref<MappingPair[]>([]);
  const expectedClick = ref<'camera' | 'map'>('camera');

  // Step 3 — Entry zones
  const entryPolygons = ref<Point[][]>([]);     // confirmed polygons
  const entryDraft = ref<Point[]>([]);          // polygon being drawn

  // Step 4 — POD points
  const podPoints = ref<Point[]>([]);

  // Step 5 — Room boundary (a single polygon, with confirm gate)
  const boundary = ref<Point[]>([]);
  const boundaryConfirmed = ref<boolean>(false);

  // --- Step 2: pair operations -----------------------------------------
  function appendCameraPoint(p: Point): void {
    if (expectedClick.value !== 'camera') return;
    pairs.value = [...pairs.value, { fx: p[0], fy: p[1], mx: null, my: null }];
    expectedClick.value = 'map';
  }
  function appendMapPoint(p: Point): void {
    if (expectedClick.value !== 'map') return;
    const last = pairs.value[pairs.value.length - 1];
    if (!last) return;
    last.mx = p[0];
    last.my = p[1];
    pairs.value = [...pairs.value];
    expectedClick.value = 'camera';
  }
  function undoPair(): void {
    if (expectedClick.value === 'map') {
      // Incomplete pair — just discard the camera click
      pairs.value = pairs.value.slice(0, -1);
      expectedClick.value = 'camera';
      return;
    }
    pairs.value = pairs.value.slice(0, -1);
    expectedClick.value = 'camera';
  }
  function deletePair(idx: number): void {
    if (expectedClick.value === 'map') return; // refuse mid-pair
    if (idx < 0 || idx >= pairs.value.length) return;
    pairs.value = pairs.value.filter((_, i) => i !== idx);
  }
  function clearPairs(): void {
    pairs.value = [];
    expectedClick.value = 'camera';
  }
  function moveCameraPoint(idx: number, p: Point): void {
    if (idx < 0 || idx >= pairs.value.length) return;
    pairs.value[idx].fx = p[0];
    pairs.value[idx].fy = p[1];
    pairs.value = [...pairs.value];
  }
  function moveMapPoint(idx: number, p: Point): void {
    if (idx < 0 || idx >= pairs.value.length) return;
    pairs.value[idx].mx = p[0];
    pairs.value[idx].my = p[1];
    pairs.value = [...pairs.value];
  }

  // --- Step 3: entry polygons ------------------------------------------
  function entryAddPoint(p: Point): void { entryDraft.value = [...entryDraft.value, p]; }
  function entryUndo(): void { entryDraft.value = entryDraft.value.slice(0, -1); }
  function entryMoveDraftPoint(idx: number, p: Point): void {
    if (idx < 0 || idx >= entryDraft.value.length) return;
    entryDraft.value[idx] = p;
    entryDraft.value = [...entryDraft.value];
  }
  function entryConfirm(minPts: number = 3): boolean {
    if (entryDraft.value.length < minPts) return false;
    entryPolygons.value = [...entryPolygons.value, entryDraft.value];
    entryDraft.value = [];
    return true;
  }
  function entryDelete(idx: number): void {
    if (idx < 0 || idx >= entryPolygons.value.length) return;
    entryPolygons.value = entryPolygons.value.filter((_, i) => i !== idx);
  }
  function entryClear(): void {
    entryPolygons.value = [];
    entryDraft.value = [];
  }

  // --- Step 4: POD points ----------------------------------------------
  function podAdd(p: Point): void { podPoints.value = [...podPoints.value, p]; }
  function podUndo(): void { podPoints.value = podPoints.value.slice(0, -1); }
  function podMove(idx: number, p: Point): void {
    if (idx < 0 || idx >= podPoints.value.length) return;
    podPoints.value[idx] = p;
    podPoints.value = [...podPoints.value];
  }
  function podDelete(idx: number): void {
    if (idx < 0 || idx >= podPoints.value.length) return;
    podPoints.value = podPoints.value.filter((_, i) => i !== idx);
  }
  function podClear(): void { podPoints.value = []; }

  // --- Step 5: room boundary -------------------------------------------
  function boundaryAdd(p: Point): void {
    if (boundaryConfirmed.value) return;
    boundary.value = [...boundary.value, p];
  }
  function boundaryUndo(): void {
    if (boundaryConfirmed.value) return;
    boundary.value = boundary.value.slice(0, -1);
  }
  function boundaryMove(idx: number, p: Point): void {
    if (boundaryConfirmed.value) return;
    if (idx < 0 || idx >= boundary.value.length) return;
    boundary.value[idx] = p;
    boundary.value = [...boundary.value];
  }
  function boundaryConfirm(minPts: number = 4): boolean {
    if (boundary.value.length < minPts) return false;
    boundaryConfirmed.value = true;
    return true;
  }
  function boundaryClear(): void {
    boundary.value = [];
    boundaryConfirmed.value = false;
  }

  // --- Whole-store reset (used by "Start Over") ------------------------
  function reset(): void {
    pairs.value = [];
    expectedClick.value = 'camera';
    entryPolygons.value = [];
    entryDraft.value = [];
    podPoints.value = [];
    boundary.value = [];
    boundaryConfirmed.value = false;
  }

  return {
    // state
    pairs, expectedClick,
    entryPolygons, entryDraft,
    podPoints,
    boundary, boundaryConfirmed,
    // step 2 actions
    appendCameraPoint, appendMapPoint, undoPair, deletePair, clearPairs,
    moveCameraPoint, moveMapPoint,
    // step 3
    entryAddPoint, entryUndo, entryMoveDraftPoint, entryConfirm, entryDelete, entryClear,
    // step 4
    podAdd, podUndo, podMove, podDelete, podClear,
    // step 5
    boundaryAdd, boundaryUndo, boundaryMove, boundaryConfirm, boundaryClear,
    // whole-store
    reset,
  };
});
