<template>
  <div class="canvas-wrap">
    <header class="canvas-header">
      <span class="title">{{ title }}</span>
      <span class="hint muted small" v-if="hint">{{ hint }}</span>
      <button class="zoom-reset" @click="resetView" :disabled="!imageReady">Reset view</button>
    </header>
    <div class="canvas-host" ref="hostRef">
      <canvas
        ref="canvasRef"
        @mousedown="onMouseDown"
        @mousemove="onMouseMove"
        @mouseup="onMouseUp"
        @mouseleave="onMouseLeave"
        @wheel.prevent="onWheel"
      />
      <div v-if="!imageReady" class="empty muted">{{ emptyText }}</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from 'vue';

import type { Vec2 } from '@/utils/scaling';

interface WallShape {
  id: number;
  p1: Vec2;
  p2: Vec2;
  color: string;
  label?: string;
  width?: number;
}
interface PointShape {
  x: number;
  y: number;
  color: string;
  label?: string;
}

const props = defineProps<{
  src: string;
  title: string;
  emptyText?: string;
  hint?: string;
  mode: 'walls' | 'points';
  walls?: WallShape[];
  points?: PointShape[];
  ghost?: { x: number; y: number } | null;
  corner?: Vec2 | null;
  pendingStart?: Vec2 | null;
  /** Endpoints the next wall vertex can snap to. */
  snapPoints?: Vec2[];
}>();

const emit = defineEmits<{
  (e: 'vertexClicked', x: number, y: number): void;
  (e: 'wallPicked', id: number): void;
}>();

const hostRef = ref<HTMLElement>();
const canvasRef = ref<HTMLCanvasElement>();

const imageReady = ref(false);
const imgEl = new Image();
imgEl.crossOrigin = '';

const viewX = ref(0);
const viewY = ref(0);
const viewScale = ref(1);

const panning = ref(false);
const panStart = ref<{ sx: number; sy: number; vx: number; vy: number } | null>(null);
const downAtScreen = ref<{ x: number; y: number } | null>(null);
const cursorImg = ref<Vec2 | null>(null);

let resizeObserver: ResizeObserver | null = null;
let rafHandle: number | null = null;

const SNAP_EPS_SCREEN = 10;
const WALL_HIT_SCREEN = 8;

// ----------------------------- image loading ---------------------------
watch(
  () => props.src,
  (url) => {
    if (!url) {
      imageReady.value = false;
      requestDraw();
      return;
    }
    imageReady.value = false;
    imgEl.onload = () => {
      imageReady.value = true;
      fitView();
      requestDraw();
    };
    imgEl.onerror = () => {
      imageReady.value = false;
      requestDraw();
    };
    imgEl.src = url;
  },
  { immediate: true },
);

// ----------------------------- transforms ------------------------------
function screenToImage(sx: number, sy: number): Vec2 {
  return [(sx - viewX.value) / viewScale.value, (sy - viewY.value) / viewScale.value];
}
function imageToScreen(ix: number, iy: number): Vec2 {
  return [ix * viewScale.value + viewX.value, iy * viewScale.value + viewY.value];
}
function clampToImage(x: number, y: number): Vec2 {
  if (!imageReady.value) return [x, y];
  return [
    Math.max(0, Math.min(imgEl.naturalWidth - 1, x)),
    Math.max(0, Math.min(imgEl.naturalHeight - 1, y)),
  ];
}

function fitView(): void {
  const host = hostRef.value;
  const canvas = canvasRef.value;
  if (!host || !canvas || !imageReady.value) return;
  const ww = host.clientWidth;
  const hh = host.clientHeight;
  if (ww <= 0 || hh <= 0) return;
  const iw = imgEl.naturalWidth;
  const ih = imgEl.naturalHeight;
  const s = Math.min(ww / iw, hh / ih);
  viewScale.value = s;
  viewX.value = (ww - iw * s) / 2;
  viewY.value = (hh - ih * s) / 2;
}
function resetView(): void {
  fitView();
  requestDraw();
}

// ----------------------------- snap / hit-test -------------------------
function snap(ix: number, iy: number): Vec2 {
  const [sx, sy] = imageToScreen(ix, iy);
  let best: { p: Vec2; d: number } | null = null;
  for (const sp of props.snapPoints || []) {
    const [px, py] = imageToScreen(sp[0], sp[1]);
    const d = Math.hypot(px - sx, py - sy);
    if (d <= SNAP_EPS_SCREEN && (best === null || d < best.d)) best = { p: sp, d };
  }
  return best ? best.p : [ix, iy];
}

function segDistScreen(px: number, py: number, a: Vec2, b: Vec2): number {
  const [ax, ay] = imageToScreen(a[0], a[1]);
  const [bx, by] = imageToScreen(b[0], b[1]);
  const dx = bx - ax;
  const dy = by - ay;
  const l2 = dx * dx + dy * dy;
  let t = l2 === 0 ? 0 : ((px - ax) * dx + (py - ay) * dy) / l2;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

function pickWall(sx: number, sy: number): number | null {
  let best: { id: number; d: number } | null = null;
  for (const w of props.walls || []) {
    const d = segDistScreen(sx, sy, w.p1, w.p2);
    if (d <= WALL_HIT_SCREEN && (best === null || d < best.d)) best = { id: w.id, d };
  }
  return best ? best.id : null;
}

// ----------------------------- events ----------------------------------
function clientToCanvas(ev: MouseEvent): { x: number; y: number } {
  const rect = canvasRef.value!.getBoundingClientRect();
  return { x: ev.clientX - rect.left, y: ev.clientY - rect.top };
}

function onMouseDown(ev: MouseEvent): void {
  if (!imageReady.value || ev.button !== 0) return;
  const { x, y } = clientToCanvas(ev);
  downAtScreen.value = { x, y };
  panning.value = true;
  panStart.value = { sx: x, sy: y, vx: viewX.value, vy: viewY.value };
}

function onMouseMove(ev: MouseEvent): void {
  if (!imageReady.value) return;
  const { x, y } = clientToCanvas(ev);
  cursorImg.value = screenToImage(x, y);

  if (panning.value && panStart.value) {
    viewX.value = panStart.value.vx + (x - panStart.value.sx);
    viewY.value = panStart.value.vy + (y - panStart.value.sy);
  }
  requestDraw();
}

function onMouseUp(ev: MouseEvent): void {
  if (!imageReady.value || ev.button !== 0) {
    panning.value = false;
    return;
  }
  const { x, y } = clientToCanvas(ev);
  const wasPan = panning.value;
  panning.value = false;
  panStart.value = null;

  if (wasPan && downAtScreen.value) {
    const moved = Math.hypot(x - downAtScreen.value.x, y - downAtScreen.value.y);
    if (moved <= 4) {
      // It's a click, not a drag.
      if (props.mode === 'walls') {
        const [ix, iy] = clampToImage(...screenToImage(x, y));
        const [snx, sny] = snap(ix, iy);
        emit('vertexClicked', Math.round(snx), Math.round(sny));
      } else {
        const id = pickWall(x, y);
        if (id != null) emit('wallPicked', id);
      }
    }
  }
  downAtScreen.value = null;
  requestDraw();
}

function onMouseLeave(): void {
  panning.value = false;
  panStart.value = null;
  downAtScreen.value = null;
  cursorImg.value = null;
  requestDraw();
}

function onWheel(ev: WheelEvent): void {
  if (!imageReady.value) return;
  const { x, y } = clientToCanvas(ev);
  if (ev.ctrlKey || ev.metaKey) {
    const factor = -ev.deltaY > 0 ? 1.08 : 1 / 1.08;
    const [ix, iy] = screenToImage(x, y);
    viewScale.value = Math.max(0.05, Math.min(40, viewScale.value * factor));
    viewX.value = x - ix * viewScale.value;
    viewY.value = y - iy * viewScale.value;
  } else {
    viewX.value -= ev.deltaX;
    viewY.value -= ev.deltaY;
  }
  requestDraw();
}

// ----------------------------- drawing ---------------------------------
function requestDraw(): void {
  if (rafHandle != null) return;
  rafHandle = requestAnimationFrame(() => {
    rafHandle = null;
    draw();
  });
}

function dot(ctx: CanvasRenderingContext2D, p: Vec2, r: number, fill: string, stroke = '#ffffff'): void {
  const [sx, sy] = imageToScreen(p[0], p[1]);
  ctx.beginPath();
  ctx.arc(sx, sy, r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = stroke;
  ctx.stroke();
}

function label(ctx: CanvasRenderingContext2D, text: string, sx: number, sy: number): void {
  ctx.font = '600 12px ui-sans-serif, system-ui';
  ctx.lineWidth = 3;
  ctx.lineJoin = 'round';
  ctx.strokeStyle = '#000000';
  ctx.fillStyle = '#ffffff';
  ctx.strokeText(text, sx, sy);
  ctx.fillText(text, sx, sy);
}

function draw(): void {
  const canvas = canvasRef.value;
  const host = hostRef.value;
  if (!canvas || !host) return;

  const dpr = window.devicePixelRatio || 1;
  const ww = host.clientWidth;
  const hh = host.clientHeight;
  const targetW = Math.floor(ww * dpr);
  const targetH = Math.floor(hh * dpr);
  if (canvas.width !== targetW || canvas.height !== targetH) {
    canvas.width = targetW;
    canvas.height = targetH;
    canvas.style.width = `${ww}px`;
    canvas.style.height = `${hh}px`;
  }

  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, ww, hh);
  if (!imageReady.value) return;

  ctx.save();
  ctx.translate(viewX.value, viewY.value);
  ctx.scale(viewScale.value, viewScale.value);
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(imgEl, 0, 0);
  ctx.restore();

  // Walls
  for (const w of props.walls || []) {
    const [ax, ay] = imageToScreen(w.p1[0], w.p1[1]);
    const [bx, by] = imageToScreen(w.p2[0], w.p2[1]);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.strokeStyle = w.color;
    ctx.lineWidth = w.width ?? 3;
    ctx.lineCap = 'round';
    ctx.stroke();
    // endpoint dots
    dot(ctx, w.p1, 3.5, w.color);
    dot(ctx, w.p2, 3.5, w.color);
    if (w.label) label(ctx, w.label, (ax + bx) / 2 + 6, (ay + by) / 2 - 6);
  }

  // Pending wall: first vertex + rubber band to cursor
  if (props.pendingStart) {
    dot(ctx, props.pendingStart, 4, '#06b6d4');
    if (cursorImg.value) {
      const [ax, ay] = imageToScreen(props.pendingStart[0], props.pendingStart[1]);
      const [bx, by] = imageToScreen(cursorImg.value[0], cursorImg.value[1]);
      ctx.beginPath();
      ctx.setLineDash([5, 4]);
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // Shared corner highlight
  if (props.corner) {
    const [cx, cy] = imageToScreen(props.corner[0], props.corner[1]);
    ctx.beginPath();
    ctx.arc(cx, cy, 9, 0, Math.PI * 2);
    ctx.strokeStyle = '#a855f7';
    ctx.lineWidth = 2.5;
    ctx.stroke();
  }

  // Committed points
  for (const p of props.points || []) {
    dot(ctx, [p.x, p.y], 5, p.color);
    if (p.label) {
      const [sx, sy] = imageToScreen(p.x, p.y);
      label(ctx, p.label, sx + 9, sy - 8);
    }
  }

  // Ghost preview (dashed ring + center)
  if (props.ghost) {
    const [gx, gy] = imageToScreen(props.ghost.x, props.ghost.y);
    ctx.beginPath();
    ctx.setLineDash([4, 3]);
    ctx.arc(gx, gy, 8, 0, Math.PI * 2);
    ctx.strokeStyle = '#f97316';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.setLineDash([]);
    dot(ctx, [props.ghost.x, props.ghost.y], 3, '#f97316');
  }
}

// ----------------------------- lifecycle -------------------------------
onMounted(() => {
  resizeObserver = new ResizeObserver(() => {
    if (imageReady.value) fitView();
    requestDraw();
  });
  if (hostRef.value) resizeObserver.observe(hostRef.value);
  requestDraw();
});
onUnmounted(() => {
  resizeObserver?.disconnect();
  if (rafHandle != null) cancelAnimationFrame(rafHandle);
});

watch(
  () => [props.walls, props.points, props.ghost, props.corner, props.pendingStart, props.snapPoints],
  () => requestDraw(),
  { deep: true },
);

defineExpose({ resetView });
</script>

<style scoped>
.canvas-wrap {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-width: 0;
}
.canvas-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 0 6px;
}
.title {
  font-weight: 600;
  font-size: 0.95em;
  letter-spacing: 0.02em;
}
.hint {
  flex: 1;
  font-size: 0.85em;
}
.zoom-reset {
  background: var(--color-bg-elev);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  padding: 3px 10px;
  border-radius: 4px;
  font-size: 0.82em;
  cursor: pointer;
}
.zoom-reset:hover:not(:disabled) { background: var(--color-border); }
.zoom-reset:disabled { opacity: 0.5; cursor: not-allowed; }

.canvas-host {
  position: relative;
  flex: 1;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  background: var(--color-bg-deep);
  overflow: hidden;
  user-select: none;
  min-height: 280px;
}
.canvas-host canvas {
  display: block;
  cursor: crosshair;
}
.empty {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 16px;
  text-align: center;
  font-size: 0.9em;
  pointer-events: none;
}
.muted { color: var(--color-muted); }
.small { font-size: 0.85em; }
</style>
