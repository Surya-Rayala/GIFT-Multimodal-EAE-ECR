<template>
  <div class="canvas-wrap" ref="wrapRef">
    <header class="canvas-header">
      <span class="title">{{ title }}</span>
      <span class="hint muted small" v-if="hint">{{ hint }}</span>
      <button class="zoom-reset" @click="resetView" :disabled="!imageReady">
        Reset view
      </button>
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
      <div v-if="!imageReady" class="empty muted">
        {{ emptyText }}
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue';

import type { Point } from '@/types/models';

/**
 * MapperCanvas — interactive image canvas with pan/zoom/click/drag.
 *
 * Renders a single source image and an overlay layer driven entirely from
 * props (markers, polylines). Emits image-space coordinates on click +
 * drag, so parents work in pixel units, not screen units.
 */

const props = defineProps<{
  /** Backend-served image URL, or empty when nothing is loaded. */
  src: string;
  title: string;
  emptyText?: string;
  hint?: string;
  /** Disable adding new points (drag still works). */
  readonly?: boolean;
  /** Markers to render on top of the image. */
  markers?: Array<{
    x: number;
    y: number;
    label?: string;
    color: string;
    /** When provided, this marker is draggable and emits `pointMoved` with this id. */
    handleId?: string;
  }>;
  /** Polylines/polygons to render. */
  polylines?: Array<{
    points: Point[];
    closed: boolean;
    color: string;
    width?: number;
  }>;
}>();

const emit = defineEmits<{
  /** User clicked at an image-space coordinate (no drag). */
  (e: 'pointClicked', x: number, y: number): void;
  /** User dragged a known handle to a new image-space coordinate. */
  (e: 'pointMoved', handleId: string, x: number, y: number): void;
}>();

const wrapRef = ref<HTMLElement>();
const hostRef = ref<HTMLElement>();
const canvasRef = ref<HTMLCanvasElement>();

const imageReady = ref(false);
const imgEl = new Image();
imgEl.crossOrigin = '';   // /image endpoint serves locally; no CORS issue

// View transform — translation in pixels (canvas-screen space) + scale.
const viewX = ref(0);
const viewY = ref(0);
const viewScale = ref(1);

// User-interaction state.
const panning = ref(false);
const panStart = ref<{ sx: number; sy: number; vx: number; vy: number } | null>(null);
const dragHandleId = ref<string | null>(null);
const downAtScreen = ref<{ x: number; y: number } | null>(null);

let resizeObserver: ResizeObserver | null = null;
let rafHandle: number | null = null;

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

function screenToImage(sx: number, sy: number): Point {
  return [
    (sx - viewX.value) / viewScale.value,
    (sy - viewY.value) / viewScale.value,
  ];
}
function imageToScreen(ix: number, iy: number): Point {
  return [
    ix * viewScale.value + viewX.value,
    iy * viewScale.value + viewY.value,
  ];
}
function clampToImage(x: number, y: number): Point {
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

// ----------------------------- event handlers --------------------------

function clientToCanvas(ev: MouseEvent): { x: number; y: number } {
  const canvas = canvasRef.value!;
  const rect = canvas.getBoundingClientRect();
  return {
    x: ev.clientX - rect.left,
    y: ev.clientY - rect.top,
  };
}

function hitTestHandle(sx: number, sy: number): string | null {
  // Hit-test in screen space so the click radius is constant regardless of zoom.
  const HIT_R = 12;
  for (const m of props.markers || []) {
    if (!m.handleId) continue;
    const [ix, iy] = imageToScreen(m.x, m.y);
    const dx = ix - sx;
    const dy = iy - sy;
    if (dx * dx + dy * dy <= HIT_R * HIT_R) return m.handleId;
  }
  return null;
}

function onMouseDown(ev: MouseEvent): void {
  if (!imageReady.value) return;
  if (ev.button !== 0) return;
  const { x, y } = clientToCanvas(ev);
  downAtScreen.value = { x, y };

  const id = hitTestHandle(x, y);
  if (id) {
    dragHandleId.value = id;
    return;
  }

  // Start panning otherwise.
  panning.value = true;
  panStart.value = { sx: x, sy: y, vx: viewX.value, vy: viewY.value };
}

function onMouseMove(ev: MouseEvent): void {
  if (!imageReady.value) return;
  const { x, y } = clientToCanvas(ev);

  if (dragHandleId.value) {
    const [ix, iy] = screenToImage(x, y);
    const [cx, cy] = clampToImage(ix, iy);
    emit('pointMoved', dragHandleId.value, Math.round(cx), Math.round(cy));
    requestDraw();
    return;
  }

  if (panning.value && panStart.value) {
    viewX.value = panStart.value.vx + (x - panStart.value.sx);
    viewY.value = panStart.value.vy + (y - panStart.value.sy);
    requestDraw();
  }
}

function onMouseUp(ev: MouseEvent): void {
  if (!imageReady.value) {
    panning.value = false;
    dragHandleId.value = null;
    return;
  }
  if (ev.button !== 0) return;
  const { x, y } = clientToCanvas(ev);

  // Drag-end → no click event.
  if (dragHandleId.value) {
    dragHandleId.value = null;
    panning.value = false;
    panStart.value = null;
    downAtScreen.value = null;
    return;
  }

  // Click vs pan: tiny movement → click. Otherwise pan-end, no click.
  if (panning.value && downAtScreen.value) {
    const dx = x - downAtScreen.value.x;
    const dy = y - downAtScreen.value.y;
    const moved = Math.hypot(dx, dy);
    panning.value = false;
    panStart.value = null;
    if (moved <= 4 && !props.readonly) {
      const [ix, iy] = screenToImage(x, y);
      const [cx, cy] = clampToImage(ix, iy);
      emit('pointClicked', Math.round(cx), Math.round(cy));
    }
  }
  downAtScreen.value = null;
}

function onMouseLeave(): void {
  panning.value = false;
  panStart.value = null;
  dragHandleId.value = null;
  downAtScreen.value = null;
}

function onWheel(ev: WheelEvent): void {
  if (!imageReady.value) return;

  // Trackpad scroll (no ctrl) → pan. Ctrl + wheel → zoom centred on cursor.
  const { x, y } = clientToCanvas(ev);
  if (ev.ctrlKey || ev.metaKey) {
    const delta = -ev.deltaY;
    const factor = delta > 0 ? 1.08 : 1 / 1.08;
    const [ix, iy] = screenToImage(x, y);
    viewScale.value = Math.max(0.05, Math.min(40, viewScale.value * factor));
    // Keep the point under the cursor stationary in image space.
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

function draw(): void {
  const canvas = canvasRef.value;
  const host = hostRef.value;
  if (!canvas || !host) return;

  // Match canvas backing-store to host size for crisp drawing.
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

  // Draw image at the current pan/zoom
  ctx.save();
  ctx.translate(viewX.value, viewY.value);
  ctx.scale(viewScale.value, viewScale.value);
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(imgEl, 0, 0);
  ctx.restore();

  // Polylines (drawn in screen space using a transformed path).
  for (const poly of props.polylines || []) {
    if (!poly.points.length) continue;
    ctx.beginPath();
    poly.points.forEach((p, i) => {
      const [sx, sy] = imageToScreen(p[0], p[1]);
      if (i === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    });
    if (poly.closed && poly.points.length >= 3) ctx.closePath();
    ctx.strokeStyle = poly.color;
    ctx.lineWidth = poly.width ?? 2.5;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.stroke();
  }

  // Markers — constant screen-pixel radius.
  for (const m of props.markers || []) {
    const [sx, sy] = imageToScreen(m.x, m.y);
    ctx.fillStyle = m.color;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(sx, sy, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    if (m.label) {
      ctx.font = '600 12px ui-sans-serif, system-ui';
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 3;
      ctx.lineJoin = 'round';
      const lx = sx + 10;
      const ly = sy - 8;
      ctx.strokeText(m.label, lx, ly);
      ctx.fillText(m.label, lx, ly);
    }
  }
}

// ----------------------------- lifecycle -------------------------------

onMounted(() => {
  resizeObserver = new ResizeObserver(() => {
    if (imageReady.value) {
      // Soft refit: keep the current scale if user has zoomed in; otherwise
      // recentre. For a simple, predictable feel we always recentre on
      // window resize.
      fitView();
    }
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
  () => [props.markers, props.polylines],
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
.zoom-reset:hover:not(:disabled) {
  background: var(--color-border);
}
.zoom-reset:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

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
  cursor: grab;
}
.canvas-host:active canvas {
  cursor: grabbing;
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
.muted {
  color: var(--color-muted);
}
.small {
  font-size: 0.85em;
}
</style>
