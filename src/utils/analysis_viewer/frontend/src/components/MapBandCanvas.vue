<template>
  <!--
    Always-mounted host so the parent .video-cell.aux exists before we
    look up the sibling <video> element. The <canvas> inside is sized to
    the cell's CSS box (so it lines up 1:1 with the displayed video) and
    drawn imperatively whenever inputs change. Pointer-events disabled
    so the canvas never steals scrub clicks.
  -->
  <div ref="hostEl" class="map-band-host" aria-hidden="true">
    <canvas ref="canvasEl" class="map-band-canvas" />
  </div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { useSessionStore } from '@/stores/session';
import { useUIStore } from '@/stores/ui';
import { usePlaybackStore } from '@/stores/playback';
import { colorForTrackId } from '@/theme/tokens';
import type {
  TimelineItem,
  WallExcursionTimelineItem,
  MoveAlongWallEntrant,
} from '@/types/models';

const session = useSessionStore();
const ui = useUIStore();
const playback = usePlaybackStore();

const hostEl = ref<HTMLDivElement | null>(null);
const canvasEl = ref<HTMLCanvasElement | null>(null);

let videoEl: HTMLVideoElement | null = null;
let metaListener: (() => void) | null = null;
let resizeObserver: ResizeObserver | null = null;

// ---- DOM glue: locate the sibling aux <video> ----------------------

function findVideo(): HTMLVideoElement | null {
  const host = hostEl.value;
  if (!host) return null;
  const cell = host.closest('.video-cell');
  return (cell?.querySelector('video') as HTMLVideoElement) ?? null;
}

function attachToVideo(): void {
  // Attempt counter is closure-local: every call to attachToVideo gets
  // its own retry budget, with no risk of cross-call interference.
  let attempts = 0;
  const tryOnce = (): void => {
    attempts += 1;
    const next = findVideo();
    if (!next) {
      // Sibling video may still be mounting (mode switch). Bounded retry
      // across animation frames keeps us from busy-looping forever.
      if (attempts < 12) requestAnimationFrame(tryOnce);
      return;
    }
    if (videoEl === next) {
      // Same element, but its metadata or display size may have changed.
      ensureCanvasMatchesVideo();
      draw();
      return;
    }
    detachFromVideo();
    videoEl = next;
    metaListener = (): void => {
      ensureCanvasMatchesVideo();
      draw();
    };
    videoEl.addEventListener('loadedmetadata', metaListener);
    videoEl.addEventListener('emptied', metaListener);
    if (resizeObserver === null && typeof ResizeObserver !== 'undefined') {
      resizeObserver = new ResizeObserver(() => {
        ensureCanvasMatchesVideo();
        draw();
      });
    }
    if (resizeObserver) resizeObserver.observe(videoEl);
    ensureCanvasMatchesVideo();
    draw();
  };
  queueMicrotask(tryOnce);
}

function detachFromVideo(): void {
  if (videoEl && metaListener) {
    videoEl.removeEventListener('loadedmetadata', metaListener);
    videoEl.removeEventListener('emptied', metaListener);
  }
  if (resizeObserver && videoEl) resizeObserver.unobserve(videoEl);
  videoEl = null;
  metaListener = null;
}

// ---- Canvas sizing: 1:1 with the video element's CSS box -----------

function ensureCanvasMatchesVideo(): void {
  const c = canvasEl.value;
  const v = videoEl;
  if (!c || !v) return;
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const cssW = Math.max(1, Math.floor(v.clientWidth));
  const cssH = Math.max(1, Math.floor(v.clientHeight));
  // CSS box and pixel buffer separately. Drawing buffer scales by DPR
  // so the band edges stay crisp on hi-density displays.
  if (c.style.width !== `${cssW}px`) c.style.width = `${cssW}px`;
  if (c.style.height !== `${cssH}px`) c.style.height = `${cssH}px`;
  const bufW = Math.max(1, Math.round(cssW * dpr));
  const bufH = Math.max(1, Math.round(cssH * dpr));
  if (c.width !== bufW) c.width = bufW;
  if (c.height !== bufH) c.height = bufH;
}

// ---- Selection chain: flag → wall_excursion item → entrant ----------

function selectedWallEntrant(): MoveAlongWallEntrant | null {
  const flagId = ui.selectedFlagId;
  if (!flagId) return null;
  const flag = session.flagById.get(flagId);
  if (!flag || flag.metric_id !== 'move_along_wall') return null;
  if (!flag.linked_item_id) return null;
  const item: TimelineItem | undefined = session.itemById.get(flag.linked_item_id);
  if (!item || item.kind !== 'wall_excursion') return null;
  const excursion = item as WallExcursionTimelineItem;
  const metric = session.metricById.get('move_along_wall');
  if (!metric || metric.metric_id !== 'move_along_wall') return null;
  const list = metric.summary.per_entrant ?? [];
  const trackId = excursion.data.track_id;
  return list.find((e) => e.track_id === trackId) ?? null;
}

function sessionBoundary(): Array<[number, number]> | null {
  const b = session.session?.boundary;
  if (!b || b.length < 3) return null;
  return b;
}

// ---- Coordinate transform: intrinsic video px → CSS canvas px ------
//
// The video uses object-fit: contain — so its drawn pixels occupy a
// centered sub-rect of the element's CSS box, letterboxed if the box
// aspect ratio differs from the video's intrinsic aspect ratio.
// We compute that sub-rect and map polygon points (which live in the
// video's intrinsic pixel space) into it.

function buildTransform(): { scale: number; offX: number; offY: number } | null {
  const v = videoEl;
  const c = canvasEl.value;
  if (!v || !c) return null;
  const vw = v.videoWidth;
  const vh = v.videoHeight;
  const cssW = v.clientWidth;
  const cssH = v.clientHeight;
  if (vw <= 0 || vh <= 0 || cssW <= 0 || cssH <= 0) return null;
  const aspectV = vw / vh;
  const aspectC = cssW / cssH;
  let scale: number;
  let offX = 0;
  let offY = 0;
  if (aspectV > aspectC) {
    // Video wider than container: letterbox top + bottom.
    scale = cssW / vw;
    offY = (cssH - vh * scale) / 2;
  } else {
    // Video taller (or equal): pillarbox left + right.
    scale = cssH / vh;
    offX = (cssW - vw * scale) / 2;
  }
  return { scale, offX, offY };
}

// ---- Draw ----------------------------------------------------------

function draw(): void {
  const c = canvasEl.value;
  if (!c) return;
  const ctx = c.getContext('2d');
  if (!ctx) return;

  // Reset transform + clear regardless of whether we'll redraw, so the
  // canvas blanks out when the user clears the selection or switches
  // out of motion mode.
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, c.width / dpr, c.height / dpr);

  // Visibility gates. The overlay is restricted to ``motion`` mode —
  // gaze-mode artifact videos already have cones / trails baked in, so
  // adding a band annulus would compete visually.
  if (playback.mode !== 'motion') return;
  const entrant = selectedWallEntrant();
  if (!entrant) return;
  const inner = entrant.band_inner_polygon;
  if (!inner || inner.length < 3) return;
  const outer = sessionBoundary();
  if (!outer) return;
  const t = buildTransform();
  if (!t) return;

  // Build the safe-band path: outer minus inner using even-odd fill.
  const path = new Path2D();
  path.moveTo(outer[0][0] * t.scale + t.offX, outer[0][1] * t.scale + t.offY);
  for (let i = 1; i < outer.length; i++) {
    path.lineTo(outer[i][0] * t.scale + t.offX, outer[i][1] * t.scale + t.offY);
  }
  path.closePath();
  path.moveTo(inner[0][0] * t.scale + t.offX, inner[0][1] * t.scale + t.offY);
  for (let i = 1; i < inner.length; i++) {
    path.lineTo(inner[i][0] * t.scale + t.offX, inner[i][1] * t.scale + t.offY);
  }
  path.closePath();

  const color = colorForTrackId(entrant.track_id);
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.30;
  ctx.fill(path, 'evenodd');

  ctx.globalAlpha = 0.85;
  ctx.lineWidth = 2;
  ctx.strokeStyle = color;
  ctx.stroke(path);
  ctx.globalAlpha = 1.0;
}

// ---- Lifecycle + reactivity ----------------------------------------

onMounted(() => {
  attachToVideo();
});

onBeforeUnmount(() => {
  detachFromVideo();
  if (resizeObserver) {
    resizeObserver.disconnect();
    resizeObserver = null;
  }
});

// Re-attach when the aux <video> is replaced (mode change re-keys it)
// and on session change. Then redraw.
watch(() => playback.mode, () => {
  // When the new aux video element shows up, attachToVideo() observes
  // it and triggers draw() once intrinsic dims arrive.
  attachToVideo();
});
watch(() => session.session, () => {
  attachToVideo();
});

// Selection / data changes: just redraw using the existing video ref.
watch(
  () => ui.selectedFlagId,
  () => draw(),
  { flush: 'post' },
);
</script>

<style scoped>
.map-band-host {
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 4;
}
.map-band-canvas {
  position: absolute;
  inset: 0;
  pointer-events: none;
}
</style>
