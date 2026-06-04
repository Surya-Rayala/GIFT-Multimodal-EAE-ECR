<template>
  <div ref="rootEl" class="timeline">
    <div class="timeline-body">
      <!-- Fixed-width label gutter — sits beside the timeline canvas, not on
           top of it. Vertical scroll is mirrored from the canvas container
           so labels always line up with their metric rows; horizontal scroll
           leaves the gutter untouched. -->
      <div class="label-column">
        <div ref="labelRailEl" class="label-rail">
          <div
            v-for="m in metricLabels"
            :key="m.metricId"
            class="metric-label"
            :style="{ top: `${m.startY + m.height / 2}px` }"
          >
            <span class="swatch" :style="{ background: m.color }"></span>
            <span class="name">{{ m.name }}</span>
          </div>
        </div>
      </div>
      <div ref="canvasContainer" class="canvas-container always-scroll"></div>
    </div>
    <div
      v-if="tooltip"
      class="tooltip"
      :class="{ below: tooltip.below }"
      :style="{ left: `${tooltip.x}px`, top: `${tooltip.y}px` }"
    >
      {{ tooltip.label }}
    </div>
    <div class="bar">
      <div class="zoom-controls">
        <button class="zoom-btn" title="Jump to first artifact" @click="onJumpToFirst">
          <span aria-hidden="true">⇤</span>
        </button>
        <button class="zoom-btn" title="Zoom out" @click="onZoomOut">−</button>
        <button class="zoom-btn" title="Fit to range" @click="onFit">Fit</button>
        <button class="zoom-btn" title="Zoom in" @click="onZoomIn">+</button>
      </div>
      <span class="hint muted">Swipe, drag, or scroll to navigate · tap an empty track to seek</span>
      <span class="range-readout muted">{{ rangeLabel }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { useSessionStore } from '@/stores/session';
import { usePlaybackStore } from '@/stores/playback';
import { useUIStore } from '@/stores/ui';
import type { TimelineItem } from '@/types/models';
import {
  TimelineScene,
  type HoverPayload,
  type MetricLabelInfo,
} from './TimelineScene';

const session = useSessionStore();
const playback = usePlaybackStore();
const ui = useUIStore();

const rootEl = ref<HTMLDivElement | null>(null);
const canvasContainer = ref<HTMLDivElement | null>(null);
const labelRailEl = ref<HTMLDivElement | null>(null);
let scene: TimelineScene | null = null;

const tooltip = ref<{ x: number; y: number; label: string; below: boolean } | null>(null);
const metricLabels = ref<MetricLabelInfo[]>([]);

// Visible range follows the active video mode.
const visibleRange = computed(() => {
  if (!session.session) return { start: 0, end: 1 };
  const total = session.session.video.total_frames;
  if (playback.mode === 'original') return { start: 0, end: total };
  const dw = session.session.drill_window;
  if (!dw) return { start: 0, end: total };
  return { start: dw.start_frame, end: dw.end_frame };
});

const rangeLabel = computed(() => {
  const { start, end } = visibleRange.value;
  return `frames ${start}–${end}`;
});

onMounted(async () => {
  if (!canvasContainer.value) return;
  scene = new TimelineScene();
  await scene.init(canvasContainer.value, {
    onItemClicked: (itemId, seekFrame) => {
      ui.selectedItemId = itemId;
      ui.selectedFlagId = null;
      ui.selectedMetricId = null;
      playback.requestSeek(seekFrame);
    },
    onFlagClicked: (flagId) => {
      ui.selectedFlagId = flagId;
      ui.selectedItemId = null;
      ui.selectedMetricId = null;
      const flag = session.flagById.get(flagId);
      if (flag?.frame != null) playback.requestSeek(flag.frame);
    },
    onEmptyClicked: (frame) => {
      ui.selectedItemId = null;
      ui.selectedFlagId = null;
      playback.requestSeek(frame);
    },
    onHover: (payload) => updateTooltip(payload),
    onLayoutChanged: (labels) => {
      metricLabels.value = labels;
    },
  });

  // The label gutter is a sibling of the canvas container, not a child.
  // Mirror the canvas's vertical scroll into the rail so labels always line
  // up with their metric rows. Horizontal scroll is independent: the gutter
  // never moves sideways, so labels can't overlap canvas content.
  canvasContainer.value.addEventListener('scroll', onCanvasScroll, { passive: true });

  pushData();
  scene.setRange(visibleRange.value.start, visibleRange.value.end);
  scene.setMetricVisibility(ui.metricVisibility);
  scene.setSelectedMetric(ui.selectedMetricId);
  scene.setSelectedFlag(ui.selectedFlagId);
  scene.setCurrentFrame(playback.currentFrame);
});

function onCanvasScroll(): void {
  if (!canvasContainer.value || !labelRailEl.value) return;
  labelRailEl.value.style.transform = `translateY(${-canvasContainer.value.scrollTop}px)`;
}

function pushData(): void {
  if (!scene) return;
  if (!session.session) {
    scene.setData({ items: [], flags: [], drillWindow: null, totalFrames: 1 });
    return;
  }
  const dw = session.session.drill_window;
  scene.setData({
    items: session.session.timeline.items,
    flags: session.session.flags,
    drillWindow: dw
      ? {
          start_frame: dw.start_frame,
          end_frame: dw.end_frame,
          end_uncertain: dw.end_uncertain,
        }
      : null,
    totalFrames: session.session.video.total_frames,
  });
}

// Threshold (px from top of timeline div) below which the tooltip flips
// below the cursor instead of above it. Picked to comfortably clear the
// flag-row + tooltip height (~30 px).
const TOOLTIP_FLIP_THRESHOLD = 56;

function updateTooltip(payload: HoverPayload): void {
  if (!payload || !rootEl.value) {
    tooltip.value = null;
    return;
  }
  const label =
    payload.kind === 'item'
      ? itemTooltipLabel(session.itemById.get(payload.id))
      : payload.kind === 'flag'
        ? flagTooltipLabel(payload.id)
        : drillTooltipLabel(payload.id);
  if (!label) {
    tooltip.value = null;
    return;
  }
  // Position relative to the timeline root so the tooltip follows the cursor
  // regardless of the canvas-container's scroll offset. Auto-flip below the
  // cursor when hovering near the top so the tooltip never escapes upward.
  const rect = rootEl.value.getBoundingClientRect();
  const x = payload.clientX - rect.left;
  const y = payload.clientY - rect.top;
  tooltip.value = {
    x,
    y,
    label,
    below: y < TOOLTIP_FLIP_THRESHOLD,
  };
}

function itemTooltipLabel(item: TimelineItem | undefined): string | null {
  if (!item) return null;
  if (item.kind === 'entry') {
    return `${item.label} · track ${item.data.track_id} · ${item.time_sec.toFixed(2)}s`;
  }
  if (item.kind === 'vector') {
    return `${item.label} · ${item.data.direction_label} · ${item.time_sec.toFixed(2)}s`;
  }
  if (item.kind === 'pair_gap') {
    const tag = item.data.violates_time_limit ? ' · ⚠ violation' : '';
    return `${item.label} · gap ${item.data.gap_sec.toFixed(2)}s / allowed ${item.data.allowed_gap_sec.toFixed(2)}s${tag}`;
  }
  if (item.kind === 'duration') {
    const allowed = item.data.derived_allowed_duration_sec;
    const allowedStr = typeof allowed === 'number' ? allowed.toFixed(2) : '—';
    const tag = item.data.violates_total_entry_limit ? ' · ⚠ violation' : '';
    return `${item.label} · ${item.data.duration_sec.toFixed(2)}s / allowed ${allowedStr}s${tag}`;
  }
  if (item.kind === 'wall_excursion') {
    const human = item.data.label_kind === 'too_close' ? 'too close' : 'too far';
    return `${item.label} · ${human} for ${item.data.duration_sec.toFixed(2)}s · ${item.time_sec.toFixed(2)}s`;
  }
  return null;
}

function flagTooltipLabel(flagId: string): string | null {
  const flag = session.flagById.get(flagId);
  if (!flag) return null;
  return `${flag.title} (${flag.severity})`;
}

function drillTooltipLabel(id: string): string | null {
  const dw = session.session?.drill_window;
  if (!dw) return null;
  if (id === 'drill_start') {
    return `Drill start · frame ${dw.start_frame} · ${dw.start_time_sec.toFixed(2)}s`;
  }
  if (id === 'drill_end') {
    const tag = dw.end_uncertain ? ' (uncertain)' : '';
    return `Drill end · frame ${dw.end_frame} · ${dw.end_time_sec.toFixed(2)}s${tag}`;
  }
  return null;
}

watch(() => session.session, pushData);
watch(visibleRange, (range) => {
  scene?.setRange(range.start, range.end);
});
watch(
  () => ({ ...ui.metricVisibility }),
  (vis) => scene?.setMetricVisibility(vis),
  { deep: true },
);
watch(
  () => ui.selectedMetricId,
  (id) => {
    scene?.setSelectedMetric(id);
    if (id) scene?.scrollToMetric(id);
  },
);
watch(
  () => ui.selectedFlagId,
  (id) => {
    if (id) {
      // Force the flag's owning metric visible — otherwise the flag (and the
      // row that hosts it) isn't in the layout, and the auto-scroll below
      // would land on nothing. Matches user intent: picking a flag is an
      // implicit "show me this".
      const flag = session.flagById.get(id);
      if (flag?.metric_id && !ui.isMetricVisible(flag.metric_id)) {
        ui.setMetricVisibility(flag.metric_id, true);
      }
    }
    scene?.setSelectedFlag(id);
    if (id) scene?.scrollToFlag(id);
  },
);
watch(
  () => playback.currentFrame,
  (f) => scene?.setCurrentFrame(f),
);

function onZoomIn(): void {
  scene?.zoomBy(1.4);
}
function onZoomOut(): void {
  scene?.zoomBy(1 / 1.4);
}
function onFit(): void {
  scene?.fit();
}
function onJumpToFirst(): void {
  if (!scene) return;
  const f = scene.firstArtifactFrame();
  if (f == null) return;
  playback.requestSeek(f);
  scene.scrollToFrame(f, 'start');
}

onBeforeUnmount(() => {
  canvasContainer.value?.removeEventListener('scroll', onCanvasScroll);
  scene?.destroy();
  scene = null;
});
</script>

<style scoped>
.timeline {
  position: relative;
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--color-bg-elev);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  overflow: hidden;
}

/* Body holds the label gutter and the canvas side-by-side so the gutter
   never overlaps canvas content. */
.timeline-body {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: row;
  overflow: hidden;
}

.canvas-container {
  flex: 1;
  min-width: 0;
  min-height: 0;
  /* The actual scrollbar styling lives in theme.css under .always-scroll —
   * applied via class on the template above so the ::-webkit-scrollbar
   * pseudo-elements bind reliably (no Vue scoped-css dance). */
}

/* Fixed-width left gutter for metric labels. Vertical scroll is mirrored
   from the canvas container via JS; horizontal scroll never affects this
   column, so labels can't overlap timeline content. */
.label-column {
  flex: 0 0 130px;
  position: relative;
  overflow: hidden;
  border-right: 1px solid var(--color-border);
  background: var(--color-bg-elev);
}
.label-rail {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  /* `transform: translateY` is set imperatively in onCanvasScroll. */
  will-change: transform;
}
.metric-label {
  position: absolute;
  left: 6px;
  right: 6px;
  /* Vertically centred on its metric region — `top` is set to the region's
     midpoint, transform shifts the label up by half its own height. */
  transform: translateY(-50%);
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 2px 6px;
  background: transparent;
  border: 0;
  border-radius: 3px;
  font-size: 0.72em;
  color: var(--color-text);
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.4;
  font-variant-numeric: tabular-nums;
}
.metric-label .swatch {
  width: 8px;
  height: 8px;
  border-radius: 1.5px;
  display: inline-block;
  flex-shrink: 0;
}
.metric-label .name {
  letter-spacing: 0.01em;
}

.tooltip {
  position: absolute;
  background: rgba(15, 17, 21, 0.96);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  padding: 4px 8px;
  font-size: 0.78em;
  pointer-events: none;
  white-space: nowrap;
  /* Default: appear above the cursor */
  transform: translate(-50%, calc(-100% - 10px));
  z-index: 5;
  font-variant-numeric: tabular-nums;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
}
/* Flip below the cursor when hovering near the top of the timeline so the
   tooltip never escapes the visible area. */
.tooltip.below {
  transform: translate(-50%, 14px);
}

.bar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 4px 10px;
  border-top: 1px solid var(--color-border);
  background: var(--color-bg);
  font-size: 0.75em;
}
.zoom-controls {
  display: inline-flex;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  overflow: hidden;
}
.zoom-btn {
  background: transparent;
  color: var(--color-text);
  border: none;
  border-right: 1px solid var(--color-border);
  padding: 2px 10px;
  font-size: 1em;
  cursor: pointer;
  min-width: 28px;
  font-variant-numeric: tabular-nums;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}
.zoom-btn:last-child {
  border-right: none;
}
.zoom-btn:hover,
.zoom-btn:active {
  background: var(--color-border);
}
.hint {
  flex: 1;
}

/* Touch: enlarge the zoom/jump buttons to comfortable tap targets and let the
 * bottom bar wrap instead of clipping the hint on a narrow screen. */
@media (hover: none) {
  .zoom-btn {
    min-width: 42px;
    padding: 9px 12px;
  }
  .bar {
    flex-wrap: wrap;
    row-gap: 4px;
  }
}
.range-readout {
  font-variant-numeric: tabular-nums;
}
.muted {
  color: var(--color-muted);
}
</style>
