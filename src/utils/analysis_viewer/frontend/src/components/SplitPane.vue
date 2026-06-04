<template>
  <div class="split-pane" :style="gridStyle">
    <div class="pane left"><slot name="left" /></div>
    <div
      class="resizer"
      :class="{ active: activeSide === 'left' }"
      @pointerdown="onPointerDown('left', $event)"
    ></div>
    <div class="pane center"><slot name="center" /></div>
    <div
      v-if="!ui.rightCollapsed"
      class="resizer"
      :class="{ active: activeSide === 'right' }"
      @pointerdown="onPointerDown('right', $event)"
    ></div>
    <div v-else class="resizer collapsed-spacer"></div>
    <div class="pane right" :class="{ collapsed: ui.rightCollapsed }">
      <button
        class="collapse-toggle"
        :title="ui.rightCollapsed ? 'Show details panel' : 'Hide details panel'"
        :aria-label="ui.rightCollapsed ? 'Show details panel' : 'Hide details panel'"
        :aria-expanded="!ui.rightCollapsed"
        @click="ui.toggleRightCollapsed"
      >
        <svg
          class="chevron"
          viewBox="0 0 12 12"
          width="10"
          height="10"
          aria-hidden="true"
        >
          <polyline
            :points="ui.rightCollapsed ? '7.5,2.5 3.5,6 7.5,9.5' : '4.5,2.5 8.5,6 4.5,9.5'"
            fill="none"
            stroke="currentColor"
            stroke-width="1.6"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
        </svg>
      </button>
      <div v-show="!ui.rightCollapsed" class="pane-content">
        <slot name="right" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue';
import { useUIStore } from '@/stores/ui';

const ui = useUIStore();

const MIN_LEFT = 200;
const MAX_LEFT = 600;
const MIN_RIGHT = 240;
const MAX_RIGHT = 700;
const MIN_CENTER = 320;
const COLLAPSED_RIGHT = 22;

const activeSide = ref<'left' | 'right' | null>(null);

const gridStyle = computed(() => {
  const rightWidth = ui.rightCollapsed ? COLLAPSED_RIGHT : ui.splitterRight;
  const rightResizer = ui.rightCollapsed ? 0 : 5;
  return {
    gridTemplateColumns: `${ui.splitterLeft}px 5px minmax(${MIN_CENTER}px, 1fr) ${rightResizer}px ${rightWidth}px`,
  };
});

let startX = 0;
let startSize = 0;

function onPointerDown(side: 'left' | 'right', ev: PointerEvent): void {
  if (side === 'right' && ui.rightCollapsed) return;
  activeSide.value = side;
  startX = ev.clientX;
  startSize = side === 'left' ? ui.splitterLeft : ui.splitterRight;
  (ev.currentTarget as HTMLElement).setPointerCapture(ev.pointerId);
  window.addEventListener('pointermove', onPointerMove);
  window.addEventListener('pointerup', onPointerUp);
  window.addEventListener('pointercancel', onPointerUp);
}

function onPointerMove(ev: PointerEvent): void {
  if (!activeSide.value) return;
  const dx = ev.clientX - startX;
  const winW = window.innerWidth;
  if (activeSide.value === 'left') {
    const rightCol = ui.rightCollapsed ? COLLAPSED_RIGHT : ui.splitterRight;
    const maxAllowed = Math.min(MAX_LEFT, winW - MIN_CENTER - rightCol - 10);
    ui.splitterLeft = clamp(startSize + dx, MIN_LEFT, maxAllowed);
  } else {
    const maxAllowed = Math.min(MAX_RIGHT, winW - MIN_CENTER - ui.splitterLeft - 10);
    ui.splitterRight = clamp(startSize - dx, MIN_RIGHT, maxAllowed);
  }
}

function onPointerUp(): void {
  activeSide.value = null;
  window.removeEventListener('pointermove', onPointerMove);
  window.removeEventListener('pointerup', onPointerUp);
  window.removeEventListener('pointercancel', onPointerUp);
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

onBeforeUnmount(onPointerUp);
</script>

<style scoped>
.split-pane {
  display: grid;
  height: 100%;
  width: 100%;
}
.pane {
  overflow: auto;
  min-width: 0;
  background: var(--color-bg-elev);
}
.pane.center {
  background: var(--color-bg);
}
.resizer {
  background: var(--color-border);
  cursor: col-resize;
  user-select: none;
  transition: background-color 120ms ease;
}
.resizer:hover,
.resizer.active {
  background: var(--color-accent);
}
.resizer.collapsed-spacer {
  background: transparent;
  cursor: default;
  pointer-events: none;
}

/* Right pane uses flex layout so the toggle reserves its own column on the
 * left edge without overlapping DetailPanel's tab bar. The content child
 * owns its own scrolling. */
.pane.right {
  display: flex;
  align-items: stretch;
  overflow: hidden;
}
.pane.right.collapsed {
  border-left: 1px solid var(--color-border);
}
.pane-content {
  flex: 1 1 auto;
  min-width: 0;
  height: 100%;
  overflow: auto;
}

/* Subtle chevron toggle. Background matches the panel surface so the button
 * reads as part of the chrome; only the chevron glyph stands out, and the
 * surface lifts on hover. */
.collapse-toggle {
  flex: 0 0 14px;
  height: 100%;
  padding: 0;
  border: none;
  border-right: 1px solid var(--color-border);
  background: var(--color-bg-elev);
  color: var(--color-muted);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition:
    background-color var(--transition-quick),
    color var(--transition-quick);
}
.collapse-toggle:hover {
  background: var(--color-bg-elev-2);
  color: var(--color-text);
}
.collapse-toggle:focus-visible {
  outline: none;
  box-shadow: var(--focus-ring);
}
/* When collapsed, the entire pane strip *is* the affordance — drop the
 * inner border so it reads as a single rail, not a button on a panel. */
.pane.right.collapsed .collapse-toggle {
  flex: 1 1 auto;
  border-right: none;
  background: transparent;
}
.pane.right.collapsed .collapse-toggle:hover {
  background: var(--color-bg-elev-2);
}
.chevron {
  display: block;
}
</style>
