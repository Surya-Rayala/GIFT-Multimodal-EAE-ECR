<template>
  <div class="compare-stage">
    <header class="stage-header">
      <h3 class="title">{{ headerTitle }}</h3>
      <p v-if="subtitle" class="subtitle muted">{{ subtitle }}</p>
      <!-- Layout toggle — only shown when there are 2+ visualizations, so
           single-chart metrics keep a clean header. -->
      <div
        v-if="result && result.visualizations.length >= 2"
        class="layout-toggle"
        role="group"
        aria-label="Visualization layout"
      >
        <button
          type="button"
          :class="{ active: compare.vizLayout === 'stack' }"
          @click="compare.setVizLayout('stack')"
          title="Stack vertically — each chart at full pane width"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
            <rect x="1" y="2"  width="12" height="4" rx="1" fill="currentColor" />
            <rect x="1" y="8"  width="12" height="4" rx="1" fill="currentColor" />
          </svg>
          Stacked
        </button>
        <button
          type="button"
          :class="{ active: compare.vizLayout === 'row' }"
          @click="compare.setVizLayout('row')"
          title="Side by side — charts share the pane width"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
            <rect x="2" y="1" width="4" height="12" rx="1" fill="currentColor" />
            <rect x="8" y="1" width="4" height="12" rx="1" fill="currentColor" />
          </svg>
          Side by side
        </button>
      </div>
    </header>

    <div class="stage-body">
      <p v-if="compare.loading" class="muted status">Computing comparison…</p>
      <p v-else-if="compare.error" class="message error">{{ compare.error }}</p>

      <template v-else-if="result">
        <div class="viz-stack" :class="{ row: compare.vizLayout === 'row' && result.visualizations.length >= 2 }">
          <section
            v-for="(viz, i) in result.visualizations"
            :key="i"
            class="viz-card"
            :class="{ first: i === 0, last: i === result.visualizations.length - 1 }"
          >
            <header class="viz-card-header">
              <span class="viz-index">{{ i + 1 }} / {{ result.visualizations.length }}</span>
              <h4 class="viz-label">{{ viz.label || `Visualization ${i + 1}` }}</h4>
            </header>
            <div class="viz-img-wrap">
              <img
                class="viz-img"
                :src="imageSrc(viz.image_path)"
                :alt="viz.label || result.label"
              />
            </div>
            <p class="viz-caption">{{ viz.caption }}</p>
          </section>
        </div>

        <p v-if="result.explanation" class="message explanation">
          {{ result.explanation }}
        </p>
      </template>

      <p v-else class="muted hint">
        Pick a run and a metric on the left to see a comparison.
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';

import { imageUrl } from '@/api/client';
import { useCompareStore } from '@/stores/compare';
import { useSessionStore } from '@/stores/session';

const compare = useCompareStore();
const session = useSessionStore();

const result = computed(() => compare.result);

const headerTitle = computed(() => result.value?.label ?? 'Compare');

const subtitle = computed(() => {
  if (!result.value) return '';
  const other = compare.runs.find((r) => r.path === compare.selectedOtherRunPath);
  if (!other) return '';
  const badge = other.role === 'expert' ? ' (expert)' : '';
  return `vs ${other.title}${badge}`;
});

function imageSrc(absPath: string): string {
  const sessionPath = session.session?.session_json_path ?? '';
  return imageUrl(absPath, sessionPath);
}
</script>

<style scoped>
.compare-stage {
  display: flex;
  flex-direction: column;
  height: 100%;
  gap: 10px;
}

.stage-header {
  /* The CenterPanel renders an absolutely-positioned Compare/Analysis
   * mode toggle in the top-right corner of the central pane (z-index 10).
   * Reserve enough right-padding here so the right-aligned layout toggle
   * and the title text don't slide under it at any pane width. */
  padding: 4px 130px 0 4px;
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.title {
  margin: 0;
  font-size: 1.05em;
  font-weight: 600;
}
.subtitle {
  margin: 0;
  font-size: 0.9em;
  flex: 1;
}

/* Segmented control: Stacked / Side by side. */
.layout-toggle {
  display: inline-flex;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  overflow: hidden;
  margin-left: auto;
  flex-shrink: 0;
}
.layout-toggle button {
  background: transparent;
  color: var(--color-muted);
  border: none;
  padding: 5px 10px;
  font-size: 0.82em;
  font-weight: 500;
  cursor: pointer;
  border-right: 1px solid var(--color-border);
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.layout-toggle button:last-child {
  border-right: none;
}
.layout-toggle button:hover:not(.active) {
  color: var(--color-text);
}
.layout-toggle button.active {
  background: var(--color-accent);
  color: var(--color-bg);
}

.stage-body {
  flex: 1;
  overflow: auto;
  padding: 4px 6px 6px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.status,
.hint {
  margin: 0;
}

/* Default: charts stack vertically — each card at full pane width so the
 * rendered visualization is as large as available space allows. Users
 * scroll within the pane when there are multiple charts. */
.viz-stack {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

/* Row variant: CSS grid with auto-fit so 2 charts split the pane 50/50,
 * 3 charts take a third each, etc. If the pane is too narrow to fit
 * cards at their minimum width, the grid gracefully falls back to a
 * single column — so the layout is always sane regardless of pane size. */
.viz-stack.row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(360px, 100%), 1fr));
  gap: 14px;
  align-items: start;
}

.viz-card {
  border: 1px solid var(--color-border);
  border-radius: 8px;
  background: var(--color-bg-elev);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.viz-card-header {
  display: flex;
  align-items: baseline;
  gap: 10px;
  padding: 10px 14px;
  background: var(--color-bg-elev-2, var(--color-bg-deep));
  border-bottom: 1px solid var(--color-border);
}
.viz-index {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.78em;
  color: var(--color-muted);
  letter-spacing: 0.04em;
}
.viz-label {
  margin: 0;
  font-size: 0.95em;
  font-weight: 600;
  color: var(--color-text);
  letter-spacing: 0.02em;
}

/* Image area: centred, with a generous max-height so portrait images
 * don't dominate the viewport. The wrapper provides a subtle inset so
 * the image stands off the card edges. */
.viz-img-wrap {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 14px;
  background: var(--color-bg-deep);
}
.viz-img {
  display: block;
  width: 100%;
  height: auto;
  max-height: 70vh;
  object-fit: contain;
  border-radius: 4px;
}
/* Row layout — cap image height tighter so two side-by-side cards
 * don't each occupy 70vh and overflow the pane. */
.viz-stack.row .viz-img {
  max-height: 56vh;
}

/* Phones / small tablets. Two things:
 *  1. Cap image height so a card doesn't eat the whole screen.
 *  2. Honour an explicit "Side by side" choice even here — the desktop 360px
 *     column floor can't fit twice on a phone, so auto-fit would silently drop
 *     to one column. Lower the floor so the user's toggle actually shows two
 *     columns sharing the width (smaller charts; "Stacked" remains the default
 *     for full-width readability). Wider panes still pack more columns. */
@media (max-width: 1000px) {
  .viz-img {
    max-height: 52vh;
  }
  .viz-stack.row {
    grid-template-columns: repeat(auto-fit, minmax(min(150px, 47%), 1fr));
  }
  .viz-stack.row .viz-img {
    max-height: 40vh;
  }
}

.viz-caption {
  margin: 0;
  padding: 10px 14px 14px;
  font-size: 0.92em;
  color: var(--color-text-secondary);
  line-height: 1.5;
}

.message {
  background: var(--color-bg-elev);
  padding: 10px 12px;
  border-radius: 4px;
  margin: 0;
  border-left: 3px solid var(--color-accent);
  font-size: 0.92em;
}
.message.explanation {
  border-left-color: var(--color-accent);
}
.message.error {
  border-left-color: var(--color-danger-text, #ef4444);
  color: var(--color-danger-text);
}
.muted {
  color: var(--color-muted);
}
</style>
