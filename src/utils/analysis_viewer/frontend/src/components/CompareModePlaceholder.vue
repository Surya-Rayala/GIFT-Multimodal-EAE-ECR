<template>
  <div class="compare-placeholder">
    <div class="card">
      <h4 class="title">Compare mode</h4>
      <p class="body muted">
        The Summary / Metrics / Raw tabs are part of Analysis mode.
        Switch back to use them on the active run.
      </p>
      <p v-if="contextLabel" class="context muted">{{ contextLabel }}</p>
      <button class="switch" type="button" @click="onSwitch">
        Switch to Analysis
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';

import { useCompareStore } from '@/stores/compare';
import { useUIStore } from '@/stores/ui';

const ui = useUIStore();
const compare = useCompareStore();

const contextLabel = computed(() => {
  const other = compare.runs.find((r) => r.path === compare.selectedOtherRunPath);
  if (!other) return '';
  const badge = other.role === 'expert' ? ' (expert)' : '';
  return `vs ${other.title}${badge}`;
});

function onSwitch(): void {
  ui.viewMode = 'analysis';
}
</script>

<style scoped>
.compare-placeholder {
  display: flex;
  align-items: flex-start;
  justify-content: center;
  height: 100%;
  padding: 24px 16px;
  box-sizing: border-box;
}

.card {
  display: flex;
  flex-direction: column;
  gap: 10px;
  max-width: 320px;
  width: 100%;
  padding: 16px 18px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  background: var(--color-bg-elev);
}

.title {
  margin: 0;
  font-size: 0.95em;
  font-weight: 600;
  color: var(--color-text);
}

.body {
  margin: 0;
  font-size: 0.88em;
  line-height: 1.5;
}

.context {
  margin: 0;
  font-size: 0.82em;
  font-variant-numeric: tabular-nums;
  padding: 6px 8px;
  border-radius: 4px;
  background: var(--color-bg-deep, var(--color-bg));
}

.switch {
  align-self: flex-start;
  margin-top: 4px;
  padding: 6px 12px;
  background: var(--color-accent);
  color: var(--color-bg);
  border: none;
  border-radius: 4px;
  font-size: 0.85em;
  font-weight: 500;
  cursor: pointer;
}
.switch:hover {
  filter: brightness(1.08);
}

.muted {
  color: var(--color-muted);
}
</style>
