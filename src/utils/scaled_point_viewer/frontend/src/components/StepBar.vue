<template>
  <ol class="step-bar">
    <li
      v-for="(s, idx) in STEPS"
      :key="s.id"
      class="step"
      :class="stateOf(s.key)"
      @click="onStepClick(s.key)"
    >
      <span class="num">{{ s.id }}</span>
      <span class="title">{{ s.title }}</span>
      <span v-if="idx < STEPS.length - 1" class="connector" aria-hidden="true" />
    </li>
  </ol>
</template>

<script setup lang="ts">
import { useUIStore } from '@/stores/ui';
import { STEPS, type StepKey } from '@/types/models';

const ui = useUIStore();

function stateOf(key: StepKey): string {
  if (ui.currentStep === key) return 'active';
  if (ui.visited[key]) return 'visited';
  return 'pending';
}

function onStepClick(key: StepKey): void {
  if (key === ui.currentStep) return;
  if (ui.visited[key]) ui.goTo(key);
}
</script>

<style scoped>
.step-bar {
  list-style: none;
  margin: 0;
  padding: 0 14px;
  display: flex;
  align-items: center;
  gap: 0;
  background: var(--color-bg-elev);
  border-bottom: 1px solid var(--color-border);
}
.step {
  position: relative;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 14px 12px 4px;
  font-size: 0.92em;
  color: var(--color-muted);
  cursor: default;
  flex: 1;
  min-width: 0;
}
.step.visited { cursor: pointer; }
.step.visited:hover { color: var(--color-text); }
.step .num {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.85em;
  font-weight: 700;
  background: var(--color-border);
  color: var(--color-text-secondary);
  flex-shrink: 0;
}
.step.active .num {
  background: var(--color-accent);
  color: var(--color-bg);
}
.step.active {
  color: var(--color-text);
  font-weight: 600;
}
.step .title {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.step .connector {
  flex: 1;
  height: 2px;
  background: var(--color-border);
  margin: 0 4px 0 8px;
}
</style>
