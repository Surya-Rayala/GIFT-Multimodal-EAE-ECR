import { defineStore } from 'pinia';
import { ref } from 'vue';

import type { StepKey } from '@/types/models';

export const useUIStore = defineStore('ui', () => {
  const currentStep = ref<StepKey>('setup');

  const visited = ref<Record<StepKey, boolean>>({
    setup: true,
    walls: false,
    points: false,
  });

  function goTo(step: StepKey): void {
    currentStep.value = step;
    visited.value = { ...visited.value, [step]: true };
  }

  function reset(): void {
    currentStep.value = 'setup';
    visited.value = { setup: true, walls: false, points: false };
  }

  return { currentStep, visited, goTo, reset };
});
