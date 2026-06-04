import { defineStore } from 'pinia';
import { ref } from 'vue';

import type { StepKey } from '@/types/models';

export const useUIStore = defineStore('ui', () => {
  const currentStep = ref<StepKey>('setup');

  // Per-step "have I been here before" tracking — drives the StepBar's
  // visited/done state visuals.
  const visited = ref<Record<StepKey, boolean>>({
    setup: true,
    align: false,
    entries: false,
    pods: false,
    boundary: false,
  });

  // Per-step "have I saved" tracking — drives the done checkmark in StepBar
  // and the dirty-state warnings on close.
  const saved = ref<Record<StepKey, boolean>>({
    setup: false,
    align: false,
    entries: false,
    pods: false,
    boundary: false,
  });

  // When true, the main shell hides the StepBar / footer and shows the
  // completion screen instead. Set by App.vue when the user clicks Finish.
  const showComplete = ref<boolean>(false);

  function goTo(step: StepKey): void {
    currentStep.value = step;
    visited.value = { ...visited.value, [step]: true };
  }

  function markSaved(step: StepKey, was: boolean = true): void {
    saved.value = { ...saved.value, [step]: was };
  }

  // Wipe all in-memory state back to "fresh open" — used by "Start Over"
  // so the user can do another room without re-launching the app. Source
  // file paths in the project store are reset separately by the caller.
  function reset(): void {
    currentStep.value = 'setup';
    visited.value = {
      setup: true,
      align: false,
      entries: false,
      pods: false,
      boundary: false,
    };
    saved.value = {
      setup: false,
      align: false,
      entries: false,
      pods: false,
      boundary: false,
    };
    showComplete.value = false;
  }

  return { currentStep, visited, saved, showComplete, goTo, markSaved, reset };
});
