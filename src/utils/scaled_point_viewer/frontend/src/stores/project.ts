import { defineStore } from 'pinia';
import { ref } from 'vue';

export type Unit = 'ft' | 'm' | 'in';

export const useProjectStore = defineStore('project', () => {
  // Seeded from VITE_INITIAL_PROJECT_DIR when launched with `--project-dir`.
  const outputDir = ref<string>(
    (import.meta.env.VITE_INITIAL_PROJECT_DIR as string | undefined) ?? '',
  );
  const projectName = ref<string>('project');
  const mapImagePath = ref<string>('');
  // Display-only label for measurements; everything must use the same unit.
  const unit = ref<Unit>('ft');

  function reset(): void {
    outputDir.value = '';
    projectName.value = 'project';
    mapImagePath.value = '';
    unit.value = 'ft';
  }

  return { outputDir, projectName, mapImagePath, unit, reset };
});
