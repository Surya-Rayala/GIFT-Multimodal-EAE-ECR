import { defineStore } from 'pinia';
import { ref } from 'vue';

export const useProjectStore = defineStore('project', () => {
  // Default project name and save folder. Seeded from VITE_INITIAL_PROJECT_DIR
  // when the user launched with `--project-dir`.
  const outputDir = ref<string>(
    (import.meta.env.VITE_INITIAL_PROJECT_DIR as string | undefined) ?? '',
  );
  const projectName = ref<string>('project');

  // Source files chosen during Setup.
  const mapImagePath = ref<string>('');
  const cameraVideoPath = ref<string>('');
  const cameraFrameImagePath = ref<string>('');
  const cameraFrameIndex = ref<number>(0);
  const cameraFrameCount = ref<number>(0);

  function reset(): void {
    outputDir.value = '';
    projectName.value = 'project';
    mapImagePath.value = '';
    cameraVideoPath.value = '';
    cameraFrameImagePath.value = '';
    cameraFrameIndex.value = 0;
    cameraFrameCount.value = 0;
  }

  return {
    outputDir,
    projectName,
    mapImagePath,
    cameraVideoPath,
    cameraFrameImagePath,
    cameraFrameIndex,
    cameraFrameCount,
    reset,
  };
});
