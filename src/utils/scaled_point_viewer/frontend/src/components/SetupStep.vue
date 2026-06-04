<template>
  <div class="setup">
    <div class="setup-grid">
      <HelpCard
        title="Pick your map and where to save"
        :steps="[
          'Choose the room map image (PNG/JPG) — the same one you use elsewhere.',
          'Pick the folder to save into and give the project a short name.',
          'Choose the unit you measured in (feet, meters, or inches).',
        ]"
        why="This tool drops real-world reference points onto the map. It saves a points file, an annotated map image, and a reloadable project — all named with your project prefix. The map image is not modified."
      />

      <form class="form" @submit.prevent>
        <div class="row">
          <label>Map image</label>
          <div class="value-row">
            <span class="path-text" :title="project.mapImagePath || ''">
              {{ project.mapImagePath || 'No file selected' }}
            </span>
            <button class="primary" type="button" @click="pickMap">Choose…</button>
          </div>
          <div v-if="project.mapImagePath" class="preview">
            <img :src="mapSrc" alt="Map preview" @error="mapBroken = true" @load="mapBroken = false" />
            <p v-if="mapBroken" class="preview-error">
              Preview failed — the file may not be a readable image, or the path moved.
            </p>
          </div>
        </div>

        <div class="row">
          <label>Save folder</label>
          <div class="value-row">
            <span class="path-text" :title="project.outputDir || ''">
              {{ project.outputDir || 'No folder selected' }}
            </span>
            <button class="primary" type="button" @click="pickDir">Choose…</button>
          </div>
        </div>

        <div class="row">
          <label>Project name</label>
          <input class="text" type="text" v-model="project.projectName" placeholder="e.g. room_a" />
        </div>

        <div class="row">
          <label>Measurement unit</label>
          <div class="unit-tabs">
            <button
              v-for="u in (['ft', 'm', 'in'] as const)"
              :key="u"
              type="button"
              :class="{ active: project.unit === u }"
              @click="project.unit = u"
            >
              {{ u }}
            </button>
          </div>
        </div>
      </form>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';

import HelpCard from './HelpCard.vue';
import { imageUrl } from '@/api/client';
import { useProjectStore } from '@/stores/project';

const project = useProjectStore();
const mapBroken = ref(false);

const IMAGE_EXTS = ['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'webp'];

const mapSrc = computed(() => (project.mapImagePath ? imageUrl(project.mapImagePath) : ''));

async function nativeOpen(filters: { name: string; extensions: string[] }[]): Promise<string | null> {
  try {
    const dialog = await import('@tauri-apps/plugin-dialog');
    const picked = await dialog.open({ multiple: false, directory: false, filters });
    return typeof picked === 'string' ? picked : null;
  } catch {
    return window.prompt('File path:') || null;
  }
}

async function nativeOpenDir(): Promise<string | null> {
  try {
    const dialog = await import('@tauri-apps/plugin-dialog');
    const picked = await dialog.open({ directory: true, multiple: false });
    return typeof picked === 'string' ? picked : null;
  } catch {
    return window.prompt('Folder path:') || null;
  }
}

async function pickMap(): Promise<void> {
  const p = await nativeOpen([{ name: 'Image', extensions: IMAGE_EXTS }]);
  if (p) {
    project.mapImagePath = p;
    mapBroken.value = false;
  }
}

async function pickDir(): Promise<void> {
  const p = await nativeOpenDir();
  if (p) project.outputDir = p;
}
</script>

<style scoped>
.setup {
  padding: 18px;
  overflow: auto;
  height: 100%;
}
.setup-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 18px;
  max-width: 760px;
  margin: 0 auto;
}
@media (min-width: 900px) {
  .setup-grid {
    grid-template-columns: 300px 1fr;
    max-width: 1100px;
    align-items: start;
  }
}
.form {
  background: var(--color-bg-elev);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  padding: 16px 18px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.row {
  display: grid;
  grid-template-columns: 1fr;
  gap: 6px;
}
.row label {
  font-size: 0.78em;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--color-muted);
}
.value-row {
  display: flex;
  align-items: center;
  gap: 8px;
}
.path-text {
  flex: 1;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.85em;
  background: var(--color-bg-deep);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  padding: 6px 8px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.primary {
  background: var(--color-accent);
  color: var(--color-bg);
  border: none;
  padding: 6px 14px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.88em;
  font-weight: 600;
}
.primary:hover { background: var(--color-accent-soft); color: var(--color-text); }
.text {
  background: var(--color-bg-deep);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  padding: 6px 8px;
  font-size: 0.9em;
}
.unit-tabs {
  display: inline-flex;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  overflow: hidden;
  width: fit-content;
}
.unit-tabs button {
  background: transparent;
  color: var(--color-muted);
  border: none;
  border-right: 1px solid var(--color-border);
  padding: 5px 16px;
  font-size: 0.88em;
  cursor: pointer;
}
.unit-tabs button:last-child { border-right: none; }
.unit-tabs button.active {
  background: var(--color-accent);
  color: var(--color-bg);
}
.preview {
  margin-top: 4px;
  background: var(--color-bg-deep);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  overflow: hidden;
}
.preview img {
  display: block;
  width: 100%;
  max-height: 280px;
  object-fit: contain;
  background: #000;
}
.preview-error {
  margin: 0;
  padding: 8px 10px;
  font-size: 0.85em;
  color: var(--color-danger-text);
  background: var(--color-danger-bg);
  border-radius: 4px;
}
</style>
