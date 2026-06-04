<template>
  <div class="setup">
    <div class="setup-grid">
      <HelpCard
        title="Pick your files and where to save"
        :steps="[
          'Choose the room map image (PNG/JPG).',
          'Choose a camera reference — either a video or a single frame image.',
          'Pick where to save your outputs and give the project a short name.',
        ]"
        why="The four files this tool produces are stored in your chosen folder and named with your project prefix. The map and camera images aren't moved or modified."
      />

      <form class="form" @submit.prevent>
        <!-- ---------------------------------------------------------- -->
        <!-- Map image                                                  -->
        <!-- ---------------------------------------------------------- -->
        <div class="row">
          <label>Map image</label>
          <div class="value-row">
            <span class="path-text" :title="project.mapImagePath || ''">
              {{ project.mapImagePath || 'No file selected' }}
            </span>
            <button class="primary" type="button" @click="pickMap">Choose…</button>
          </div>
          <div v-if="project.mapImagePath" class="preview">
            <img :src="mapSrc" alt="Map preview" @error="mapPreviewBroken = true" @load="mapPreviewBroken = false" />
            <p v-if="mapPreviewBroken" class="preview-error">
              Preview failed — the file may not be a readable image, or the path moved.
            </p>
          </div>
        </div>

        <!-- ---------------------------------------------------------- -->
        <!-- Camera mode toggle                                         -->
        <!-- ---------------------------------------------------------- -->
        <div class="row">
          <label>Camera reference</label>
          <div class="cam-tabs">
            <button
              type="button"
              :class="{ active: camMode === 'video' }"
              @click="camMode = 'video'"
            >Video</button>
            <button
              type="button"
              :class="{ active: camMode === 'frame' }"
              @click="camMode = 'frame'"
            >Frame image</button>
          </div>
        </div>

        <!-- ---------------------------------------------------------- -->
        <!-- Video file + frame scrubber + preview                      -->
        <!-- ---------------------------------------------------------- -->
        <div class="row" v-if="camMode === 'video'">
          <label>Video file</label>
          <div class="value-row">
            <span class="path-text" :title="project.cameraVideoPath || ''">
              {{ project.cameraVideoPath || 'No file selected' }}
            </span>
            <button class="primary" type="button" @click="pickVideo">Choose…</button>
          </div>
          <p v-if="videoError" class="row-error">{{ videoError }}</p>
        </div>

        <div class="row" v-if="camMode === 'video' && project.cameraFrameCount > 0">
          <label>Frame</label>
          <div class="frame-row">
            <input
              type="range"
              :min="0"
              :max="Math.max(0, project.cameraFrameCount - 1)"
              :value="project.cameraFrameIndex"
              @input="onFrameSliderInput(($event.target as HTMLInputElement).value)"
            />
            <span class="frame-readout">
              {{ project.cameraFrameIndex + 1 }} / {{ project.cameraFrameCount }}
            </span>
          </div>
          <div class="preview">
            <img
              :src="cameraSrc"
              alt="Camera frame preview"
              @error="cameraPreviewBroken = true"
              @load="cameraPreviewBroken = false"
            />
            <p v-if="cameraPreviewBroken" class="preview-error">
              The sidecar couldn’t render this frame. Check the engine log, then try a different frame or video.
            </p>
          </div>
        </div>

        <!-- ---------------------------------------------------------- -->
        <!-- Frame image alternative                                    -->
        <!-- ---------------------------------------------------------- -->
        <div class="row" v-if="camMode === 'frame'">
          <label>Frame image</label>
          <div class="value-row">
            <span class="path-text" :title="project.cameraFrameImagePath || ''">
              {{ project.cameraFrameImagePath || 'No file selected' }}
            </span>
            <button class="primary" type="button" @click="pickFrameImage">Choose…</button>
          </div>
          <div v-if="project.cameraFrameImagePath" class="preview">
            <img
              :src="cameraSrc"
              alt="Camera frame preview"
              @error="cameraPreviewBroken = true"
              @load="cameraPreviewBroken = false"
            />
            <p v-if="cameraPreviewBroken" class="preview-error">
              Preview failed — the file may not be a readable image, or the path moved.
            </p>
          </div>
        </div>

        <!-- ---------------------------------------------------------- -->
        <!-- Save folder + name                                         -->
        <!-- ---------------------------------------------------------- -->
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
          <input
            class="text"
            type="text"
            v-model="project.projectName"
            placeholder="e.g. crested_gecko"
          />
        </div>
      </form>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';

import HelpCard from './HelpCard.vue';
import { imageUrl, probeVideo, videoFrameUrl } from '@/api/client';
import { useProjectStore } from '@/stores/project';

const project = useProjectStore();
const camMode = ref<'video' | 'frame'>('video');
const videoError = ref<string>('');
const mapPreviewBroken = ref(false);
const cameraPreviewBroken = ref(false);

// Same lists as the backend whitelist. Kept in sync manually; if you add
// an extension here, update VIDEO_EXTS / IMAGE_EXTS in routers/media.py too.
const VIDEO_EXTS = ['mp4', 'mov', 'avi', 'mkv', 'm4v', 'webm', 'wmv', 'flv', 'mpg', 'mpeg', 'mts', 'm2ts', 'ts', '3gp'];
const IMAGE_EXTS = ['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'webp'];

const mapSrc = computed(() =>
  project.mapImagePath ? imageUrl(project.mapImagePath) : '',
);

const cameraSrc = computed(() => {
  if (project.cameraVideoPath) {
    return videoFrameUrl(project.cameraVideoPath, project.cameraFrameIndex);
  }
  if (project.cameraFrameImagePath) {
    return imageUrl(project.cameraFrameImagePath);
  }
  return '';
});

onMounted(() => {
  if (project.cameraFrameImagePath && !project.cameraVideoPath) camMode.value = 'frame';
});

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
    mapPreviewBroken.value = false;
  }
}

async function pickVideo(): Promise<void> {
  videoError.value = '';
  const p = await nativeOpen([{ name: 'Video', extensions: VIDEO_EXTS }]);
  if (!p) return;
  try {
    const info = await probeVideo(p);
    project.cameraVideoPath = p;
    project.cameraFrameCount = info.frame_count;
    project.cameraFrameIndex = 0;
    project.cameraFrameImagePath = '';     // clear the other source
    cameraPreviewBroken.value = false;
  } catch (e) {
    // Don't write to the store — keep the previous selection (if any) intact.
    const msg = (e as Error).message || String(e);
    videoError.value = `Could not open video: ${msg}`;
  }
}

async function pickFrameImage(): Promise<void> {
  const p = await nativeOpen([{ name: 'Image', extensions: IMAGE_EXTS }]);
  if (!p) return;
  project.cameraFrameImagePath = p;
  project.cameraVideoPath = '';            // clear the other source
  project.cameraFrameCount = 0;
  project.cameraFrameIndex = 0;
  cameraPreviewBroken.value = false;
}

async function pickDir(): Promise<void> {
  const p = await nativeOpenDir();
  if (p) project.outputDir = p;
}

function onFrameSliderInput(v: string): void {
  const n = Number(v);
  if (!Number.isFinite(n)) return;
  project.cameraFrameIndex = Math.max(0, Math.min(project.cameraFrameCount - 1, Math.round(n)));
}

// If the user toggles modes, clear the path from the other side so the
// downstream views don't keep showing stale data.
watch(camMode, (mode) => {
  videoError.value = '';
  cameraPreviewBroken.value = false;
  if (mode === 'video') {
    project.cameraFrameImagePath = '';
  } else {
    project.cameraVideoPath = '';
    project.cameraFrameCount = 0;
    project.cameraFrameIndex = 0;
  }
});
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

.cam-tabs {
  display: inline-flex;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  overflow: hidden;
  width: fit-content;
}
.cam-tabs button {
  background: transparent;
  color: var(--color-muted);
  border: none;
  padding: 5px 14px;
  font-size: 0.88em;
  cursor: pointer;
  border-right: 1px solid var(--color-border);
}
.cam-tabs button:last-child { border-right: none; }
.cam-tabs button.active {
  background: var(--color-accent);
  color: var(--color-bg);
}

.frame-row {
  display: flex;
  align-items: center;
  gap: 12px;
}
.frame-row input[type="range"] { flex: 1; }
.frame-readout {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.85em;
  color: var(--color-text-secondary);
  min-width: 90px;
  text-align: right;
}

.text {
  background: var(--color-bg-deep);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  padding: 6px 8px;
  font-size: 0.9em;
}

/* Inline image preview under each chosen-file row. */
.preview {
  margin-top: 4px;
  background: var(--color-bg-deep);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.preview img {
  display: block;
  width: 100%;
  max-height: 280px;
  object-fit: contain;
  background: #000;
}
.preview-error,
.row-error {
  margin: 0;
  padding: 8px 10px;
  font-size: 0.85em;
  color: var(--color-danger-text);
  background: var(--color-danger-bg);
  border-radius: 4px;
}
.row-error {
  border: 1px solid var(--color-danger-text);
}
</style>
