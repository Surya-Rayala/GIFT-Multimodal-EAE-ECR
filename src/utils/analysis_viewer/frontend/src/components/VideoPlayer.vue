<template>
  <div class="video-player" :class="{ 'has-error': !!errorMsg, 'is-aux': props.role === 'aux' }">
    <video
      v-if="props.src"
      ref="videoEl"
      :src="props.src"
      :muted="props.muted"
      :controls="false"
      preload="auto"
      playsinline
      @loadedmetadata="onLoadedMetadata"
      @canplay="onCanPlay"
      @play="emit('play')"
      @pause="emit('pause')"
      @error="onError"
    />
    <div v-else class="placeholder">
      <p class="muted">Video not available</p>
    </div>
    <div v-if="errorMsg" class="error-banner">{{ errorMsg }}</div>
  </div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from 'vue';

type PlayerRole = 'primary' | 'aux';

const props = defineProps<{
  src: string | null;
  role: PlayerRole;
  muted: boolean;
  frameRate: number;
  // For aux only: latest authoritative frame index from the primary clock.
  syncFrame?: number;
  // Only seek aux when |currFrame - syncFrame| >= this many frames.
  syncMinDelta?: number;
}>();

// Latched desired playback state. Parents call ``play()`` / ``pause()``
// through the exposed surface; we record the intent here and reapply it
// whenever the underlying element reaches a state where ``play()`` is
// likely to actually start playback (loadedmetadata, canplay, after a seek
// completes). This is the fix for the aux video freezing on initial split-view
// load: ``primary.play()`` and ``aux.play()`` are issued at the same time, but
// aux is often still buffering so its play promise silently rejects. With a
// latched intent, the next readiness event resumes it without user input.
let desiredPlaying = false;

const emit = defineEmits<{
  frame: [frameIdx: number, timeSec: number];
  ready: [duration: number, totalFrames: number];
  play: [];
  pause: [];
  error: [message: string];
}>();

const videoEl = ref<HTMLVideoElement | null>(null);
const errorMsg = ref<string | null>(null);

let rvfcHandle: number | null = null;

function startFrameLoop(): void {
  const v = videoEl.value;
  if (!v) return;
  if (typeof (v as any).requestVideoFrameCallback !== 'function') return;
  const tick = (_now: number, meta: { mediaTime?: number }) => {
    if (!videoEl.value) return;
    const time = typeof meta?.mediaTime === 'number' ? meta.mediaTime : videoEl.value.currentTime;
    const idx = Math.max(0, Math.floor(time * props.frameRate));
    emit('frame', idx, time);
    rvfcHandle = (videoEl.value as any).requestVideoFrameCallback(tick);
  };
  rvfcHandle = (v as any).requestVideoFrameCallback(tick);
}

function stopFrameLoop(): void {
  const v = videoEl.value;
  if (rvfcHandle != null && v && typeof (v as any).cancelVideoFrameCallback === 'function') {
    (v as any).cancelVideoFrameCallback(rvfcHandle);
  }
  rvfcHandle = null;
}

function onLoadedMetadata(): void {
  const v = videoEl.value;
  if (!v) return;
  errorMsg.value = null;
  emit('ready', v.duration, Math.round(v.duration * props.frameRate));

  // Primary drives the clock; aux follows via syncFrame watcher.
  if (props.role === 'primary') {
    stopFrameLoop();
    startFrameLoop();
    applyDesiredPlayState();
    return;
  }
  // Aux: snap to the current authoritative frame on first load.
  if (props.role === 'aux' && typeof props.syncFrame === 'number' && props.syncFrame > 0) {
    v.currentTime = props.syncFrame / props.frameRate;
  }
  applyDesiredPlayState();
}

function onCanPlay(): void {
  // Fired whenever enough data is buffered to play forward — including
  // after a seek. Reapplying the desired state here is what unsticks aux
  // when its initial autoplay raced the buffer and silently rejected.
  applyDesiredPlayState();
}

function applyDesiredPlayState(): void {
  const v = videoEl.value;
  if (!v) return;
  if (desiredPlaying && v.paused) {
    // play() is async and can reject (autoplay policy, not enough data,
    // src changed mid-flight). On rejection we just log and wait for the
    // next readiness event — no UI noise.
    const p = v.play();
    if (p && typeof p.catch === 'function') {
      p.catch(() => undefined);
    }
  } else if (!desiredPlaying && !v.paused) {
    v.pause();
  }
}

function onError(): void {
  errorMsg.value = 'Failed to load video';
  emit('error', 'load failed');
}

// Aux follows primary's clock. The naive policy ("seek aux whenever |delta|
// exceeds 2 frames") causes a vicious cycle during normal playback: the
// browser deprioritises decoding a muted aux video, drift accumulates past
// the 2-frame threshold within a few ticks, the watcher issues a seek, the
// seek itself takes 50-200ms (demux to keyframe, decode forward), more
// primary frames arrive during that pause, the next watcher tick issues
// another seek before the first finishes — and the map view appears frozen
// for hundreds of milliseconds at a time while the camera plays smoothly.
//
// Three guards together avoid the freeze without sacrificing scrub accuracy:
//   1. Don't stack seeks: bail if a previous seek hasn't completed.
//   2. Tier the drift threshold by play state — small while paused (so
//      post-scrub frames line up), generous while playing (sub-second drift
//      is invisible on a top-down map; jitter from unnecessary seeks isn't).
//   3. Always seek immediately on a "hard jump" (≥ 1 second), which is what
//      a scrub or mode-change emits — those still need exact alignment.
watch(
  () => props.syncFrame,
  (newFrame) => {
    if (props.role !== 'aux' || typeof newFrame !== 'number' || !videoEl.value) return;
    const v = videoEl.value;
    if (Number.isNaN(v.duration) || !Number.isFinite(v.duration)) return; // not ready yet
    if (v.seeking) return; // a previous seek is still in flight; let it finish

    const targetTime = newFrame / props.frameRate;
    const currentFrame = Math.floor(v.currentTime * props.frameRate);
    const dt = Math.abs(currentFrame - newFrame);

    const fps = props.frameRate || 60;
    const hardJumpDelta = Math.max(60, fps); // ≥ 1 s → assume scrub / mode change
    const playDriftDelta = Math.max(30, Math.round(fps * 0.5)); // ≥ 0.5 s drift while playing
    const pausedMinDelta = props.syncMinDelta ?? 2;

    let threshold: number;
    if (dt >= hardJumpDelta) {
      threshold = 0; // always seek
    } else if (v.paused) {
      threshold = pausedMinDelta;
    } else {
      threshold = playDriftDelta;
    }

    if (dt >= threshold) {
      v.currentTime = targetTime;
    }
  },
);

// When ``src`` changes the browser tears down + reloads the underlying
// HTMLMediaElement; the upcoming loadedmetadata + canplay events will reapply
// ``desiredPlaying`` for us. This watcher exists only to clear any stale
// error banner so a fresh src starts with a clean UI.
watch(
  () => props.src,
  () => {
    errorMsg.value = null;
  },
);

defineExpose({
  play: () => {
    desiredPlaying = true;
    applyDesiredPlayState();
  },
  pause: () => {
    desiredPlaying = false;
    applyDesiredPlayState();
  },
  seek: (timeSec: number) => {
    if (videoEl.value) videoEl.value.currentTime = timeSec;
  },
  setRate: (rate: number) => {
    if (videoEl.value) videoEl.value.playbackRate = rate;
  },
  element: () => videoEl.value,
});

onBeforeUnmount(stopFrameLoop);
</script>

<style scoped>
.video-player {
  position: relative;
  background: #000;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  overflow: hidden;
  border-radius: 4px;
}
.video-player video {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}
.video-player .placeholder {
  text-align: center;
  padding: 16px;
}
.video-player .placeholder .muted {
  color: var(--color-muted);
  margin: 0;
}
.video-player.is-aux {
  border: 1px solid var(--color-border);
}
.error-banner {
  position: absolute;
  bottom: 8px;
  left: 8px;
  background: rgba(220, 38, 38, 0.85);
  color: white;
  padding: 4px 9px;
  border-radius: 3px;
  font-size: 0.82em;
  pointer-events: none;
}
</style>
