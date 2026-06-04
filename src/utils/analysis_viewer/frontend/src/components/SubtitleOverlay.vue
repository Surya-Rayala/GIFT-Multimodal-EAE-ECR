<template>
  <transition name="fade">
    <div v-if="activeText" class="subtitle-overlay">
      {{ activeText }}
    </div>
  </transition>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useSessionStore } from '@/stores/session';
import { usePlaybackStore } from '@/stores/playback';
import type { TranscriptionSegment } from '@/types/models';

const session = useSessionStore();
const playback = usePlaybackStore();

const segments = computed<TranscriptionSegment[]>(() => {
  const t = session.session?.transcription;
  if (!t || !Array.isArray(t.segments)) return [];
  // The PyQt5 viewer assumes segments are sorted by `start`. Trust the engine.
  return t.segments;
});

const activeText = computed<string | null>(() => {
  const segs = segments.value;
  if (segs.length === 0) return null;
  const time = playback.currentTime;

  // Binary search: find the rightmost segment with start <= time.
  let lo = 0;
  let hi = segs.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (segs[mid].start <= time) lo = mid + 1;
    else hi = mid;
  }
  if (lo === 0) return null;
  const seg = segs[lo - 1];
  if (seg.end >= time) return seg.text || null;
  return null;
});
</script>

<style scoped>
.subtitle-overlay {
  position: absolute;
  left: 50%;
  bottom: 6%;
  transform: translateX(-50%);
  pointer-events: none;
  background: rgba(0, 0, 0, 0.62);
  color: #f3f4f6;
  font-size: 1.05em;
  font-weight: 500;
  padding: 6px 14px;
  border-radius: 5px;
  max-width: 88%;
  text-align: center;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.8);
  white-space: pre-wrap;
  letter-spacing: 0.01em;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 140ms ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
