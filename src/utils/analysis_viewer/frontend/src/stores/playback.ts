import { defineStore } from 'pinia';
import { ref } from 'vue';

export type VideoMode = 'original' | 'motion' | 'gaze';

/**
 * Single, monotonically increasing token tagged onto a target frame so that
 * VideoStage can detect new seek requests issued from elsewhere (timeline
 * clicks, keyboard shortcuts, etc.) without re-seeking on its own
 * frame-changed echoes.
 */
export type SeekRequest = { frame: number; token: number };

export const usePlaybackStore = defineStore('playback', () => {
  const currentFrame = ref(0);
  const currentTime = ref(0);
  const isPlaying = ref(false);
  const speed = ref(1);
  const mode = ref<VideoMode>('original');

  const seekRequest = ref<SeekRequest | null>(null);
  let seekToken = 0;

  function setFrame(frame: number, time: number): void {
    currentFrame.value = frame;
    currentTime.value = time;
  }

  function requestSeek(frame: number): void {
    seekToken += 1;
    seekRequest.value = { frame, token: seekToken };
  }

  return {
    currentFrame,
    currentTime,
    isPlaying,
    speed,
    mode,
    seekRequest,
    setFrame,
    requestSeek,
  };
});
