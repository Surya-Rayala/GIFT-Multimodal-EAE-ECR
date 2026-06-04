<template>
  <div v-if="session.session" class="session-overview">
    <header>
      <h4>{{ session.session.session.video_basename }}</h4>
      <p class="muted small">
        Mode: {{ session.session.session.analysis_mode }} •
        {{ session.session.video.total_frames }} frames •
        {{ session.session.video.duration_sec.toFixed(2) }}s •
        {{ session.session.video.frame_rate.toFixed(2) }} fps
      </p>
    </header>

    <div class="counts">
      <div class="count-card">
        <span class="count">{{ session.session.metrics.length }}</span>
        <span class="label">Metrics</span>
      </div>
      <div class="count-card">
        <span class="count">{{ session.session.timeline.items.length }}</span>
        <span class="label">Items</span>
      </div>
      <div class="count-card">
        <span class="count">{{ session.session.flags.length }}</span>
        <span class="label">Flags</span>
      </div>
    </div>

    <section v-if="dw">
      <h5>Drill window</h5>
      <table class="kv">
        <tbody>
          <tr><td>Frames</td><td>{{ dw.start_frame }} → {{ dw.end_frame }}</td></tr>
          <tr><td>Time</td><td>{{ dw.start_time_sec.toFixed(2) }}s → {{ dw.end_time_sec.toFixed(2) }}s</td></tr>
          <tr v-if="dw.end_uncertain"><td>End</td><td>uncertain ({{ dw.decision_reason }})</td></tr>
        </tbody>
      </table>
    </section>

    <p class="muted small">
      Click a metric, timeline item, or flag to inspect.
    </p>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useSessionStore } from '@/stores/session';

const session = useSessionStore();
const dw = computed(() => session.session?.drill_window ?? null);
</script>

<style scoped>
.session-overview {
  display: flex;
  flex-direction: column;
  gap: 14px;
}
header h4 {
  margin: 0 0 4px;
  font-size: 1em;
  font-weight: 600;
}
.muted {
  color: var(--color-muted);
}
.small {
  font-size: 0.85em;
}
.counts {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
}
.count-card {
  background: var(--color-bg-elev);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  padding: 8px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
}
.count-card .count {
  font-size: 1.3em;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}
.count-card .label {
  font-size: 0.72em;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--color-muted);
}
h5 {
  margin: 0 0 6px;
  font-size: 0.78em;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--color-muted);
}
.kv {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88em;
}
.kv td {
  padding: 3px 0;
  border-bottom: 1px dashed var(--color-border);
}
.kv td:first-child {
  color: var(--color-muted);
  width: 35%;
}
</style>
