<template>
  <aside class="help-card">
    <header>
      <span class="icon" aria-hidden="true">i</span>
      <h4>{{ title }}</h4>
    </header>
    <ol v-if="steps && steps.length" class="steps">
      <li v-for="(s, i) in steps" :key="i">{{ s }}</li>
    </ol>
    <p v-if="why" class="why">
      <strong>Why this matters:</strong> {{ why }}
    </p>
  </aside>
</template>

<script setup lang="ts">
defineProps<{
  title: string;
  steps?: string[];
  why?: string;
}>();
</script>

<style scoped>
.help-card {
  background: var(--color-bg-elev);
  border: 1px solid var(--color-border);
  border-left: 3px solid var(--color-accent);
  border-radius: 6px;
  padding: 12px 14px;
  font-size: 0.9em;
  line-height: 1.45;
}
.help-card header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}
.help-card .icon {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: var(--color-accent);
  color: var(--color-bg);
  font-weight: 700;
  font-style: italic;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.85em;
  font-family: serif;
}
.help-card h4 {
  margin: 0;
  font-size: 0.95em;
  font-weight: 600;
}
.help-card .steps {
  margin: 0;
  padding-left: 20px;
}
.help-card .steps li {
  margin-bottom: 4px;
  color: var(--color-text-secondary);
}
.help-card .why {
  margin: 10px 0 0;
  padding: 8px 10px;
  background: var(--color-bg-deep);
  border-radius: 4px;
  font-size: 0.88em;
  color: var(--color-muted);
}
</style>
