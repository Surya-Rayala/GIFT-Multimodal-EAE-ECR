<template>
  <div class="app-shell">
    <SplitPane>
      <template #left>
        <LeftPanel />
      </template>
      <template #center>
        <CenterPanel />
      </template>
      <template #right>
        <RightPanel />
      </template>
    </SplitPane>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue';
import SplitPane from '@/components/SplitPane.vue';
import LeftPanel from '@/components/LeftPanel.vue';
import CenterPanel from '@/components/CenterPanel.vue';
import RightPanel from '@/components/RightPanel.vue';
import { useSessionStore } from '@/stores/session';

const session = useSessionStore();

onMounted(() => {
  // Priority: ?session=… in the URL > VITE_INITIAL_SESSION (set by the Python
  // launcher when a path was passed on the CLI) > last successfully-loaded
  // path remembered by LeftPanel via localStorage.
  const params = new URLSearchParams(window.location.search);
  const fromUrl = params.get('session');
  const fromEnv = import.meta.env.VITE_INITIAL_SESSION;
  const fromStorage = (() => {
    try {
      return localStorage.getItem('Vanderbilt-GIFT.AnalysisViewer.recentSession') ?? '';
    } catch {
      return '';
    }
  })();
  const path = fromUrl || fromEnv || fromStorage;
  if (path) {
    void session.load(path);
  }
});
</script>

<style scoped>
.app-shell {
  height: 100vh;
  width: 100vw;
}
</style>
