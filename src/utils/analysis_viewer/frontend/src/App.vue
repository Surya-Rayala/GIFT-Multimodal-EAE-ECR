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
import { fetchConfig } from '@/api/client';
import { useSessionStore } from '@/stores/session';
import { useCompareStore } from '@/stores/compare';
import { useUIStore } from '@/stores/ui';

const session = useSessionStore();
const compare = useCompareStore();
const ui = useUIStore();

onMounted(async () => {
  // The path before `?run=` selects the initial view: `/compare[/]` opens
  // Compare, `/analysis[/]` (and the bare `/`) opens Analysis. A deep-linked
  // mode wins over the persisted `viewMode` so a shared link always lands where
  // intended. In desktop/dev the pathname is `/`, leaving the stored mode as-is.
  const pathname = window.location.pathname;
  if (pathname.startsWith('/compare')) ui.viewMode = 'compare';
  else if (pathname.startsWith('/analysis')) ui.viewMode = 'analysis';

  // In LAN web mode, learn the served outputs root so Compare can browse every
  // run under it (and so a deep-linked session's siblings show up).
  try {
    const cfg = await fetchConfig();
    if (cfg.outputs_root) compare.outputsRoot = cfg.outputs_root;
  } catch {
    // desktop mode / endpoint absent — ignore
  }

  // Which session to open. Priority: ?run=…/?session=… in the URL (the host
  // path to a run folder, used for network deep-links) > VITE_INITIAL_SESSION
  // (set by the desktop launcher) > last-loaded path remembered in localStorage.
  const params = new URLSearchParams(window.location.search);
  const fromUrl = params.get('run') || params.get('session');
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
  /* width:100% (not 100vw) avoids a horizontal scrollbar from the vw/scrollbar
   * mismatch; dvh tracks the dynamic viewport so mobile browser chrome (URL
   * bar) doesn't clip the bottom. Falls back to vh on older engines. */
  width: 100%;
  height: 100vh;
  height: 100dvh;
  overflow: hidden;
}
</style>
