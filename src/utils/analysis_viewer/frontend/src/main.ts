import { createApp } from 'vue';
import { createPinia } from 'pinia';
import App from './App.vue';
import { initSidecarPort } from '@/api/client';
import './theme/theme.css';

async function bootstrap(): Promise<void> {
  // Resolve the sidecar port before mounting so the very first /session call
  // hits the right URL. Fails open: any error here just falls through to the
  // 8765 dev default inside `sidecarBaseUrl()`.
  try {
    await initSidecarPort();
  } catch (err) {
    console.warn('[analysis-viewer] sidecar port init failed:', err);
  }

  const app = createApp(App);
  app.use(createPinia());
  app.mount('#app');
}

void bootstrap();
