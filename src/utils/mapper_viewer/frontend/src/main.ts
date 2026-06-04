import { createApp } from 'vue';
import { createPinia } from 'pinia';

import App from './App.vue';
import './theme/theme.css';
import { initSidecarPort } from './api/client';

async function bootstrap(): Promise<void> {
  // Resolve the sidecar port up front so every later API call is sync-fast.
  await initSidecarPort();

  const app = createApp(App);
  app.use(createPinia());
  app.mount('#app');
}

void bootstrap();
