import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import { fileURLToPath, URL } from 'node:url';

// Port 5175 — analysis_viewer uses 5173, mapper_viewer 5174. Running them
// simultaneously needs disjoint ports; the Tauri shell honors this via
// tauri.conf.json devUrl.
export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    port: 5175,
    strictPort: false,
  },
});
