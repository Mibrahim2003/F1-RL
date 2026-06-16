import { defineConfig } from "vite";

// Backend (FastAPI) runs on 127.0.0.1:8000. The dev server proxies all
// engine routes so the frontend can use same-origin paths (/track, /api,
// /recordings, /ws) with no CORS handling.
const BACKEND = "http://127.0.0.1:8000";

export default defineConfig({
  root: ".",
  server: {
    port: 5173,
    proxy: {
      "/track": { target: BACKEND, changeOrigin: true },
      "/api": { target: BACKEND, changeOrigin: true },
      "/recordings": { target: BACKEND, changeOrigin: true },
      "/ws": { target: BACKEND, changeOrigin: true, ws: true },
    },
  },
  build: {
    target: "es2022",
    sourcemap: true,
  },
});
