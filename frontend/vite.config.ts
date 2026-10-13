import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, Vite serves the UI on :5173 and forwards /api to the FastAPI backend on :8000.
// `npm run build` writes to dist/, which FastAPI serves directly in production.
export default defineConfig({
  plugins: [react()],
  // three.js alone is ~700 kB minified; one chunk is fine for this app.
  build: { chunkSizeWarningLimit: 1500 },
  server: {
    port: 5173,
    proxy: { "/api": "http://localhost:8000" },
  },
});
