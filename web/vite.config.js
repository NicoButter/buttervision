import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: "http://127.0.0.1:7861", changeOrigin: true },
      "/outputs": { target: "http://127.0.0.1:7861", changeOrigin: true },
    },
  },
});
