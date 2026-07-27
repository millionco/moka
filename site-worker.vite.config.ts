import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const SITE_WORKER_OUTPUT_DIRECTORY = fileURLToPath(new URL("./public/models", import.meta.url));
const SITE_WORKER_OUTPUT_FILE_NAME = "moka-worker.js";

export default defineConfig({
  build: {
    emptyOutDir: false,
    lib: {
      entry: fileURLToPath(new URL("./web/worker.ts", import.meta.url)),
      fileName: "moka-worker",
      formats: ["es"],
    },
    minify: true,
    outDir: SITE_WORKER_OUTPUT_DIRECTORY,
    rollupOptions: {
      output: {
        entryFileNames: SITE_WORKER_OUTPUT_FILE_NAME,
      },
    },
  },
  publicDir: false,
});
