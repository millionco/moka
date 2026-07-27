import { createReadStream } from "node:fs";
import { basename, extname, join } from "node:path";
import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

const TEACHER_MODEL_DIRECTORY = fileURLToPath(new URL("./teachers/", import.meta.url));
const TEACHER_MODEL_PREFIX = "/teacher-model/";
const TEACHER_RUNTIME_DIRECTORY = fileURLToPath(
  new URL("./node_modules/onnxruntime-web/dist/", import.meta.url),
);
const TEACHER_RUNTIME_PREFIX = "/teacher-runtime/";
const WASM_EXTENSION = ".wasm";

const serveDirectory =
  (directory: string, contentType: (requestedName: string) => string) =>
  (
    request: { url?: string },
    response: {
      setHeader: (name: string, value: string) => void;
    },
    next: (error?: Error) => void,
  ) => {
    const requestedName = basename(request.url?.split("?")[0] ?? "");
    const assetPath = join(directory, requestedName);
    response.setHeader("Content-Type", contentType(requestedName));
    createReadStream(assetPath).on("error", next).pipe(response);
  };

export default defineConfig({
  build: {
    outDir: "web-dist",
    rollupOptions: {
      input: {
        arena: fileURLToPath(new URL("./web/arena.html", import.meta.url)),
        benchmark: fileURLToPath(new URL("./web/benchmark.html", import.meta.url)),
      },
    },
  },
  plugins: [
    tailwindcss(),
    {
      configureServer: (server) => {
        server.middlewares.use(
          TEACHER_RUNTIME_PREFIX,
          serveDirectory(TEACHER_RUNTIME_DIRECTORY, (requestedName) =>
            extname(requestedName) === WASM_EXTENSION ? "application/wasm" : "text/javascript",
          ),
        );
        server.middlewares.use(
          TEACHER_MODEL_PREFIX,
          serveDirectory(TEACHER_MODEL_DIRECTORY, () => "application/octet-stream"),
        );
      },
      name: "teacher-assets",
    },
  ],
  publicDir: false,
});
