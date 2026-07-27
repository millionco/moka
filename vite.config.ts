import { createReadStream } from "node:fs";
import { basename, extname, join } from "node:path";
import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite-plus";

const TEACHER_RUNTIME_DIRECTORY = fileURLToPath(new URL("./public/onnxruntime/", import.meta.url));
const TEACHER_RUNTIME_PREFIX = "/teacher-runtime/";
const WASM_EXTENSION = ".wasm";

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
  fmt: {
    ignorePatterns: ["public/onnxruntime/**"],
  },
  lint: {
    ignorePatterns: ["public/onnxruntime/**"],
    options: { typeAware: true, typeCheck: true },
  },
  plugins: [
    tailwindcss(),
    {
      configureServer: (server) => {
        server.middlewares.use(TEACHER_RUNTIME_PREFIX, (request, response, next) => {
          const requestedName = basename(request.url?.split("?")[0] ?? "");
          const assetPath = join(TEACHER_RUNTIME_DIRECTORY, requestedName);
          response.setHeader(
            "Content-Type",
            extname(requestedName) === WASM_EXTENSION ? "application/wasm" : "text/javascript",
          );
          createReadStream(assetPath).on("error", next).pipe(response);
        });
      },
      name: "teacher-runtime",
    },
  ],
  publicDir: false,
});
