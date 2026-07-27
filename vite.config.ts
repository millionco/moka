import { defineConfig } from "vite-plus";

export default defineConfig({
  pack: [
    {
      clean: true,
      dts: true,
      entry: ["./src/index.ts"],
      format: ["esm"],
      minify: true,
      platform: "browser",
      sourcemap: false,
    },
    {
      clean: false,
      dts: false,
      entry: ["./src/worker.ts"],
      format: ["esm"],
      minify: true,
      platform: "browser",
      sourcemap: false,
    },
  ],
});
