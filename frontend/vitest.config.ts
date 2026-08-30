import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";

export default defineConfig({
  resolve: { alias: { "@": fileURLToPath(new URL("./", import.meta.url)) } },
  test: {
    env: { NEXT_PUBLIC_CRIMENET_DATA_MODE: "fixture" },
    environment: "jsdom",
    include: ["tests/**/*.test.{ts,tsx}"],
    setupFiles: ["./vitest.setup.ts"],
    pool: "threads",
    fileParallelism: false,
    maxWorkers: 1,
  },
});
