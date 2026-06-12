import { defineConfig } from "vitest/config";

// Parity tests run in Node so they can read the golden-vector fixture from disk
// via fs (keeping the 857 KB file out of the app bundle and out of tsc's literal
// type-checking path).
export default defineConfig({
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
