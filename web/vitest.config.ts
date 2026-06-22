import { defineConfig } from "vitest/config";

// Headless DOM tests (jsdom) for the vanilla-TS UI modules. There is no browser in CI;
// jsdom gives ConfigPanel a real document to build into and dispatch events against.
export default defineConfig({
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.ts"],
  },
});
