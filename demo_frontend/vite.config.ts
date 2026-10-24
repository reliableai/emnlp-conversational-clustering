/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // @ts-expect-error test is added by vitest
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
  },
});
