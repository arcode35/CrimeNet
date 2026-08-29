import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  fullyParallel: false,
  reporter: "line",
  use: {
    baseURL: "http://127.0.0.1:3100",
    channel: "chrome",
    colorScheme: "dark",
  },
  projects: [
    { name: "desktop-1920", use: { viewport: { width: 1920, height: 1080 } } },
    { name: "desktop-1440", use: { viewport: { width: 1440, height: 900 } } },
    { name: "laptop-1280", use: { viewport: { width: 1280, height: 800 } } },
    { name: "tablet", use: { viewport: { width: 820, height: 1180 }, isMobile: true } },
    { name: "mobile", use: { viewport: { width: 390, height: 844 }, isMobile: true } },
  ],
  webServer: {
    command: "npm run start -- --port 3100",
    url: "http://127.0.0.1:3100",
    reuseExistingServer: true,
    timeout: 30_000,
  },
});
