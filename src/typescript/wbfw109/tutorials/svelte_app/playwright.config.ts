import type { PlaywrightTestConfig } from "@playwright/test";

const config: PlaywrightTestConfig = {
  webServer: {
    command: "yarn run vite build && yarn run vite preview",
    port: 4173,
  },
  testDir: "tests",
};

export default config;
