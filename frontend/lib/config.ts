export type CrimeNetDataMode = "fixture" | "api";

export const crimeNetDataMode: CrimeNetDataMode =
  process.env.NEXT_PUBLIC_CRIMENET_DATA_MODE === "fixture" ? "fixture" : "api";

export const DEFAULT_CRIMESENSE_API_URL = "https://api.crimesense.ai";

export const crimeSenseApiUrl = (
  process.env.NEXT_PUBLIC_API_BASE_URL || DEFAULT_CRIMESENSE_API_URL
).replace(/\/$/, "");

export const isLiveMode = crimeNetDataMode === "api";
export const isFixtureMode = !isLiveMode;
