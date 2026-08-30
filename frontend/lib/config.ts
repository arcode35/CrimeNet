export type CrimeNetDataMode = "fixture" | "api";

export const crimeNetDataMode: CrimeNetDataMode =
  process.env.NEXT_PUBLIC_CRIMENET_DATA_MODE === "api" ? "api" : "fixture";

export const crimeNetApiUrl = (
  process.env.NEXT_PUBLIC_CRIMENET_API_URL || "http://localhost:8000"
).replace(/\/$/, "");

export const isLiveMode = crimeNetDataMode === "api";
export const isFixtureMode = !isLiveMode;
