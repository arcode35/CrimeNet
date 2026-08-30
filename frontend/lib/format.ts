import type { CoverageStatus } from "@/lib/domain";

export const formatIntensity = (value: number | null) =>
  value === null ? "No prediction" : value < 0.01 ? value.toFixed(4) : value.toFixed(3);

export const formatDetailedIntensity = (value: number | null) =>
  value === null
    ? "No prediction"
    : value < 0.0001
      ? value.toFixed(6)
      : value < 0.01
        ? value.toFixed(5)
        : value.toFixed(3);

export const formatPercent = (value: number | null) =>
  value === null ? "—" : `${Math.round(value * 100)}th`;

export const coverageLabel: Record<CoverageStatus, string> = {
  full: "Full model",
  partial: "Limited coverage",
  unsupported: "Inference unavailable",
};

export const featureLabels: Record<string, string> = {
  temporal: "Temporal",
  lighting: "Lighting",
  weather: "Weather",
  osm: "Built environment",
  socioeconomic: "Socioeconomic",
  crime_history: "Historical crime",
  neighbor_history: "Neighbor crime history",
};

export function formatTimestamp(timestamp: string, timezone: string, includeDate = true) {
  return new Intl.DateTimeFormat("en-US", {
    ...(includeDate ? { month: "short", day: "numeric", year: "numeric" } : {}),
    hour: "numeric",
    minute: "2-digit",
    hour12: false,
    timeZone: timezone,
    timeZoneName: "short",
  }).format(new Date(timestamp));
}
