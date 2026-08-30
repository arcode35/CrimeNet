import { CITIES } from "@/lib/domain";
import type { BasemapMode } from "@/stores/explorer-store";

export type ExplorerUrlState = {
  cityId: string;
  timestamp: string;
  horizonHours: number;
  selectedH3: string | null;
  basemapMode: BasemapMode;
};

export function parseExplorerUrl(search: string): Partial<ExplorerUrlState> {
  const params = new URLSearchParams(search);
  const result: Partial<ExplorerUrlState> = {};
  const city = params.get("city");
  if (city && CITIES.some((candidate) => candidate.id === city)) result.cityId = city;
  const time = params.get("time");
  if (time && !Number.isNaN(Date.parse(time))) result.timestamp = new Date(time).toISOString();
  const horizon = Number(params.get("horizon"));
  if ([1, 6, 12, 24].includes(horizon)) result.horizonHours = horizon;
  const cell = params.get("cell");
  if (cell) result.selectedH3 = cell;
  const basemap = params.get("basemap");
  if (basemap === "dark" || basemap === "satellite") result.basemapMode = basemap;
  return result;
}

export function serializeExplorerUrl(state: ExplorerUrlState) {
  const params = new URLSearchParams({
    city: state.cityId,
    time: state.timestamp,
    horizon: String(state.horizonHours),
  });
  if (state.selectedH3) params.set("cell", state.selectedH3);
  if (state.basemapMode !== "dark") params.set("basemap", state.basemapMode);
  return `?${params}`;
}
