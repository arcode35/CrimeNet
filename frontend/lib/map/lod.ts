import { cellToLatLng, getResolution } from "h3-js";
import type { PredictionCell } from "@/lib/domain";

export const NATIVE_MARK_RESOLUTION = 9;

export function isNativeMarkCell(h3: string): boolean {
  try {
    return getResolution(h3) === NATIVE_MARK_RESOLUTION;
  } catch {
    return false;
  }
}

export function logIntensityVisualScore(eventsPerR9CellHour: number): number {
  const logRate = Math.log10(Math.max(eventsPerR9CellHour, 1e-5));
  return Math.max(0, Math.min(1, (logRate + 5) / 4));
}

export function getCellVisualizationIntensity(cell: PredictionCell): number | null {
  return cell.visualIntensity ?? cell.intensity;
}

export function getCellVisualizationScore(cell: PredictionCell): number {
  return logIntensityVisualScore(getCellVisualizationIntensity(cell) ?? 0);
}

export type MapCellClickAction =
  | { kind: "select"; h3: string }
  | { kind: "zoom"; center: [longitude: number, latitude: number]; zoom: number };

export function resolveMapCellClick(
  h3: string,
  responseResolution: number,
  currentZoom: number,
): MapCellClickAction {
  if (responseResolution === NATIVE_MARK_RESOLUTION && isNativeMarkCell(h3)) {
    return { kind: "select", h3 };
  }
  const [latitude, longitude] = cellToLatLng(h3);
  return {
    kind: "zoom",
    center: [longitude, latitude],
    zoom: Math.min(currentZoom + 2.5, 14),
  };
}

export function shouldClearSelectionForResolution(
  resolution: number | undefined,
  selectedH3: string | null,
): boolean {
  return Boolean(selectedH3 && resolution !== undefined && resolution < NATIVE_MARK_RESOLUTION);
}
