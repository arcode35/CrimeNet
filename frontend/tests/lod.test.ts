import { latLngToCell } from "h3-js";
import { describe, expect, it } from "vitest";
import { predictionCellSchema } from "@/lib/domain";
import {
  getCellVisualizationIntensity,
  getCellVisualizationScore,
  logIntensityVisualScore,
  resolveMapCellClick,
  shouldClearSelectionForResolution,
} from "@/lib/map/lod";

const cell = (intensity: number, visualIntensity: number) =>
  predictionCellSchema.parse({
    h3: latLngToCell(29.76, -95.37, 5),
    intensity,
    visualIntensity,
    modeledR9Cells: 15000,
    percentile: null,
    coverage: "full",
    missingReason: null,
    features: [],
  });

describe("zoom-aware H3 rendering semantics", () => {
  it("uses mean r9 intensity rather than summed parent intensity for color", () => {
    const coarse = cell(12.5, 0.0002);
    expect(getCellVisualizationIntensity(coarse)).toBe(0.0002);
    expect(getCellVisualizationScore(coarse)).toBe(logIntensityVisualScore(0.0002));
    expect(getCellVisualizationScore(coarse)).not.toBe(logIntensityVisualScore(12.5));
  });

  it("turns a coarse-cell click into camera drill-down without selecting it", () => {
    const coarseH3 = latLngToCell(29.76, -95.37, 5);
    const action = resolveMapCellClick(coarseH3, 5, 5);
    expect(action).toMatchObject({ kind: "zoom", zoom: 7.5 });
    expect(action).not.toHaveProperty("h3");
  });

  it("permits selection only for a confirmed r9 response and r9 cell", () => {
    const nativeH3 = latLngToCell(29.76, -95.37, 9);
    expect(resolveMapCellClick(nativeH3, 9, 12)).toEqual({ kind: "select", h3: nativeH3 });
    expect(resolveMapCellClick(nativeH3, 8, 10).kind).toBe("zoom");
  });

  it("clears a stale selection when the active response becomes coarse", () => {
    const selected = latLngToCell(29.76, -95.37, 9);
    expect(shouldClearSelectionForResolution(6, selected)).toBe(true);
    expect(shouldClearSelectionForResolution(9, selected)).toBe(false);
    expect(shouldClearSelectionForResolution(6, null)).toBe(false);
  });
});
