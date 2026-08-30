import { render, screen } from "@testing-library/react";
import { latLngToCell } from "h3-js";
import { describe, expect, it } from "vitest";
import { MapLegend } from "@/components/explorer/map-legend";
import { predictionResponseSchema } from "@/lib/domain";
import { useExplorerStore } from "@/stores/explorer-store";

describe("LOD map legend", () => {
  it("labels mean-r9 color semantics and the backend-selected resolution", () => {
    useExplorerStore.setState((state) => ({
      layers: { ...state.layers, coverage: false, prediction: true },
    }));
    const data = predictionResponseSchema.parse({
      city: "houston",
      timestamp: "2026-08-30T00:00:00.000Z",
      horizonHours: 1,
      unit: "events_per_cell_hour",
      modelVersion: "live/snapshot",
      snapshotId: "snapshot",
      source: "live",
      resolution: 5,
      aggregation: "sum_r9_child_intensity",
      visualizationMetric: "mean_r9_events_per_hour",
      cells: [
        {
          h3: latLngToCell(29.76, -95.37, 5),
          intensity: 1.381,
          visualIntensity: 0.000091,
          modeledR9Cells: 15176,
          percentile: null,
          coverage: "full",
          missingReason: null,
          features: [],
        },
      ],
    });
    render(<MapLegend data={data} error={null} />);
    expect(screen.getByText("LOCAL MODEL INTENSITY")).toBeInTheDocument();
    expect(screen.getByText("events / r9-equivalent cell / hour")).toBeInTheDocument();
    expect(screen.getByText("H3-R5")).toBeInTheDocument();
    expect(screen.getByText("Aggregated from modeled r9 cells")).toBeInTheDocument();
  });
});
