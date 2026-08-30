import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { latLngToCell } from "h3-js";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Inspector } from "@/components/explorer/inspector";
import { Providers } from "@/components/providers";
import { predictionResponseSchema } from "@/lib/domain";
import * as geocoding from "@/lib/geocoding";
import { inferenceProvider } from "@/lib/inference";
import { useExplorerStore } from "@/stores/explorer-store";

vi.mock("@/lib/geocoding", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/geocoding")>()),
  reverseCellLocation: vi.fn(),
}));

const reverseCellLocation = vi.mocked(geocoding.reverseCellLocation);

const h3 = latLngToCell(41.8781, -87.6298, 9);
const base = {
  city: "chicago",
  timestamp: "2024-08-21T22:00:00.000Z",
  horizonHours: 1,
  unit: "events_per_cell_hour",
  modelVersion: "test",
} as const;

describe("cell inspector", () => {
  beforeEach(() => {
    reverseCellLocation.mockReset();
    reverseCellLocation.mockResolvedValue({
      label: "Chicago, Illinois",
      primaryLabel: "Chicago",
      secondaryLabel: "Illinois",
      longitude: -87.6298,
      latitude: 41.8781,
      source: "reverse-geocoder",
    });
    useExplorerStore.setState({ cityId: "chicago", selectedH3: h3 });
  });

  const renderInspector = (data: ReturnType<typeof predictionResponseSchema.parse>) =>
    render(
      <Providers>
        <Inspector data={data} />
      </Providers>,
    );

  it("renders valid hierarchical prediction values", async () => {
    const data = predictionResponseSchema.parse({
      ...base,
      cells: [
        {
          h3,
          intensity: 0.125,
          percentile: 0.8,
          coverage: "full",
          missingReason: null,
          features: [],
        },
      ],
    });
    renderInspector(data);
    expect(await screen.findByText("EVENT INTENSITY")).toBeInTheDocument();
    expect(screen.getByText("CRIME MIX")).toBeInTheDocument();
    expect(screen.getByText("FIXTURE DATA")).toBeInTheDocument();
    expect(screen.getByText("Full model")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Larceny \/ Theft/ })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "INTENSITY" }));
    expect(screen.getByText("λfamily · events / cell / hour")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "VIEW ALL 87 →" }));
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText("All 87 modeled crime types")).toBeInTheDocument();
    fireEvent.change(within(dialog).getByPlaceholderText("Search modeled crime type..."), {
      target: { value: "theft from vehicle" },
    });
    expect(within(dialog).getByText("Theft From Vehicle")).toBeInTheDocument();
  });

  it("renders an explicit no-prediction state for unsupported cells", async () => {
    const data = predictionResponseSchema.parse({
      ...base,
      cells: [
        {
          h3,
          intensity: null,
          percentile: null,
          coverage: "unsupported",
          missingReason: "History unavailable",
          features: [],
        },
      ],
    });
    renderInspector(data);
    expect(
      await screen.findByText(/Missing data is not interpreted as zero intensity/),
    ).toBeInTheDocument();
    expect(screen.queryByText("0.000")).not.toBeInTheDocument();
    expect(screen.queryByText("CRIME MIX")).not.toBeInTheDocument();
  });

  it("does not request or render mark inference for a coarse LOD response", () => {
    const coarseH3 = latLngToCell(41.8781, -87.6298, 6);
    useExplorerStore.setState({ selectedH3: coarseH3 });
    const data = predictionResponseSchema.parse({
      ...base,
      resolution: 6,
      aggregation: "sum_r9_child_intensity",
      cells: [
        {
          h3: coarseH3,
          intensity: 2.4,
          visualIntensity: 0.0012,
          modeledR9Cells: 2000,
          percentile: null,
          coverage: "full",
          missingReason: null,
          features: [],
        },
      ],
    });
    renderInspector(data);
    expect(screen.queryByText("H3 CELL INSPECTOR")).not.toBeInTheDocument();
    expect(screen.queryByText("CRIME MIX")).not.toBeInTheDocument();
  });

  it("displays the clicked cell location instead of the sidebar city", async () => {
    const losAngelesH3 = latLngToCell(34.0522, -118.2437, 9);
    reverseCellLocation.mockResolvedValue({
      label: "Los Angeles, California",
      primaryLabel: "Los Angeles",
      secondaryLabel: "California",
      longitude: -118.2437,
      latitude: 34.0522,
      source: "reverse-geocoder",
    });
    useExplorerStore.setState({ cityId: "chicago", selectedH3: losAngelesH3 });
    const data = predictionResponseSchema.parse({
      ...base,
      cells: [
        {
          h3: losAngelesH3,
          intensity: 0.125,
          percentile: 0.8,
          coverage: "full",
          missingReason: null,
          features: [],
        },
      ],
    });
    renderInspector(data);

    expect(await screen.findByText(/Los Angeles, California/)).toBeInTheDocument();
    expect(screen.queryByText(/Chicago ·/)).not.toBeInTheDocument();

    useExplorerStore.setState({ cityId: "seattle" });
    expect(screen.getByText(/Los Angeles, California/)).toBeInTheDocument();
    expect(screen.queryByText(/Seattle ·/)).not.toBeInTheDocument();
    expect(reverseCellLocation).toHaveBeenCalledTimes(1);
  });

  it("keeps clicked-cell location and exact forecast hour aligned", async () => {
    const losAngelesH3 = latLngToCell(34.0522, -118.2437, 9);
    const validUtcHour = "2026-08-31T04:00:00.000Z";
    reverseCellLocation.mockResolvedValue({
      label: "Los Angeles, California",
      primaryLabel: "Los Angeles",
      secondaryLabel: "California",
      longitude: -118.2437,
      latitude: 34.0522,
      source: "reverse-geocoder",
    });
    useExplorerStore.setState({ cityId: "chicago", selectedH3: losAngelesH3 });
    const predictionSpy = vi.spyOn(inferenceProvider, "getCellPrediction");
    const data = predictionResponseSchema.parse({
      ...base,
      timestamp: validUtcHour,
      horizonHours: 6,
      cells: [
        {
          h3: losAngelesH3,
          intensity: 0.125,
          percentile: 0.8,
          coverage: "full",
          missingReason: null,
          features: [],
        },
      ],
    });
    const rendered = render(
      <Providers>
        <Inspector
          data={data}
          asOfUtcHour="2026-08-30T22:00:00.000Z"
          selectedSnapshot={{
            snapshot_id: "forecast-plus-6",
            valid_utc_hour: validUtcHour,
            horizon_hours: 6,
            kind: "forecast",
          }}
        />
      </Providers>,
    );

    expect(await screen.findByText(/Los Angeles, California/)).toBeInTheDocument();
    expect(predictionSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        h3: losAngelesH3,
        validUtcHour,
        forecastHorizonHours: 6,
      }),
    );

    const nextValidUtcHour = "2026-08-31T05:00:00.000Z";
    const nextData = predictionResponseSchema.parse({
      ...data,
      timestamp: nextValidUtcHour,
    });
    rendered.rerender(
      <Providers>
        <Inspector
          data={nextData}
          asOfUtcHour="2026-08-30T22:00:00.000Z"
          selectedSnapshot={{
            snapshot_id: "forecast-plus-7",
            valid_utc_hour: nextValidUtcHour,
            horizon_hours: 7,
            kind: "forecast",
          }}
        />
      </Providers>,
    );

    await waitFor(() =>
      expect(predictionSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          h3: losAngelesH3,
          validUtcHour: nextValidUtcHour,
          forecastHorizonHours: 7,
        }),
      ),
    );
    expect(useExplorerStore.getState().selectedH3).toBe(losAngelesH3);
    predictionSpy.mockRestore();
  });

  it("uses coordinates for an unknown cell instead of the sidebar city", async () => {
    const unknownH3 = latLngToCell(34.42, -118.58, 9);
    reverseCellLocation.mockResolvedValue(geocoding.coordinateCellLocation(-118.58, 34.42));
    useExplorerStore.setState({ cityId: "chicago", selectedH3: unknownH3 });
    const data = predictionResponseSchema.parse({
      ...base,
      cells: [
        {
          h3: unknownH3,
          intensity: null,
          percentile: null,
          coverage: "unsupported",
          missingReason: "Outside supported features",
          features: [],
        },
      ],
    });
    renderInspector(data);

    expect(await screen.findByText(/34\.4200° N, 118\.5800° W/)).toBeInTheDocument();
    expect(screen.queryByText(/Chicago ·/)).not.toBeInTheDocument();
  });
});
