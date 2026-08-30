import { latLngToCell } from "h3-js";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  CrimeNetApiError,
  adaptViewportResponse,
  getLiveViewport,
  liveViewportQueryKey,
} from "@/lib/api";
import { ApiInferenceProvider } from "@/lib/inference/api-provider";
import { liveCellPredictionQueryKey } from "@/lib/inference";
import { CRIME_SUBTYPES } from "@/lib/taxonomy";

const h3 = latLngToCell(41.8781, -87.6298, 9);
const coarseH3 = latLngToCell(29.76, -95.37, 5);
const bounds = { west: -87.7, south: 41.84, east: -87.6, north: 41.92, zoom: 11.4 };

afterEach(() => vi.unstubAllGlobals());

describe("live viewport adapter", () => {
  it("keeps coarse aggregate intensity distinct from its visualization intensity", () => {
    const result = adaptViewportResponse("chicago", {
      snapshot_id: "20260829T2200",
      valid_utc_hour: "2026-08-29T22:00:00+00:00",
      resolution: 5,
      aggregation: "sum_r9_child_intensity",
      visualization_metric: "mean_r9_events_per_hour",
      candidate_count: 2,
      count: 1,
      cells: [
        {
          h3: coarseH3,
          events_per_hour: 1.381,
          mean_r9_events_per_hour: 0.000091,
          modeled_r9_cells: 15176,
        },
      ],
    });
    expect(result).toMatchObject({
      city: "chicago",
      timestamp: "2026-08-29T22:00:00.000Z",
      horizonHours: 1,
      snapshotId: "20260829T2200",
      source: "live",
      resolution: 5,
      aggregation: "sum_r9_child_intensity",
      visualizationMetric: "mean_r9_events_per_hour",
      candidateCount: 2,
    });
    expect(result.cells[0]).toMatchObject({
      h3: coarseH3,
      intensity: 1.381,
      visualIntensity: 0.000091,
      modeledR9Cells: 15176,
      percentile: null,
      coverage: "full",
    });
  });

  it("includes snapshot and rounded bounds in viewport query freshness", () => {
    expect(liveViewportQueryKey("snapshot-a", bounds)).not.toEqual(
      liveViewportQueryKey("snapshot-b", bounds),
    );
    expect(liveViewportQueryKey("snapshot-a", bounds)).toEqual(
      liveViewportQueryKey("snapshot-a", { ...bounds, west: -87.7000001 }),
    );
  });

  it("maps HTTP 413 to the viewport-too-large state", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("{}", { status: 413 })));
    await expect(getLiveViewport("chicago", bounds)).rejects.toMatchObject({
      kind: "viewport-too-large",
      status: 413,
      message: "Zoom in to load live H3 predictions.",
    } satisfies Partial<CrimeNetApiError>);
  });
});

describe("live selected-cell provider", () => {
  it("uses top_k=87, keys by snapshot, and sends no fixture-era time inputs", async () => {
    const totalIntensity = 0.087;
    const response = {
      h3,
      snapshot_id: "20260829T2200",
      valid_utc_hour: "2026-08-29T22:00:00+00:00",
      intensity: {
        log_intensity: -10,
        events_per_second: totalIntensity / 3600,
        events_per_hour: totalIntensity,
      },
      mark: {
        model_run_id: "run-1",
        num_classes: 87,
        labels_available: true,
        distribution: [...CRIME_SUBTYPES].reverse().map((item) => ({
          class_id: CRIME_SUBTYPES.indexOf(item),
          subtype: item.subtypeKey,
          probability: 1 / 87,
          events_per_hour: totalIntensity / 87,
        })),
      },
      center: { lat: 41.8781, lon: -87.6298 },
    };
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(response), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const provider = new ApiInferenceProvider("http://serving.test");
    const result = await provider.getCellPrediction({
      cityId: "chicago",
      h3,
      timestamp: "2024-01-01T00:00:00.000Z",
      horizonHours: 24,
      snapshotId: "20260829T2200",
    });
    const requestedUrl = String(fetchMock.mock.calls[0][0]);
    expect(requestedUrl).toBe(`http://serving.test/api/v1/predict/cell/${h3}?top_k=87`);
    expect(requestedUrl).not.toContain("timestamp");
    expect(requestedUrl).not.toContain("horizon");
    expect(result.timestamp).toBe("2026-08-29T22:00:00.000Z");
    expect(result.horizonSeconds).toBe(3600);
    expect(result.subtypeDistribution).toHaveLength(87);
    expect(liveCellPredictionQueryKey("snapshot-a", h3)).not.toEqual(
      liveCellPredictionQueryKey("snapshot-b", h3),
    );
  });
});
