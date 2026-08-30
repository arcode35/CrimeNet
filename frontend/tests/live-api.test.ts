import { latLngToCell } from "h3-js";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  CrimeNetApiError,
  adaptViewportResponse,
  crimeSenseApiUrl,
  fetchIntensityTimeline,
  getLiveViewport,
  intensityTimelineQueryKey,
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
  it("defaults networking to the public CrimeSense API", () => {
    expect(crimeSenseApiUrl).toBe("https://api.crimesense.ai");
  });

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

  it("includes forecast generation, selected UTC hour, and rounded bounds in freshness", () => {
    expect(liveViewportQueryKey("as-of-a", "valid-a", bounds)).not.toEqual(
      liveViewportQueryKey("as-of-b", "valid-a", bounds),
    );
    expect(liveViewportQueryKey("as-of-a", "valid-a", bounds)).not.toEqual(
      liveViewportQueryKey("as-of-a", "valid-b", bounds),
    );
    expect(liveViewportQueryKey("as-of-a", "valid-a", bounds)).toEqual(
      liveViewportQueryKey("as-of-a", "valid-a", { ...bounds, west: -87.7000001 }),
    );
  });

  it("validates the rolling live plus forecast timeline", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        Response.json({
          schema: "crimenet_intensity_timeline_v1",
          generated_at_utc: "2026-08-30T04:08:35.841156+00:00",
          as_of_utc_hour: "2026-08-30T04:00:00+00:00",
          hours_requested: 24,
          hours_available: 24,
          snapshots: [
            {
              snapshot_id: "20260830T0400",
              valid_utc_hour: "2026-08-30T04:00:00+00:00",
              horizon_hours: 0,
              kind: "live",
            },
            {
              snapshot_id: "20260830T0500",
              valid_utc_hour: "2026-08-30T05:00:00+00:00",
              horizon_hours: 1,
              kind: "forecast",
            },
          ],
          live: {
            snapshot_id: "20260830T0400",
            valid_utc_hour: "2026-08-30T04:00:00+00:00",
          },
        }),
      ),
    );
    const result = await fetchIntensityTimeline();
    expect(result.snapshots).toHaveLength(2);
    expect(result.snapshots[1]).toMatchObject({ kind: "forecast", horizon_hours: 1 });
    expect(intensityTimelineQueryKey).toEqual(["intensity-timeline"]);
  });

  it("maps HTTP 413 to the viewport-too-large state", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("{}", { status: 413 })));
    await expect(
      getLiveViewport("chicago", bounds, "2026-08-29T22:00:00+00:00"),
    ).rejects.toMatchObject({
      kind: "viewport-too-large",
      status: 413,
      message: "Zoom in to load live H3 predictions.",
    } satisfies Partial<CrimeNetApiError>);
  });

  it("sends the exact selected UTC hour with viewport bounds", async () => {
    const validUtcHour = "2026-08-30T10:00:00+00:00";
    const fetchMock = vi.fn().mockResolvedValue(
      Response.json({
        snapshot_id: "20260830T1000",
        valid_utc_hour: validUtcHour,
        resolution: 9,
        aggregation: "native_r9",
        visualization_metric: "mean_r9_events_per_hour",
        count: 1,
        cells: [
          {
            h3,
            events_per_hour: 0.02,
            mean_r9_events_per_hour: 0.02,
            modeled_r9_cells: 1,
          },
        ],
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const result = await getLiveViewport("chicago", bounds, validUtcHour);
    const requested = new URL(String(fetchMock.mock.calls[0][0]));
    expect(requested.origin).toBe("https://api.crimesense.ai");
    expect(requested.searchParams.get("valid_utc_hour")).toBe(validUtcHour);
    expect(requested.searchParams.get("west")).toBe("-87.7");
    expect(result.snapshotId).toBe("20260830T1000");
  });
});

describe("live selected-cell provider", () => {
  it("uses top_k=87 and the same selected forecast hour as the surface", async () => {
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
      validUtcHour: "2026-08-29T22:00:00+00:00",
      asOfUtcHour: "2026-08-29T16:00:00+00:00",
      forecastHorizonHours: 6,
    });
    const requestedUrl = String(fetchMock.mock.calls[0][0]);
    const parsedUrl = new URL(requestedUrl);
    expect(`${parsedUrl.origin}${parsedUrl.pathname}`).toBe(
      `http://serving.test/api/v1/predict/cell/${h3}`,
    );
    expect(parsedUrl.searchParams.get("top_k")).toBe("87");
    expect(parsedUrl.searchParams.get("valid_utc_hour")).toBe(
      "2026-08-29T22:00:00+00:00",
    );
    expect(requestedUrl).not.toContain("timestamp");
    expect(requestedUrl).not.toContain("horizon");
    expect(result.timestamp).toBe("2026-08-29T22:00:00.000Z");
    expect(result.horizonSeconds).toBe(3600);
    expect(result.subtypeDistribution).toHaveLength(87);
    expect(result.provider.label).toBe("FORECAST INFERENCE");
    expect(liveCellPredictionQueryKey("as-of-a", "valid-a", "snapshot-a", h3)).not.toEqual(
      liveCellPredictionQueryKey("as-of-b", "valid-a", "snapshot-a", h3),
    );
    expect(liveCellPredictionQueryKey("as-of-a", "valid-a", "snapshot-a", h3)).not.toEqual(
      liveCellPredictionQueryKey("as-of-a", "valid-b", "snapshot-b", h3),
    );
  });
});
