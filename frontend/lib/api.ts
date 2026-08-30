import { cellToLatLng, gridDisk, gridDistance, latLngToCell } from "h3-js";
import { z } from "zod";
import { crimeNetApiUrl, isFixtureMode } from "@/lib/config";
import {
  CITIES,
  getCity,
  modelMetadataSchema,
  predictionResponseSchema,
  type FeatureAvailability,
  type ModelMetadata,
  type PredictionResponse,
} from "@/lib/domain";

export { crimeNetApiUrl, isFixtureMode, isLiveMode } from "@/lib/config";

export type ViewportBounds = {
  west: number;
  south: number;
  east: number;
  north: number;
  zoom: number;
};

export type CrimeNetApiErrorKind =
  | "network"
  | "bad-request"
  | "not-found"
  | "viewport-too-large"
  | "server"
  | "contract";

export class CrimeNetApiError extends Error {
  constructor(
    message: string,
    readonly kind: CrimeNetApiErrorKind,
    readonly status?: number,
  ) {
    super(message);
    this.name = "CrimeNetApiError";
  }
}

export const serviceHealthSchema = z.object({
  status: z.string(),
  snapshot_id: z.string().min(1),
  valid_utc_hour: z.string().datetime({ offset: true }),
  cells: z.number().int().nonnegative(),
  mark_model: z.object({
    status: z.string(),
    run_id: z.string().optional(),
    classes: z.number().int().nonnegative().optional(),
    labels_available: z.boolean().optional(),
  }),
});

const viewportResponseSchema = z.object({
  snapshot_id: z.string().min(1),
  valid_utc_hour: z.string().datetime({ offset: true }),
  resolution: z.number().int().min(4).max(9),
  aggregation: z.enum(["native_r9", "sum_r9_child_intensity"]),
  visualization_metric: z.literal("mean_r9_events_per_hour"),
  candidate_count: z.number().int().nonnegative().optional(),
  count: z.number().int().nonnegative(),
  cells: z.array(
    z.object({
      h3: z.string().min(15),
      events_per_hour: z.number().nonnegative(),
      mean_r9_events_per_hour: z.number().nonnegative(),
      modeled_r9_cells: z.number().int().positive(),
    }),
  ),
});

export type ServiceHealth = z.infer<typeof serviceHealthSchema>;
export type LiveViewportResponse = z.infer<typeof viewportResponseSchema>;

function errorKindForStatus(status: number): CrimeNetApiErrorKind {
  if (status === 400) return "bad-request";
  if (status === 404) return "not-found";
  if (status === 413) return "viewport-too-large";
  if (status >= 500) return "server";
  return "bad-request";
}

export async function fetchCrimeNetJson(path: string, signal?: AbortSignal): Promise<unknown> {
  let response: Response;
  try {
    response = await fetch(`${crimeNetApiUrl}${path}`, { signal });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new CrimeNetApiError("Unable to reach the CrimeNet serving API.", "network");
  }
  if (!response.ok) {
    const kind = errorKindForStatus(response.status);
    const message =
      kind === "viewport-too-large"
        ? "Zoom in to load live H3 predictions."
        : kind === "not-found"
          ? "No live prediction coverage was found for this area."
          : `CrimeNet serving API returned ${response.status}.`;
    throw new CrimeNetApiError(message, kind, response.status);
  }
  try {
    return await response.json();
  } catch {
    throw new CrimeNetApiError("CrimeNet returned an invalid JSON response.", "contract");
  }
}

function parseContract<T>(schema: z.ZodType<T>, value: unknown, label: string): T {
  const parsed = schema.safeParse(value);
  if (!parsed.success) {
    throw new CrimeNetApiError(`${label} did not match the frontend contract.`, "contract");
  }
  return parsed.data;
}

export async function getServiceHealth(signal?: AbortSignal): Promise<ServiceHealth> {
  const raw = await fetchCrimeNetJson("/health", signal);
  return parseContract(serviceHealthSchema, raw, "Health response");
}

export const serviceHealthQueryKey = ["crime-net-health"] as const;

export function roundViewportBounds(bounds: ViewportBounds): ViewportBounds {
  const round = (value: number) => Number(value.toFixed(4));
  return {
    west: round(bounds.west),
    south: round(bounds.south),
    east: round(bounds.east),
    north: round(bounds.north),
    zoom: Number(bounds.zoom.toFixed(2)),
  };
}

export const liveViewportQueryKey = (snapshotId: string, bounds: ViewportBounds) => {
  const rounded = roundViewportBounds(bounds);
  return [
    "live-intensity",
    snapshotId,
    rounded.west,
    rounded.south,
    rounded.east,
    rounded.north,
  ] as const;
};

export function adaptViewportResponse(cityId: string, raw: unknown): PredictionResponse {
  const response = parseContract(viewportResponseSchema, raw, "Viewport response");
  if (response.count !== response.cells.length) {
    throw new CrimeNetApiError("Viewport response count did not match its cells.", "contract");
  }
  return predictionResponseSchema.parse({
    city: cityId,
    timestamp: new Date(response.valid_utc_hour).toISOString(),
    horizonHours: 1,
    unit: "events_per_cell_hour",
    modelVersion: `live/${response.snapshot_id}`,
    snapshotId: response.snapshot_id,
    source: "live",
    resolution: response.resolution,
    aggregation: response.aggregation,
    visualizationMetric: response.visualization_metric,
    candidateCount: response.candidate_count,
    cells: response.cells.map((cell) => ({
      h3: cell.h3,
      intensity: cell.events_per_hour,
      percentile: null,
      visualIntensity: cell.mean_r9_events_per_hour,
      modeledR9Cells: cell.modeled_r9_cells,
      coverage: "full",
      missingReason: null,
      features: [],
    })),
  });
}

export async function getLiveViewport(
  cityId: string,
  bounds: ViewportBounds,
  signal?: AbortSignal,
): Promise<PredictionResponse> {
  const rounded = roundViewportBounds(bounds);
  const params = new URLSearchParams({
    west: String(rounded.west),
    south: String(rounded.south),
    east: String(rounded.east),
    north: String(rounded.north),
  });
  const raw = await fetchCrimeNetJson(`/api/v1/intensity/viewport?${params}`, signal);
  return adaptViewportResponse(cityId, raw);
}

export const predictionQueryKey = (cityId: string, timestamp: string, horizonHours: number) =>
  ["predictions", cityId, timestamp, horizonHours] as const;

const allFeatures = (status: "full" | "partial" | "unsupported"): FeatureAvailability[] => {
  const missing =
    status === "partial"
      ? ["neighbor_history"]
      : status === "unsupported"
        ? ["crime_history", "neighbor_history"]
        : [];
  return [
    "temporal",
    "lighting",
    "weather",
    "osm",
    "socioeconomic",
    "crime_history",
    "neighbor_history",
  ].map((group) => ({
    group: group as FeatureAvailability["group"],
    available: !missing.includes(group),
  }));
};

function fixturePredictions(
  cityId: string,
  timestamp: string,
  horizonHours: number,
): PredictionResponse {
  const city = getCity(cityId);
  const center = latLngToCell(city.center[1], city.center[0], 9);
  const time = Math.floor(new Date(timestamp).getTime() / 3_600_000);
  const radius = cityId === "new_york" ? 26 : 22;
  const gaussian = (x: number, y: number, cx: number, cy: number, spread: number) =>
    Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / spread);
  const cells = gridDisk(center, radius).map((h3) => {
    const [latitude, longitude] = cellToLatLng(h3);
    const distance = gridDistance(center, h3);
    const x =
      ((longitude - city.center[0]) * Math.cos((city.center[1] * Math.PI) / 180) * 111) /
      (radius * 0.175);
    const y = ((latitude - city.center[1]) * 111) / (radius * 0.175);
    const phase = time * 0.055;
    const boundary = radius - 2 + Math.sin(Math.atan2(y, x) * 5 + cityId.length) * 1.7;
    const coverage = distance > boundary ? "unsupported" : "full";
    const g1 = gaussian(x, y, -0.34 + Math.sin(phase) * 0.08, 0.18, 0.16);
    const g2 = gaussian(x, y, 0.3, -0.28 + Math.cos(phase * 0.8) * 0.09, 0.11);
    const g3 = gaussian(x, y, 0.2 + Math.sin(phase * 0.6) * 0.12, 0.38, 0.08);
    const corridor = Math.exp(-((y + x * 0.34 - 0.03) ** 2) / 0.045) * 0.022;
    const temporalPulse = 0.008 * (0.5 + 0.5 * Math.sin(phase * 2 + x * 2.2 - y));
    const intensity =
      coverage === "unsupported"
        ? null
        : Number(
            (0.007 + g1 * 0.13 + g2 * 0.105 + g3 * 0.075 + corridor + temporalPulse).toFixed(5),
          );
    return {
      h3,
      intensity,
      percentile: intensity === null ? null : Math.min(0.999, Math.max(0.03, intensity / 0.16)),
      visualIntensity: intensity,
      modeledR9Cells: intensity === null ? 0 : 1,
      coverage,
      missingReason:
        coverage === "unsupported"
          ? "Historical feature coverage is unavailable for this H3 cell."
          : null,
      features: allFeatures(coverage),
    };
  });
  return predictionResponseSchema.parse({
    city: cityId,
    timestamp,
    horizonHours,
    unit: "events_per_cell_hour",
    modelVersion: "fixture/full-v1-contract",
    source: "fixture",
    resolution: 9,
    cells,
  });
}

export async function getPredictions(
  cityId: string,
  timestamp: string,
  horizonHours: number,
  signal?: AbortSignal,
) {
  if (!isFixtureMode) {
    throw new CrimeNetApiError("Fixture prediction path is unavailable in live mode.", "contract");
  }
  await new Promise<void>((resolve, reject) => {
    const timer = setTimeout(resolve, 120);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      },
      { once: true },
    );
  });
  return fixturePredictions(cityId, timestamp, horizonHours);
}

export async function getModelMetadata(signal?: AbortSignal): Promise<ModelMetadata> {
  if (isFixtureMode) {
    return modelMetadataSchema.parse({
      name: "CrimeNet XGBoost Point Process",
      version: "xgb_pp_full_train_v1_depth12",
      description:
        "Full-feature point-process baseline using calendar, contextual, lighting, and leakage-safe history features.",
      status: "fixture",
      validationYear: 2024,
      h3Resolution: 9,
      featureCount: 63,
      intensityUnit: "events_per_cell_hour",
      supportedCities: CITIES.map((city) => city.id),
    });
  }
  const health = await getServiceHealth(signal);
  return modelMetadataSchema.parse({
    name: "CrimeNet National Intensity + Mark Model",
    version: health.mark_model.run_id ?? health.snapshot_id,
    description:
      "Current-hour national H3 intensity snapshot with an on-demand 87-class mark model.",
    status:
      health.status === "ok" && health.mark_model.status === "ready" ? "available" : "unavailable",
    h3Resolution: 9,
    intensityUnit: "events_per_cell_hour",
  });
}
