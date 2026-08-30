import { z } from "zod";

export const coverageStatusSchema = z.enum(["full", "partial", "unsupported"]);

export const featureAvailabilitySchema = z.object({
  group: z.enum([
    "temporal",
    "lighting",
    "weather",
    "osm",
    "socioeconomic",
    "crime_history",
    "neighbor_history",
  ]),
  available: z.boolean(),
  observedAt: z.string().datetime().nullable().optional(),
});

export const predictionCellSchema = z
  .object({
    h3: z.string().min(15),
    intensity: z.number().nonnegative().nullable(),
    percentile: z.number().min(0).max(1).nullable(),
    visualIntensity: z.number().nonnegative().nullable().optional(),
    modeledR9Cells: z.number().int().nonnegative().optional(),
    coverage: coverageStatusSchema,
    missingReason: z.string().nullable(),
    features: z.array(featureAvailabilitySchema),
  })
  .superRefine((cell, ctx) => {
    if (cell.coverage === "unsupported" && cell.intensity !== null) {
      ctx.addIssue({ code: "custom", message: "Unsupported cells cannot contain predictions" });
    }
    if (cell.coverage !== "unsupported" && cell.intensity === null) {
      ctx.addIssue({ code: "custom", message: "Supported cells must contain an intensity" });
    }
  });

export const predictionResponseSchema = z.object({
  city: z.string(),
  timestamp: z.string().datetime(),
  horizonHours: z.number().positive(),
  unit: z.literal("events_per_cell_hour"),
  modelVersion: z.string(),
  snapshotId: z.string().optional(),
  source: z.enum(["fixture", "live"]).default("fixture"),
  resolution: z.number().int().min(4).max(9).default(9),
  aggregation: z.enum(["native_r9", "sum_r9_child_intensity"]).optional(),
  visualizationMetric: z.literal("mean_r9_events_per_hour").optional(),
  candidateCount: z.number().int().nonnegative().optional(),
  cells: z.array(predictionCellSchema),
});

export const modelMetadataSchema = z.object({
  name: z.string(),
  version: z.string(),
  description: z.string(),
  status: z.enum(["available", "unavailable", "fixture"]),
  validationYear: z.number().int().optional(),
  h3Resolution: z.number().int(),
  featureCount: z.number().int().positive().optional(),
  intensityUnit: z.literal("events_per_cell_hour"),
  supportedCities: z.array(z.string()).optional(),
  metrics: z.record(z.string(), z.number()).optional(),
  featureImportance: z.array(z.object({ feature: z.string(), gain: z.number() })).optional(),
});

export type CoverageStatus = z.infer<typeof coverageStatusSchema>;
export type FeatureAvailability = z.infer<typeof featureAvailabilitySchema>;
export type PredictionCell = z.infer<typeof predictionCellSchema>;
export type PredictionResponse = z.infer<typeof predictionResponseSchema>;
export type ModelMetadata = z.infer<typeof modelMetadataSchema>;

export type City = {
  id: string;
  name: string;
  center: [longitude: number, latitude: number];
  zoom: number;
  timezone: string;
};

export const CITIES: readonly City[] = [
  {
    id: "chicago",
    name: "Chicago",
    center: [-87.6298, 41.8781],
    zoom: 11.45,
    timezone: "America/Chicago",
  },
  {
    id: "baltimore",
    name: "Baltimore",
    center: [-76.6122, 39.2904],
    zoom: 11.55,
    timezone: "America/New_York",
  },
  {
    id: "dallas",
    name: "Dallas",
    center: [-96.797, 32.7767],
    zoom: 10.95,
    timezone: "America/Chicago",
  },
  {
    id: "fort_worth",
    name: "Fort Worth",
    center: [-97.3308, 32.7555],
    zoom: 11,
    timezone: "America/Chicago",
  },
  {
    id: "new_york",
    name: "New York",
    center: [-73.9857, 40.7484],
    zoom: 10.65,
    timezone: "America/New_York",
  },
  {
    id: "san_francisco",
    name: "San Francisco",
    center: [-122.4194, 37.7749],
    zoom: 11.45,
    timezone: "America/Los_Angeles",
  },
  {
    id: "seattle",
    name: "Seattle",
    center: [-122.3321, 47.6062],
    zoom: 11.35,
    timezone: "America/Los_Angeles",
  },
  {
    id: "washington_dc",
    name: "Washington, DC",
    center: [-77.0369, 38.9072],
    zoom: 11.25,
    timezone: "America/New_York",
  },
] as const;

export function getCity(cityId: string): City {
  return CITIES.find((city) => city.id === cityId) ?? CITIES[0];
}

export function assertPredictionSemantics(cell: PredictionCell): "prediction" | "no-prediction" {
  return cell.coverage === "unsupported" ? "no-prediction" : "prediction";
}
