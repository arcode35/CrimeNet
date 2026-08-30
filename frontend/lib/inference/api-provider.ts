import { z } from "zod";
import { fetchCrimeNetJson } from "@/lib/api";
import { crimeNetApiUrl } from "@/lib/config";
import { MARK_CLASS_BY_ID } from "@/lib/taxonomy";
import { cellPredictionSchema, type CellPrediction, type CellPredictionRequest } from "./contracts";
import { aggregateFamilyDistributions } from "./distribution";
import type { CrimeNetInferenceProvider } from "./provider";

export const combinedPredictionResponseSchema = z.object({
  h3: z.string().min(15),
  snapshot_id: z.string().min(1),
  valid_utc_hour: z.string().datetime({ offset: true }),
  intensity: z.object({
    log_intensity: z.number(),
    events_per_second: z.number().nonnegative(),
    events_per_hour: z.number().nonnegative(),
  }),
  mark: z.object({
    model_run_id: z.string().min(1),
    num_classes: z.literal(87),
    labels_available: z.literal(true),
    distribution: z.array(
      z.object({
        class_id: z.number().int().min(0).max(86),
        subtype: z.string().min(1),
        probability: z.number().min(0).max(1),
        events_per_hour: z.number().nonnegative(),
      }),
    ),
  }),
  center: z.object({ lat: z.number(), lon: z.number() }).optional(),
});

export function adaptApiPrediction(raw: unknown, cityId = "unknown"): CellPrediction {
  const parsed = combinedPredictionResponseSchema.parse(raw);
  if (parsed.mark.distribution.length !== 87) {
    throw new Error("Live prediction must expose all 87 mark classes");
  }
  const classIds = new Set(parsed.mark.distribution.map((item) => item.class_id));
  if (classIds.size !== 87) throw new Error("Live prediction contains duplicate mark class IDs");

  const subtypeDistribution = parsed.mark.distribution.map((item) => {
    const metadata = MARK_CLASS_BY_ID.get(item.class_id);
    if (!metadata) throw new Error(`Unknown CrimeNet mark class_id ${item.class_id}`);
    if (metadata.subtypeKey !== item.subtype) {
      throw new Error(`Mark class ${item.class_id} does not match the canonical taxonomy`);
    }
    return {
      subtypeCode: metadata.subtypeCode,
      subtypeKey: metadata.subtypeKey,
      subtypeLabel: metadata.subtypeLabel,
      familyCode: metadata.familyCode,
      familyKey: metadata.familyKey,
      familyLabel: metadata.familyLabel,
      conditionalProbability: item.probability,
      // This multiplication was already performed by the serving API.
      intensity: item.events_per_hour,
    };
  });
  const totalIntensity = parsed.intensity.events_per_hour;
  const familyDistribution = aggregateFamilyDistributions(subtypeDistribution);

  return cellPredictionSchema.parse({
    h3: parsed.h3,
    cityId,
    snapshotId: parsed.snapshot_id,
    timestamp: new Date(parsed.valid_utc_hour).toISOString(),
    horizonSeconds: 3600,
    intensityUnit: "events_per_cell_hour",
    totalIntensity,
    // Derived from the current hourly rate; this is not a separate future forecast.
    integratedIntensity: totalIntensity,
    eventProbability: 1 - Math.exp(-totalIntensity),
    familyDistribution,
    subtypeDistribution,
    coverage: "full",
    missingReason: null,
    features: [],
    model: {
      name: "CrimeNet marked point process",
      version: parsed.mark.model_run_id,
    },
    provider: { kind: "api", label: "LIVE INFERENCE" },
  });
}

export class ApiInferenceProvider implements CrimeNetInferenceProvider {
  readonly kind = "api" as const;

  constructor(private readonly apiUrl = crimeNetApiUrl) {}

  async getCellPrediction(request: CellPredictionRequest) {
    if (!this.apiUrl) throw new Error("NEXT_PUBLIC_CRIMENET_API_URL is required in API mode");
    const path = `/api/v1/predict/cell/${encodeURIComponent(request.h3)}?top_k=87`;
    let raw: unknown;
    if (this.apiUrl === crimeNetApiUrl) {
      raw = await fetchCrimeNetJson(path, request.signal);
    } else {
      const response = await fetch(`${this.apiUrl}${path}`, { signal: request.signal });
      if (!response.ok) throw new Error(`Cell inference service returned ${response.status}`);
      raw = await response.json();
    }
    return adaptApiPrediction(raw, request.cityId);
  }
}
