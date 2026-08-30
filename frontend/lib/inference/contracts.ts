import { z } from "zod";
import type { FeatureAvailability, PredictionCell } from "@/lib/domain";
import { CRIME_FAMILIES, CRIME_SUBTYPES } from "@/lib/taxonomy";

export type ProviderKind = "fixture" | "api";

export type CellPredictionRequest = {
  cityId: string;
  h3: string;
  timestamp: string;
  horizonHours: number;
  snapshotId?: string;
  validUtcHour?: string;
  asOfUtcHour?: string;
  forecastHorizonHours?: number;
  surfaceCell?: PredictionCell;
  signal?: AbortSignal;
};

const subtypePredictionSchema = z.object({
  subtypeCode: z.string(),
  subtypeKey: z.string(),
  subtypeLabel: z.string(),
  familyCode: z.string(),
  familyKey: z.string(),
  familyLabel: z.string(),
  conditionalProbability: z.number().min(0).max(1),
  intensity: z.number().nonnegative(),
});

const familyPredictionSchema = z.object({
  familyCode: z.string(),
  familyKey: z.string(),
  familyLabel: z.string(),
  conditionalProbability: z.number().min(0).max(1),
  intensity: z.number().nonnegative(),
  subtypeCodes: z.array(z.string()),
});

const temporalPredictionSchema = z.object({
  timestamp: z.string().datetime(),
  totalIntensity: z.number().nonnegative(),
  integratedIntensity: z.number().nonnegative().optional(),
  topFamilyCode: z.string().optional(),
});

export const cellPredictionSchema = z
  .object({
    h3: z.string().min(15),
    cityId: z.string(),
    snapshotId: z.string().optional(),
    timestamp: z.string().datetime(),
    horizonSeconds: z.number().int().positive(),
    intensityUnit: z.literal("events_per_cell_hour"),
    totalIntensity: z.number().nonnegative().nullable(),
    integratedIntensity: z.number().nonnegative().nullable(),
    eventProbability: z.number().min(0).max(1).nullable(),
    familyDistribution: z.array(familyPredictionSchema),
    subtypeDistribution: z.array(subtypePredictionSchema),
    temporal: z.array(temporalPredictionSchema).optional(),
    coverage: z.enum(["full", "partial", "unsupported"]),
    missingReason: z.string().nullable(),
    features: z.array(z.custom<FeatureAvailability>()),
    model: z.object({ name: z.string(), version: z.string() }),
    provider: z.object({ kind: z.enum(["fixture", "api"]), label: z.string() }),
  })
  .superRefine((prediction, ctx) => {
    if (prediction.coverage === "unsupported") {
      if (prediction.totalIntensity !== null || prediction.subtypeDistribution.length > 0) {
        ctx.addIssue({ code: "custom", message: "Unsupported cells cannot contain inference" });
      }
      return;
    }
    if (prediction.totalIntensity === null) {
      ctx.addIssue({ code: "custom", message: "Supported cells require total intensity" });
      return;
    }
    if (prediction.subtypeDistribution.length !== CRIME_SUBTYPES.length) {
      ctx.addIssue({ code: "custom", message: "Expected all 87 modeled subtypes" });
    }
    if (prediction.familyDistribution.length !== CRIME_FAMILIES.length) {
      ctx.addIssue({ code: "custom", message: "Expected all 20 modeled families" });
    }
    const probabilitySum = prediction.subtypeDistribution.reduce(
      (sum, item) => sum + item.conditionalProbability,
      0,
    );
    if (Math.abs(probabilitySum - 1) > 1e-6) {
      ctx.addIssue({ code: "custom", message: "Subtype probabilities must sum to one" });
    }
    const intensitySum = prediction.subtypeDistribution.reduce(
      (sum, item) => sum + item.intensity,
      0,
    );
    if (Math.abs(intensitySum - prediction.totalIntensity) > 1e-6) {
      ctx.addIssue({ code: "custom", message: "Subtype intensities must sum to total intensity" });
    }
    for (const family of prediction.familyDistribution) {
      const children = prediction.subtypeDistribution.filter(
        (item) => item.familyCode === family.familyCode,
      );
      const childProbability = children.reduce((sum, item) => sum + item.conditionalProbability, 0);
      if (Math.abs(childProbability - family.conditionalProbability) > 1e-7) {
        ctx.addIssue({
          code: "custom",
          message: `${family.familyCode} does not equal its child sum`,
        });
      }
    }
  });

export type CellPrediction = z.infer<typeof cellPredictionSchema>;
export type FamilyPrediction = CellPrediction["familyDistribution"][number];
export type SubtypePrediction = CellPrediction["subtypeDistribution"][number];
export type TemporalPredictionPoint = NonNullable<CellPrediction["temporal"]>[number];
