import { buildDistributions } from "./distribution";
import { cellPredictionSchema, type CellPrediction, type CellPredictionRequest } from "./contracts";
import type { CrimeNetInferenceProvider } from "./provider";
import { CRIME_FAMILIES, CRIME_SUBTYPES } from "@/lib/taxonomy";

const FAMILY_PRIOR: Record<string, number> = {
  F01: 0.012,
  F02: 0.035,
  F03: 0.052,
  F04: 0.172,
  F05: 0.012,
  F06: 0.045,
  F07: 0.076,
  F08: 0.248,
  F09: 0.096,
  F10: 0.008,
  F11: 0.071,
  F12: 0.041,
  F13: 0.016,
  F14: 0.042,
  F15: 0.025,
  F16: 0.009,
  F17: 0.12,
  F18: 0.018,
  F19: 0.052,
  F20: 0.048,
};

const SUBTYPE_PRIOR: Record<string, number> = {
  other_larceny_theft: 1.8,
  pocket_picking: 0.35,
  purse_snatching: 0.2,
  shoplifting: 1.35,
  theft_from_building: 0.65,
  theft_from_vehicle: 3.35,
  theft_vehicle_parts: 0.8,
  aggravated_assault: 1,
  simple_assault: 1.55,
  reckless_endangerment: 0.35,
  fraud: 1.7,
  identity_cyber_fraud: 1.45,
  drug_narcotic: 1.65,
  drug_equipment: 0.35,
};

function hash(value: string) {
  let result = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    result ^= value.charCodeAt(index);
    result = Math.imul(result, 16777619);
  }
  return result >>> 0;
}

const unitFloat = (value: string) => hash(value) / 0xffffffff;

function fixtureWeights(h3: string, timestamp: string) {
  const date = new Date(timestamp);
  const hour = date.getUTCHours() + date.getUTCMinutes() / 60;
  const day = Math.floor(date.getTime() / 86_400_000);
  return CRIME_SUBTYPES.map((subtype, index) => {
    const family = CRIME_FAMILIES.find((item) => item.code === subtype.familyCode)!;
    const subtypePrior = SUBTYPE_PRIOR[subtype.subtypeKey] ?? 1;
    const familySubtypePrior = family.subtypes.reduce(
      (sum, [, key]) => sum + (SUBTYPE_PRIOR[key] ?? 1),
      0,
    );
    const baseline = FAMILY_PRIOR[subtype.familyCode] * (subtypePrior / familySubtypePrior);
    const spatial = (unitFloat(`${h3}|${subtype.subtypeCode}`) - 0.5) * 0.7;
    const phase = unitFloat(`${subtype.subtypeCode}|${day}`) * Math.PI * 2;
    const temporal = Math.sin((hour / 24) * Math.PI * 2 + phase) * 0.3;
    const subtypeShape = 0.76 + unitFloat(`${subtype.subtypeKey}|shape|${h3.slice(-6)}`) * 0.5;
    return Math.max(1e-9, baseline * subtypeShape * Math.exp(spatial + temporal + index * 0.0002));
  });
}

function unsupportedPrediction(request: CellPredictionRequest): CellPrediction {
  return cellPredictionSchema.parse({
    h3: request.h3,
    cityId: request.cityId,
    timestamp: request.timestamp,
    horizonSeconds: request.horizonHours * 3600,
    intensityUnit: "events_per_cell_hour",
    totalIntensity: null,
    integratedIntensity: null,
    eventProbability: null,
    familyDistribution: [],
    subtypeDistribution: [],
    coverage: "unsupported",
    missingReason:
      request.surfaceCell?.missingReason ?? "Required inference features are unavailable.",
    features: request.surfaceCell?.features ?? [],
    model: { name: "CrimeNet marked point process", version: "fixture/marks-v1" },
    provider: { kind: "fixture", label: "FIXTURE DATA" },
  });
}

export class FixtureInferenceProvider implements CrimeNetInferenceProvider {
  readonly kind = "fixture" as const;

  constructor(private readonly latencyMs = 0) {}

  async getCellPrediction(request: CellPredictionRequest): Promise<CellPrediction> {
    if (this.latencyMs > 0) {
      await new Promise<void>((resolve, reject) => {
        const timer = setTimeout(resolve, this.latencyMs);
        request.signal?.addEventListener(
          "abort",
          () => {
            clearTimeout(timer);
            reject(new DOMException("Aborted", "AbortError"));
          },
          { once: true },
        );
      });
    }
    const coverage =
      request.surfaceCell?.coverage ??
      (unitFloat(`${request.h3}|coverage`) < 0.03
        ? "unsupported"
        : unitFloat(`${request.h3}|coverage`) < 0.11
          ? "partial"
          : "full");
    if (coverage === "unsupported") return unsupportedPrediction(request);

    const date = new Date(request.timestamp);
    const base = 0.035 + unitFloat(`${request.h3}|intensity`) * 0.12;
    const hourWave = 0.92 + 0.14 * Math.sin((date.getUTCHours() / 24) * Math.PI * 2 - 0.8);
    const totalIntensity = request.surfaceCell?.intensity ?? base * hourWave;
    const integratedIntensity = totalIntensity * request.horizonHours;
    const eventProbability = 1 - Math.exp(-integratedIntensity);
    const distribution = buildDistributions(
      fixtureWeights(request.h3, request.timestamp),
      totalIntensity,
    );
    const temporal = Array.from({ length: 25 }, (_, index) => {
      const offset = index - 12;
      const pointDate = new Date(date.getTime() + offset * 3_600_000);
      const wave = 0.78 + 0.25 * Math.sin((pointDate.getUTCHours() / 24) * Math.PI * 2 - 0.55);
      const pointIntensity = Math.max(
        0.001,
        totalIntensity * wave * (0.94 + unitFloat(`${request.h3}|${offset}`) * 0.12),
      );
      const pointDistribution = buildDistributions(
        fixtureWeights(request.h3, pointDate.toISOString()),
        pointIntensity,
      );
      const topFamily = [...pointDistribution.familyDistribution].sort(
        (left, right) => right.conditionalProbability - left.conditionalProbability,
      )[0];
      return {
        timestamp: pointDate.toISOString(),
        totalIntensity: pointIntensity,
        integratedIntensity: pointIntensity * request.horizonHours,
        topFamilyCode: topFamily.familyCode,
      };
    });

    return cellPredictionSchema.parse({
      h3: request.h3,
      cityId: request.cityId,
      timestamp: request.timestamp,
      horizonSeconds: request.horizonHours * 3600,
      intensityUnit: "events_per_cell_hour",
      totalIntensity,
      integratedIntensity,
      eventProbability,
      ...distribution,
      temporal,
      coverage,
      missingReason:
        request.surfaceCell?.missingReason ??
        (coverage === "partial" ? "One or more contextual feature groups are unavailable." : null),
      features: request.surfaceCell?.features ?? [],
      model: { name: "CrimeNet marked point process", version: "fixture/marks-v1" },
      provider: { kind: "fixture", label: "FIXTURE DATA" },
    });
  }
}
