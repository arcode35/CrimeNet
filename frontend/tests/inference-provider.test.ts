import { describe, expect, it } from "vitest";
import { latLngToCell } from "h3-js";
import { adaptApiPrediction } from "@/lib/inference/api-provider";
import { FixtureInferenceProvider } from "@/lib/inference/fixture-provider";
import { CRIME_SUBTYPES } from "@/lib/taxonomy";

const request = {
  cityId: "chicago",
  h3: latLngToCell(41.8781, -87.6298, 9),
  timestamp: "2024-08-21T22:00:00.000Z",
  horizonHours: 6,
  surfaceCell: {
    h3: latLngToCell(41.8781, -87.6298, 9),
    intensity: 0.125,
    percentile: 0.8,
    coverage: "full" as const,
    missingReason: null,
    features: [],
  },
};

describe("fixture cell inference", () => {
  const provider = new FixtureInferenceProvider();

  it("is deterministic and mathematically coherent", async () => {
    const first = await provider.getCellPrediction(request);
    const second = await provider.getCellPrediction(request);
    expect(first).toEqual(second);
    expect(first.subtypeDistribution).toHaveLength(87);
    expect(first.familyDistribution).toHaveLength(20);
    expect(
      first.subtypeDistribution.reduce((sum, item) => sum + item.conditionalProbability, 0),
    ).toBeCloseTo(1, 10);
    expect(
      first.familyDistribution.reduce((sum, item) => sum + item.conditionalProbability, 0),
    ).toBeCloseTo(1, 10);
    expect(first.subtypeDistribution.reduce((sum, item) => sum + item.intensity, 0)).toBeCloseTo(
      first.totalIntensity!,
      10,
    );
    for (const subtype of first.subtypeDistribution) {
      expect(subtype.intensity).toBeCloseTo(
        first.totalIntensity! * subtype.conditionalProbability,
        12,
      );
    }
    expect(first.integratedIntensity).toBeCloseTo(first.totalIntensity! * 6, 10);
    expect(first.eventProbability).toBeCloseTo(1 - Math.exp(-first.integratedIntensity!), 10);
    for (const family of first.familyDistribution) {
      const children = first.subtypeDistribution.filter(
        (item) => item.familyCode === family.familyCode,
      );
      expect(children.reduce((sum, item) => sum + item.conditionalProbability, 0)).toBeCloseTo(
        family.conditionalProbability,
        10,
      );
    }
  });

  it("changes the mark distribution with time", async () => {
    const first = await provider.getCellPrediction(request);
    const later = await provider.getCellPrediction({
      ...request,
      timestamp: "2024-08-22T03:00:00.000Z",
    });
    expect(later.subtypeDistribution.map((item) => item.conditionalProbability)).not.toEqual(
      first.subtypeDistribution.map((item) => item.conditionalProbability),
    );
  });

  it("returns no fake values for unsupported coverage", async () => {
    const result = await provider.getCellPrediction({
      ...request,
      surfaceCell: {
        ...request.surfaceCell,
        intensity: null,
        percentile: null,
        coverage: "unsupported",
      },
    });
    expect(result.totalIntensity).toBeNull();
    expect(result.subtypeDistribution).toEqual([]);
  });

  it("preserves partial coverage while returning a valid distribution", async () => {
    const result = await provider.getCellPrediction({
      ...request,
      surfaceCell: { ...request.surfaceCell, coverage: "partial" },
    });
    expect(result.coverage).toBe("partial");
    expect(result.subtypeDistribution).toHaveLength(87);
  });
});

describe("API inference adapter", () => {
  it("maps an explicit backend class order into canonical order", () => {
    const reversed = [...CRIME_SUBTYPES].reverse();
    const weights = reversed.map((_, index) => index + 1);
    const total = weights.reduce((sum, value) => sum + value, 0);
    const result = adaptApiPrediction({
      h3: request.h3,
      cityId: request.cityId,
      timestamp: request.timestamp,
      horizonSeconds: 3600,
      totalIntensity: 0.1,
      coverage: "full",
      classCodes: reversed.map((item) => item.subtypeCode),
      probabilities: weights.map((value) => value / total),
      model: { name: "test", version: "test" },
    });
    expect(result.subtypeDistribution.map((item) => item.subtypeCode)).toEqual(
      CRIME_SUBTYPES.map((item) => item.subtypeCode),
    );
  });

  it("rejects missing authoritative class mappings", () => {
    expect(() =>
      adaptApiPrediction({
        h3: request.h3,
        cityId: request.cityId,
        timestamp: request.timestamp,
        horizonSeconds: 3600,
        totalIntensity: 0.1,
        coverage: "full",
        classCodes: CRIME_SUBTYPES.slice(0, 86).map((item) => item.subtypeCode),
        probabilities: Array(86).fill(1 / 86),
        model: { name: "test", version: "test" },
      }),
    ).toThrow(/87 class codes/);
  });
});
