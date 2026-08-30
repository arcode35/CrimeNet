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
  const liveResponse = (classes = CRIME_SUBTYPES.length) => {
    const reversed = [...CRIME_SUBTYPES].reverse().slice(0, classes);
    const weights = reversed.map((_, index) => index + 1);
    const total = weights.reduce((sum, value) => sum + value, 0);
    return {
      h3: request.h3,
      snapshot_id: "20260829T2200",
      valid_utc_hour: "2026-08-29T22:00:00+00:00",
      intensity: {
        log_intensity: -13.9,
        events_per_second: 0.1 / 3600,
        events_per_hour: 0.1,
      },
      mark: {
        model_run_id: "test-run",
        num_classes: 87,
        labels_available: true,
        distribution: reversed.map((item, index) => ({
          class_id: CRIME_SUBTYPES.indexOf(item),
          subtype: item.subtypeKey,
          probability: weights[index] / total,
          events_per_hour: (0.1 * weights[index]) / total,
        })),
      },
      center: { lat: 41.8781, lon: -87.6298 },
    };
  };

  it("maps class_id identity and preserves backend subtype intensities", () => {
    const result = adaptApiPrediction(liveResponse(), request.cityId);
    expect(result.subtypeDistribution).toHaveLength(87);
    expect(result.snapshotId).toBe("20260829T2200");
    expect(result.timestamp).toBe("2026-08-29T22:00:00.000Z");
    const simpleAssault = result.subtypeDistribution.find((item) => item.subtypeCode === "F04.02");
    const raw = liveResponse().mark.distribution.find((item) => item.class_id === 11)!;
    expect(simpleAssault).toMatchObject({
      subtypeKey: "simple_assault",
      familyCode: "F04",
      conditionalProbability: raw.probability,
      intensity: raw.events_per_hour,
    });
    expect(
      result.subtypeDistribution.reduce((sum, item) => sum + item.conditionalProbability, 0),
    ).toBeCloseTo(1, 10);
    expect(result.subtypeDistribution.reduce((sum, item) => sum + item.intensity, 0)).toBeCloseTo(
      result.totalIntensity!,
      10,
    );
    expect(
      result.familyDistribution.reduce((sum, item) => sum + item.conditionalProbability, 0),
    ).toBeCloseTo(1, 10);
    expect(result.familyDistribution.reduce((sum, item) => sum + item.intensity, 0)).toBeCloseTo(
      result.totalIntensity!,
      10,
    );
  });

  it("rejects missing authoritative class mappings", () => {
    expect(() => adaptApiPrediction(liveResponse(86), request.cityId)).toThrow(/87 mark classes/);
  });
});
