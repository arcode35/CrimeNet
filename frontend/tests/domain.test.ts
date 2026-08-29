import { describe, expect, it } from "vitest";
import { predictionCellSchema, predictionResponseSchema } from "@/lib/domain";

const features = [
  { group: "temporal", available: true },
  { group: "crime_history", available: true },
];

describe("inference coverage invariants", () => {
  it("permits normal inference UI for full coverage", () => {
    const result = predictionCellSchema.parse({
      h3: "892664c1a8fffff",
      intensity: 0,
      percentile: 0,
      coverage: "full",
      missingReason: null,
      features,
    });
    expect(result.intensity).toBe(0);
  });

  it("never permits unsupported geography to carry a normal prediction", () => {
    expect(() =>
      predictionCellSchema.parse({
        h3: "892664c1a8fffff",
        intensity: 0,
        percentile: null,
        coverage: "unsupported",
        missingReason: "No history",
        features,
      }),
    ).toThrow(/cannot contain predictions/);
  });

  it("keeps missing prediction distinct from zero intensity", () => {
    const unsupported = predictionCellSchema.parse({
      h3: "892664c1a8fffff",
      intensity: null,
      percentile: null,
      coverage: "unsupported",
      missingReason: "No history",
      features,
    });
    const zero = predictionCellSchema.parse({
      h3: "892664c1a8fffff",
      intensity: 0,
      percentile: 0,
      coverage: "full",
      missingReason: null,
      features,
    });
    expect(unsupported.intensity).toBeNull();
    expect(zero.intensity).toBe(0);
  });

  it("rejects malformed API responses", () => {
    expect(() =>
      predictionResponseSchema.parse({ city: "chicago", cells: [{ intensity: "high" }] }),
    ).toThrow();
  });
});
