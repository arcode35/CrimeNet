import { describe, expect, it } from "vitest";
import { CRIME_FAMILIES, CRIME_FAMILY_BY_CODE, CRIME_SUBTYPES } from "@/lib/taxonomy";

describe("canonical CrimeNet taxonomy", () => {
  it("contains exactly 20 families and 87 modeled subtypes", () => {
    expect(CRIME_FAMILIES).toHaveLength(20);
    expect(CRIME_SUBTYPES).toHaveLength(87);
    expect(CRIME_FAMILIES.reduce((sum, family) => sum + family.subtypes.length, 0)).toBe(87);
  });

  it("has unique codes and keys with valid family membership", () => {
    expect(new Set(CRIME_SUBTYPES.map((item) => item.subtypeCode)).size).toBe(87);
    expect(new Set(CRIME_SUBTYPES.map((item) => item.subtypeKey)).size).toBe(87);
    for (const subtype of CRIME_SUBTYPES) {
      expect(CRIME_FAMILY_BY_CODE.has(subtype.familyCode)).toBe(true);
    }
  });
});
