import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  CRIME_FAMILIES,
  CRIME_FAMILY_BY_CODE,
  CRIME_SUBTYPES,
  MARK_CLASS_BY_ID,
} from "@/lib/taxonomy";

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

  it("matches the authoritative class-label artifact for every class ID", () => {
    const artifact = JSON.parse(
      readFileSync(
        resolve(process.cwd(), "../src/machine_learning/artifacts/class_labels.json"),
        "utf8",
      ),
    ) as {
      schema: string;
      classes: Array<{
        class_id: number;
        code: string;
        label: string;
        family_code: string;
        family: string;
      }>;
    };
    expect(artifact.schema).toBe("crimenet_mark_class_labels_v1");
    expect(artifact.classes).toHaveLength(87);
    for (const item of artifact.classes) {
      expect(MARK_CLASS_BY_ID.get(item.class_id)).toMatchObject({
        classId: item.class_id,
        subtypeCode: item.code,
        subtypeKey: item.label,
        familyCode: item.family_code,
        familyKey: item.family,
      });
    }
  });
});
