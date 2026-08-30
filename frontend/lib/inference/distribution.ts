import type { FamilyPrediction, SubtypePrediction } from "./contracts";
import { CRIME_FAMILIES, CRIME_SUBTYPES } from "@/lib/taxonomy";

export function buildDistributions(probabilities: number[], totalIntensity: number) {
  const normalizedTotal = probabilities.reduce((sum, value) => sum + value, 0);
  const subtypeDistribution: SubtypePrediction[] = CRIME_SUBTYPES.map((subtype, index) => {
    const conditionalProbability = probabilities[index] / normalizedTotal;
    return {
      ...subtype,
      conditionalProbability,
      intensity: totalIntensity * conditionalProbability,
    };
  });
  const familyDistribution: FamilyPrediction[] = CRIME_FAMILIES.map((family) => {
    const children = subtypeDistribution.filter((item) => item.familyCode === family.code);
    const conditionalProbability = children.reduce(
      (sum, item) => sum + item.conditionalProbability,
      0,
    );
    return {
      familyCode: family.code,
      familyKey: family.key,
      familyLabel: family.label,
      conditionalProbability,
      intensity: totalIntensity * conditionalProbability,
      subtypeCodes: children.map((item) => item.subtypeCode),
    };
  });
  return { subtypeDistribution, familyDistribution };
}

export function aggregateFamilyDistributions(subtypeDistribution: SubtypePrediction[]) {
  const familyDistribution: FamilyPrediction[] = CRIME_FAMILIES.map((family) => {
    const children = subtypeDistribution.filter((item) => item.familyCode === family.code);
    return {
      familyCode: family.code,
      familyKey: family.key,
      familyLabel: family.label,
      conditionalProbability: children.reduce((sum, item) => sum + item.conditionalProbability, 0),
      intensity: children.reduce((sum, item) => sum + item.intensity, 0),
      subtypeCodes: children.map((item) => item.subtypeCode),
    };
  });
  return familyDistribution;
}
