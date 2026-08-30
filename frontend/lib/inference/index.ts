import { ApiInferenceProvider } from "./api-provider";
import type { CellPredictionRequest } from "./contracts";
import { FixtureInferenceProvider } from "./fixture-provider";
import type { CrimeNetInferenceProvider } from "./provider";
import { crimeSenseApiUrl, crimeNetDataMode } from "@/lib/config";

export const inferenceProvider: CrimeNetInferenceProvider =
  crimeNetDataMode === "api"
    ? new ApiInferenceProvider(crimeSenseApiUrl)
    : new FixtureInferenceProvider(70);

export const liveCellPredictionQueryKey = (
  asOfUtcHour: string | undefined,
  validUtcHour: string | undefined,
  snapshotId: string | undefined,
  h3: string,
) => ["cell-prediction", "api", asOfUtcHour, validUtcHour, snapshotId, h3] as const;

export const cellPredictionQueryKey = (request: CellPredictionRequest) =>
  inferenceProvider.kind === "api"
    ? liveCellPredictionQueryKey(
        request.asOfUtcHour,
        request.validUtcHour,
        request.snapshotId,
        request.h3,
      )
    : ([
        "cell-prediction",
        "fixture",
        request.cityId,
        request.h3,
        request.timestamp,
        request.horizonHours,
      ] as const);

export * from "./contracts";
