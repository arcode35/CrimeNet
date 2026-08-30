import type { CellPrediction, CellPredictionRequest, ProviderKind } from "./contracts";

export interface CrimeNetInferenceProvider {
  readonly kind: ProviderKind;
  getCellPrediction(request: CellPredictionRequest): Promise<CellPrediction>;
}
