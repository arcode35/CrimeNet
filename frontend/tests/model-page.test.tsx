import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import ModelPage from "@/app/model/page";

describe("CrimeSense model-system page", () => {
  it("explains the deployed CrimeNet architecture and verified scale", () => {
    render(<ModelPage />);

    expect(
      screen.getByRole("heading", {
        name: /Data infrastructure first,.*forecasting system second\./,
      }),
    ).toBeInTheDocument();
    expect(screen.getAllByText("POWERED BY CRIMENET").length).toBeGreaterThan(0);
    expect(screen.getAllByText(/XGBoost/).length).toBeGreaterThanOrEqual(4);
    expect(screen.getByText("STAGE 01")).toBeInTheDocument();
    expect(screen.getByText("STAGE 02")).toBeInTheDocument();
    expect(screen.getAllByText(/87 classes/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/24-hour/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/H3 r9/).length).toBeGreaterThan(0);
    expect(screen.getAllByText("16.7M", { exact: true }).length).toBeGreaterThan(0);
    expect(screen.getAllByText("180M+", { exact: true }).length).toBeGreaterThan(0);
    expect(screen.getByText(/25 static place features and 13 values/)).toBeInTheDocument();
    expect(screen.getByText("Poisson / point-process")).toBeInTheDocument();
    expect(screen.getByText("multi:softprob")).toBeInTheDocument();
    expect(screen.getByText(/The model predicts at H3 resolution 9/)).toBeInTheDocument();
  });

  it("does not expose stale city-era or repository TODO content", () => {
    render(<ModelPage />);

    expect(screen.queryByText(/8 jurisdictions/i)).not.toBeInTheDocument();
    expect(screen.queryByText("SUPPORTED GEOGRAPHIES")).not.toBeInTheDocument();
    expect(screen.queryByText("Evaluation artifact API required")).not.toBeInTheDocument();
    expect(screen.queryByText("GET /v1/model/metrics")).not.toBeInTheDocument();
  });
});
