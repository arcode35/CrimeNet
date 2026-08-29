import { fireEvent, render, screen, within } from "@testing-library/react";
import { latLngToCell } from "h3-js";
import { beforeEach, describe, expect, it } from "vitest";
import { Inspector } from "@/components/explorer/inspector";
import { Providers } from "@/components/providers";
import { predictionResponseSchema } from "@/lib/domain";
import { useExplorerStore } from "@/stores/explorer-store";

const h3 = latLngToCell(41.8781, -87.6298, 9);
const base = {
  city: "chicago",
  timestamp: "2024-08-21T22:00:00.000Z",
  horizonHours: 1,
  unit: "events_per_cell_hour",
  modelVersion: "test",
} as const;

describe("cell inspector", () => {
  beforeEach(() => useExplorerStore.setState({ cityId: "chicago", selectedH3: h3 }));

  const renderInspector = (data: ReturnType<typeof predictionResponseSchema.parse>) =>
    render(
      <Providers>
        <Inspector data={data} />
      </Providers>,
    );

  it("renders valid hierarchical prediction values", async () => {
    const data = predictionResponseSchema.parse({
      ...base,
      cells: [
        {
          h3,
          intensity: 0.125,
          percentile: 0.8,
          coverage: "full",
          missingReason: null,
          features: [],
        },
      ],
    });
    renderInspector(data);
    expect(await screen.findByText("EVENT INTENSITY")).toBeInTheDocument();
    expect(screen.getByText("CRIME MIX")).toBeInTheDocument();
    expect(screen.getByText("FIXTURE DATA")).toBeInTheDocument();
    expect(screen.getByText("Full model")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Larceny \/ Theft/ })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "INTENSITY" }));
    expect(screen.getByText("λfamily · events / cell / hour")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "VIEW ALL 87 →" }));
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText("All 87 modeled crime types")).toBeInTheDocument();
    fireEvent.change(within(dialog).getByPlaceholderText("Search modeled crime type..."), {
      target: { value: "theft from vehicle" },
    });
    expect(within(dialog).getByText("Theft From Vehicle")).toBeInTheDocument();
  });

  it("renders an explicit no-prediction state for unsupported cells", async () => {
    const data = predictionResponseSchema.parse({
      ...base,
      cells: [
        {
          h3,
          intensity: null,
          percentile: null,
          coverage: "unsupported",
          missingReason: "History unavailable",
          features: [],
        },
      ],
    });
    renderInspector(data);
    expect(
      await screen.findByText(/Missing data is not interpreted as zero intensity/),
    ).toBeInTheDocument();
    expect(screen.queryByText("0.000")).not.toBeInTheDocument();
    expect(screen.queryByText("CRIME MIX")).not.toBeInTheDocument();
  });
});
