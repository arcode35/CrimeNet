import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Timeline } from "@/components/explorer/timeline";
import type { IntensityTimelineSnapshot } from "@/lib/api";
import { useExplorerStore } from "@/stores/explorer-store";

const snapshots: IntensityTimelineSnapshot[] = [
  {
    snapshot_id: "live",
    valid_utc_hour: "2026-08-30T04:00:00+00:00",
    horizon_hours: 0,
    kind: "live",
  },
  {
    snapshot_id: "plus-1",
    valid_utc_hour: "2026-08-30T05:00:00+00:00",
    horizon_hours: 1,
    kind: "forecast",
  },
  {
    snapshot_id: "plus-2",
    valid_utc_hour: "2026-08-30T06:00:00+00:00",
    horizon_hours: 2,
    kind: "forecast",
  },
];

describe("production forecast timeline", () => {
  beforeEach(() => useExplorerStore.setState({ playing: false, horizonHours: 1 }));

  it("renders returned snapshots as a discrete index slider and navigates by index", () => {
    const select = vi.fn();
    render(
      <Timeline
        isFetching={false}
        liveMode
        snapshots={snapshots}
        selectedIndex={1}
        onSelectedIndexChange={select}
      />,
    );
    expect(screen.getByText("+1H FORECAST")).toBeInTheDocument();
    expect(screen.getByRole("slider", { name: "Forecast hour" })).toHaveAttribute("aria-valuemax", "2");
    expect(screen.getByRole("slider", { name: "Forecast hour" })).toHaveAttribute("aria-valuenow", "1");
    fireEvent.click(screen.getByRole("button", { name: "Next hour" }));
    expect(select).toHaveBeenCalledWith(2);
    fireEvent.click(screen.getByRole("button", { name: "Return to live" }));
    expect(select).toHaveBeenCalledWith(0);
  });

  it("disables forecast controls while retaining live data when timeline is unavailable", () => {
    render(
      <Timeline
        isFetching={false}
        liveMode
        snapshots={[snapshots[0]]}
        selectedIndex={0}
        forecastUnavailable
      />,
    );
    expect(screen.getByText("FORECAST UNAVAILABLE · LIVE DATA REMAINS ACTIVE")).toBeVisible();
    expect(screen.getByRole("button", { name: "Next hour" })).toBeDisabled();
    expect(screen.queryByRole("slider", { name: "Forecast hour" })).not.toBeInTheDocument();
  });
});
