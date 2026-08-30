import { describe, expect, it } from "vitest";
import type { IntensityTimelineSnapshot } from "@/lib/api";
import { adjacentForecastIndexes, resolveForecastSelection } from "@/lib/forecast";

const firstTimeline: IntensityTimelineSnapshot[] = [
  {
    snapshot_id: "20260830T0400",
    valid_utc_hour: "2026-08-30T04:00:00+00:00",
    horizon_hours: 0,
    kind: "live",
  },
  {
    snapshot_id: "20260830T0500",
    valid_utc_hour: "2026-08-30T05:00:00+00:00",
    horizon_hours: 1,
    kind: "forecast",
  },
  {
    snapshot_id: "20260830T0600",
    valid_utc_hour: "2026-08-30T06:00:00+00:00",
    horizon_hours: 2,
    kind: "forecast",
  },
];

const rolledTimeline: IntensityTimelineSnapshot[] = [
  { ...firstTimeline[1], kind: "live", horizon_hours: 0 },
  { ...firstTimeline[2], horizon_hours: 1 },
  {
    snapshot_id: "20260830T0700",
    valid_utc_hour: "2026-08-30T07:00:00+00:00",
    horizon_hours: 2,
    kind: "forecast",
  },
];

describe("rolling forecast selection", () => {
  it("keeps LIVE intent attached to the newly rolled live entry", () => {
    expect(resolveForecastSelection(firstTimeline, null)?.snapshot.snapshot_id).toBe(
      "20260830T0400",
    );
    expect(resolveForecastSelection(rolledTimeline, null)?.snapshot.snapshot_id).toBe(
      "20260830T0500",
    );
  });

  it("preserves an exact future UTC timestamp when its slider index shifts", () => {
    const selected = resolveForecastSelection(
      rolledTimeline,
      "2026-08-30T06:00:00.000Z",
    );
    expect(selected).toMatchObject({ index: 1, snapshot: { snapshot_id: "20260830T0600" } });
  });

  it("moves to the closest available entry when a timestamp falls out", () => {
    const selected = resolveForecastSelection(rolledTimeline, "2026-08-30T04:00:00+00:00");
    expect(selected).toMatchObject({ index: 0, snapshot: { kind: "live" } });
  });

  it("prefetches only two adjacent hours in each direction", () => {
    expect(adjacentForecastIndexes(3, 8)).toEqual([1, 2, 4, 5]);
    expect(adjacentForecastIndexes(0, 3)).toEqual([1, 2]);
    expect(adjacentForecastIndexes(2, 3)).toEqual([0, 1]);
  });
});
