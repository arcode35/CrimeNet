import type { IntensityTimelineSnapshot } from "@/lib/api";

export type ResolvedForecastSelection = {
  snapshot: IntensityTimelineSnapshot;
  index: number;
};

export function resolveForecastSelection(
  snapshots: readonly IntensityTimelineSnapshot[],
  preferredValidUtcHour: string | null,
): ResolvedForecastSelection | null {
  if (snapshots.length === 0) return null;
  if (preferredValidUtcHour === null) {
    const liveIndex = snapshots.findIndex((snapshot) => snapshot.kind === "live");
    const index = liveIndex >= 0 ? liveIndex : 0;
    return { snapshot: snapshots[index], index };
  }
  const preferredTime = Date.parse(preferredValidUtcHour);
  const exactIndex = snapshots.findIndex(
    (snapshot) => Date.parse(snapshot.valid_utc_hour) === preferredTime,
  );
  if (exactIndex >= 0) return { snapshot: snapshots[exactIndex], index: exactIndex };

  if (Number.isNaN(preferredTime)) return { snapshot: snapshots[0], index: 0 };
  let closestIndex = 0;
  let closestDistance = Number.POSITIVE_INFINITY;
  snapshots.forEach((snapshot, index) => {
    const distance = Math.abs(Date.parse(snapshot.valid_utc_hour) - preferredTime);
    if (distance < closestDistance) {
      closestIndex = index;
      closestDistance = distance;
    }
  });
  return { snapshot: snapshots[closestIndex], index: closestIndex };
}

export function adjacentForecastIndexes(selectedIndex: number, snapshotCount: number) {
  return [-2, -1, 1, 2]
    .map((offset) => selectedIndex + offset)
    .filter((index) => index >= 0 && index < snapshotCount);
}

export function forecastHorizonLabel(snapshot: IntensityTimelineSnapshot) {
  return snapshot.kind === "live" ? "LIVE" : `+${snapshot.horizon_hours}h`;
}
