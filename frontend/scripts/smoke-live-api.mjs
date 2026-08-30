const baseUrl = (process.env.CRIMESENSE_API_URL || "https://api.crimesense.ai").replace(
  /\/$/,
  "",
);

async function get(path, params) {
  const url = new URL(path, `${baseUrl}/`);
  if (params) url.search = new URLSearchParams(params).toString();
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url.pathname} returned HTTP ${response.status}`);
  return response.json();
}

const health = await get("/health");
const timeline = await get("/api/v1/intensity/timeline");
const live = timeline.snapshots.find((snapshot) => snapshot.kind === "live");
const forecast =
  timeline.snapshots.find((snapshot) => snapshot.horizon_hours === 6) ??
  timeline.snapshots.find((snapshot) => snapshot.kind === "forecast");
if (!live || !forecast) throw new Error("Timeline did not expose live and forecast snapshots");

const national = await get("/api/v1/intensity/viewport", {
  west: "-125",
  south: "24",
  east: "-66",
  north: "50",
  valid_utc_hour: live.valid_utc_hour,
});
const neighborhood = await get("/api/v1/intensity/viewport", {
  west: "-95.40",
  south: "29.73",
  east: "-95.32",
  north: "29.79",
  valid_utc_hour: forecast.valid_utc_hour,
});
if (!national.cells.length || national.resolution >= 9) {
  throw new Error("National viewport did not return a coarse H3 LOD surface");
}
if (!neighborhood.cells.length || neighborhood.resolution !== 9) {
  throw new Error("Forecast neighborhood viewport did not return native H3-r9 cells");
}
for (const cell of [national.cells[0], neighborhood.cells[0]]) {
  if (
    typeof cell.events_per_hour !== "number" ||
    typeof cell.mean_r9_events_per_hour !== "number" ||
    typeof cell.modeled_r9_cells !== "number"
  ) {
    throw new Error("LOD viewport cell omitted total, mean-r9, or modeled-cell metadata");
  }
}
const selected = await get(`/api/v1/predict/cell/${encodeURIComponent(neighborhood.cells[0].h3)}`, {
  top_k: "87",
  valid_utc_hour: forecast.valid_utc_hour,
});
const classIds = new Set(selected.mark.distribution.map((item) => item.class_id));
if (selected.mark.distribution.length !== 87 || classIds.size !== 87) {
  throw new Error("Selected-cell smoke response did not contain 87 unique classes");
}
if (
  live.snapshot_id !== health.snapshot_id ||
  national.snapshot_id !== live.snapshot_id ||
  neighborhood.snapshot_id !== forecast.snapshot_id ||
  selected.snapshot_id !== forecast.snapshot_id
) {
  throw new Error("Timeline, viewports, and selected-cell snapshots were not time-consistent");
}

console.log(
  JSON.stringify(
    {
      status: health.status,
      api_origin: new URL(baseUrl).origin,
      as_of_utc_hour: timeline.as_of_utc_hour,
      timeline_entries: timeline.snapshots.length,
      live_snapshot: live.snapshot_id,
      forecast_horizon_hours: forecast.horizon_hours,
      forecast_snapshot: forecast.snapshot_id,
      national_resolution: national.resolution,
      forecast_neighborhood_resolution: neighborhood.resolution,
      selected_h3: selected.h3,
      forecast_mark_classes: selected.mark.distribution.length,
    },
    null,
    2,
  ),
);
