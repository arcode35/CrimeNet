const baseUrl = (process.env.CRIMENET_API_URL || "http://localhost:8000").replace(/\/$/, "");

async function get(path) {
  const response = await fetch(`${baseUrl}${path}`);
  if (!response.ok) throw new Error(`${path} returned HTTP ${response.status}`);
  return response.json();
}

const health = await get("/health");
const national = await get("/api/v1/intensity/viewport?west=-125&south=24&east=-66&north=50");
const neighborhood = await get(
  "/api/v1/intensity/viewport?west=-95.40&south=29.73&east=-95.32&north=29.79",
);
if (!national.cells.length || national.resolution >= 9) {
  throw new Error("National viewport did not return a coarse H3 LOD surface");
}
if (!neighborhood.cells.length || neighborhood.resolution !== 9) {
  throw new Error("Neighborhood viewport did not return native H3-r9 cells");
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
const selected = await get(
  `/api/v1/predict/cell/${encodeURIComponent(neighborhood.cells[0].h3)}?top_k=87`,
);
const classIds = new Set(selected.mark.distribution.map((item) => item.class_id));
if (selected.mark.distribution.length !== 87 || classIds.size !== 87) {
  throw new Error("Selected-cell smoke response did not contain 87 unique classes");
}
if (
  selected.snapshot_id !== health.snapshot_id ||
  national.snapshot_id !== health.snapshot_id ||
  neighborhood.snapshot_id !== health.snapshot_id
) {
  throw new Error("Health, LOD viewports, and selected-cell snapshots were not consistent");
}

console.log(
  JSON.stringify(
    {
      status: health.status,
      snapshot_id: health.snapshot_id,
      national_resolution: national.resolution,
      national_cells: national.cells.length,
      neighborhood_resolution: neighborhood.resolution,
      neighborhood_cells: neighborhood.cells.length,
      selected_h3: selected.h3,
      mark_classes: selected.mark.distribution.length,
    },
    null,
    2,
  ),
);
