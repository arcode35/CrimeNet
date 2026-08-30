# CrimeSense frontend

CrimeSense is the public, map-first geospatial model-operations application powered by CrimeNet infrastructure. MapLibre owns the basemap and camera; deck.gl renders the backend-selected H3-r4 through H3-r9 analytical surface directly on the GPU. React owns controls and small analytical views, never individual spatial features.

## Requirements and setup

- Node.js 22 or newer (Node 24 is used in the current workspace)
- npm 11 or newer
- A MapLibre-compatible style URL; MapTiler is supported
- The CrimeNet inference API for production data

```bash
cd frontend
npm install
Copy-Item .env.example .env.local # PowerShell
npm run dev
```

Open `http://localhost:3000` for the public platform page. The analytical application is at `http://localhost:3000/explorer`, with diagnostics at `/model`. Production validation uses:

```bash
npm run format:check
npm run lint
npm run typecheck
npm test
npm run build
```

## Environment

| Variable                         | Purpose                                                                              |
| -------------------------------- | ------------------------------------------------------------------------------------ |
| `NEXT_PUBLIC_CRIMENET_DATA_MODE` | `api` for production live/forecast data or `fixture` for deterministic UI tests.     |
| `NEXT_PUBLIC_API_BASE_URL`       | CrimeSense API origin; defaults to `https://api.crimesense.ai` when it is omitted.   |
| `NEXT_PUBLIC_MAP_STYLE_URL`      | MapLibre style URL. May contain a public map-style token, but never a server secret. |
| `NEXT_PUBLIC_MAPTILER_KEY`       | Public browser key for MapTiler geocoding and optional satellite imagery.            |

Do not place private service credentials in `NEXT_PUBLIC_*`; those values are bundled for the browser. Keep machine-specific values in `.env.local` and review that file before publishing changes.

The Explorer's existing command palette uses `@maptiler/client` for debounced US address and place autocomplete. The current map center is passed only as a proximity bias, so a nationwide search remains possible. Selecting a result navigates the existing MapLibre camera; its normal `moveend` event triggers the CrimeNet viewport request and backend-selected H3 LOD. Production MapTiler keys should be restricted in MapTiler Cloud to the deployed origins (for example, local development and the approved production domains) rather than unrestricted or embedded as private server credentials.

## Architecture

```text
Next.js App Router / React 19
             │
             ├── Zustand ───────────── map camera-adjacent UI state
             │                         layers, selection, playback
             │
             ├── TanStack Query ────── cancellable, city-scoped server cache
             │               │
             │               └── Zod API boundary ── CrimeNet FastAPI
             │                                      predictions / metadata
             │
             └── MapLibre camera + labels
                         │
                         └── deck.gl interleaved overlay
                                  └── H3HexagonLayer (backend-selected r4–r9)
```

The surface is dynamically imported to avoid SSR/WebGL conflicts. City, timestamp, horizon, and selected H3 cell are encoded in URL search parameters. Query keys include city and time so a city transition cannot reuse the previous city's prediction surface. Expensive spatial data remains outside the React DOM.

## Complete platform and local contract

CrimeNet is a multi-system platform. Its distributed data layer uses Databricks, Apache Spark, Delta Lake, Unity Catalog, and Bronze–Silver–Gold lakehouse organization. Large historical feature operations have scanned approximately 2.3–2.6 billion rows, and materialized Parquet has reached roughly 200 GB. Dagster, Polars, DuckDB, Python, and Parquet also support orchestration and development workflows represented in this checkout.

The current repository contains the frontend plus a substantial local data/ML subsystem; it does not contain every cross-repository CrimeNet component. Absence from this checkout is not treated as absence from the complete platform.

The frontend uses facts defined by the Python repository rather than presentation placeholders:

- jurisdictions: Baltimore, Chicago, Dallas, Fort Worth, New York, San Francisco, Seattle, and Washington, DC;
- city-specific IANA time zones;
- H3 as a cross-layer spatial primitive, with the Explorer rendering backend-selected r4–r9 LOD cells derived from the canonical r9 model surface;
- model output: point-process intensity in events per cell-hour;
- validation year: 2024;
- full-v1 model: 63 inputs including city identity, six calendar fields, 27 weather/context fields, three lighting fields, and 26 leakage-safe crime-history fields;
- required feature concepts: temporal, lighting, weather, OSM/built environment, socioeconomic, local crime history, and neighbor crime history.

The complete platform also includes a Databricks lakehouse and broader online-serving direction. The browser consumes the separately running CrimeNet FastAPI service through a narrow adapter and keeps raw serving JSON out of UI components.

## Serving API and frontend domain

Live wire contracts are validated in `lib/api.ts` and `lib/inference/api-provider.ts`:

- `GET /health` provides status, current `snapshot_id`, `valid_utc_hour`, and mark readiness.
- `GET /api/v1/intensity/timeline` provides the rolling LIVE plus forecast snapshot indexes.
- `GET /api/v1/intensity/viewport` chooses an H3-r4 through H3-r9 LOD under the render budget and drives the map.
- `GET /api/v1/predict/cell/{h3}?top_k=87` provides one selected cell's current rate and full mark distribution.

### Frontend `PredictionResponse` domain

The live viewport response and deterministic fixture provider both adapt into this UI contract:

```ts
type PredictionResponse = {
  city: string;
  timestamp: string;
  horizonHours: number;
  unit: "events_per_cell_hour";
  modelVersion: string;
  resolution: number;
  aggregation?: "native_r9" | "sum_r9_child_intensity";
  cells: Array<{
    h3: string;
    // Total expected events/hour over the displayed H3 cell.
    intensity: number | null;
    // Mean r9-cell intensity used for zoom-stable map color.
    visualIntensity?: number | null;
    modeledR9Cells?: number;
    percentile: number | null;
    coverage: "full" | "partial" | "unsupported";
    missingReason: string | null;
    features: Array<{
      group:
        | "temporal"
        | "lighting"
        | "weather"
        | "osm"
        | "socioeconomic"
        | "crime_history"
        | "neighbor_history";
      available: boolean;
      observedAt?: string | null;
    }>;
  }>;
};
```

Zod enforces that an unsupported cell has `intensity: null`; a response containing `coverage: "unsupported"` and a numeric intensity is rejected. A full or partial cell must carry a numeric intensity. This preserves:

```text
missing coverage ≠ zero risk
no prediction ≠ zero prediction
unsupported geography ≠ safe geography
```

The current repository defines no cold-start model. Fixture mode therefore emits only `full` and `unsupported`; the adapter and UI can represent `partial` only if a future backend explicitly supplies it.

### Model metadata

In live mode the model page derives service/model readiness and version identity from `/health`. It continues to omit performance visualizations because the serving API does not expose validated metrics, explanations, or feature-freshness details.

Recommended future endpoints, not currently invoked, are viewport/time-bounded historical vector tiles and cell-level explanation/feature-freshness details. Do not return entire city event datasets as GeoJSON.

## Fixture mode

When `NEXT_PUBLIC_CRIMENET_DATA_MODE=fixture`, the interface displays `DEVELOPMENT FIXTURE` persistently. The adapter creates a deterministic H3 contract surface for interaction and visual testing; it is not real inference, is never random, and is never labelled live. Playwright explicitly sets this mode so browser tests stay independent of the serving process.

In `api` mode, the timeline drives a discrete LIVE through +24h forecast slider using exactly the entries returned by the service. Viewport cache identity includes both the timeline generation hour and selected valid UTC hour. Slider changes retain the previous surface while loading and prefetch only the two adjacent entries on each side for the current viewport. The same `valid_utc_hour` is sent to selected-cell mark inference, keeping map and inspector time-consistent. Coarse r4–r8 clicks still drill the camera toward finer data; only a confirmed r9 click invokes the timestamp-aware `/api/v1/predict/cell/{h3}` request.

## Performance and accessibility

- deck.gl handles thousands of H3 cells as GPU geometry; no cell becomes a DOM node.
- Queries are cancellable, city/time scoped, cached, and prefetch the next hour.
- Browser-only mapping code is route-split; model diagnostics do not load the map bundle.
- Map data transitions preserve the current view while avoiding cross-city placeholder reuse.
- Controls are semantic, keyboard reachable, visibly focused, responsive, and motion respects `prefers-reduced-motion`.
- Desktop uses floating analytical surfaces; mobile converts the inspector into a bottom sheet and preserves the map.

Keyboard shortcuts: `Ctrl/⌘ K` opens commands, arrows step time, Space toggles playback, and Escape closes the inspector.
