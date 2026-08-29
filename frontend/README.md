# CrimeNet frontend

CrimeNet's frontend is a map-first geospatial model-operations interface. MapLibre owns the basemap and camera; deck.gl renders the H3 resolution-9 analytical surface directly on the GPU. React owns controls and small analytical views, never individual spatial features.

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

| Variable                       | Purpose                                                                                      |
| ------------------------------ | -------------------------------------------------------------------------------------------- |
| `NEXT_PUBLIC_CRIMENET_API_URL` | FastAPI origin. If absent, the application enters visibly labelled development-fixture mode. |
| `NEXT_PUBLIC_MAP_STYLE_URL`    | MapLibre style URL. May contain a public map-style token, but never a server secret.         |

Do not place private service credentials in `NEXT_PUBLIC_*`; those values are bundled for the browser. `.env.local` is git-ignored.

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
                                  └── H3HexagonLayer (resolution 9)
```

The surface is dynamically imported to avoid SSR/WebGL conflicts. City, timestamp, horizon, and selected H3 cell are encoded in URL search parameters. Query keys include city and time so a city transition cannot reuse the previous city's prediction surface. Expensive spatial data remains outside the React DOM.

## Complete platform and local contract

CrimeNet is a multi-system platform. Its distributed data layer uses Databricks, Apache Spark, Delta Lake, Unity Catalog, and Bronze–Silver–Gold lakehouse organization. Large historical feature operations have scanned approximately 2.3–2.6 billion rows, and materialized Parquet has reached roughly 200 GB. Dagster, Polars, DuckDB, Python, and Parquet also support orchestration and development workflows represented in this checkout.

The current repository contains the frontend plus a substantial local data/ML subsystem; it does not contain every cross-repository CrimeNet component. Absence from this checkout is not treated as absence from the complete platform.

The frontend uses facts defined by the Python repository rather than presentation placeholders:

- jurisdictions: Baltimore, Chicago, Dallas, Fort Worth, New York, San Francisco, Seattle, and Washington, DC;
- city-specific IANA time zones;
- H3 as a cross-layer spatial primitive, with workflows spanning resolutions 6, 8, and 9 and the Explorer currently rendering resolution 9;
- model output: point-process intensity in events per cell-hour;
- validation year: 2024;
- full-v1 model: 63 inputs including city identity, six calendar fields, 27 weather/context fields, three lighting fields, and 26 leakage-safe crime-history fields;
- required feature concepts: temporal, lighting, weather, OSM/built environment, socioeconomic, local crime history, and neighbor crime history.

The complete platform also includes a Databricks lakehouse and broader online-serving direction. This checkout does **not** contain the FastAPI/OpenAPI implementation, viewport incident queries, cell-level explanations, or an explicitly deployed model record, so the frontend keeps those precise behaviors behind documented contracts.

## Required backend API

The only assumed production contracts are isolated in `lib/api.ts`. A backend implementation must expose:

### `GET /v1/predictions`

Query parameters: `city`, ISO-8601 `timestamp`, and `horizon_hours`.

```ts
type PredictionResponse = {
  city: string;
  timestamp: string;
  horizonHours: number;
  unit: "events_per_cell_hour";
  modelVersion: string;
  cells: Array<{
    h3: string;
    intensity: number | null;
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

### `GET /v1/model`

Returns model name/version, deployment status, validation year, H3 resolution, exact feature count, intensity unit, supported cities, and optionally validated metrics/global feature importance. The `/model` route omits performance visualizations until this contract is backed by a service.

Recommended future endpoints, not currently invoked, are viewport/time-bounded historical vector tiles and cell-level explanation/feature-freshness details. Do not return entire city event datasets as GeoJSON.

## Fixture mode

When `NEXT_PUBLIC_CRIMENET_API_URL` is empty, the interface displays `DEVELOPMENT FIXTURE` persistently. The adapter creates a deterministic H3 contract surface for interaction and visual testing; it is not real inference, is never random, and is never labelled live. Replace the environment variable to remove fixture mode—no UI changes are needed.

## Performance and accessibility

- deck.gl handles thousands of H3 cells as GPU geometry; no cell becomes a DOM node.
- Queries are cancellable, city/time scoped, cached, and prefetch the next hour.
- Browser-only mapping code is route-split; model diagnostics do not load the map bundle.
- Map data transitions preserve the current view while avoiding cross-city placeholder reuse.
- Controls are semantic, keyboard reachable, visibly focused, responsive, and motion respects `prefers-reduced-motion`.
- Desktop uses floating analytical surfaces; mobile converts the inspector into a bottom sheet and preserves the map.

Keyboard shortcuts: `Ctrl/⌘ K` opens commands, arrows step time, Space toggles playback, and Escape closes the inspector.
