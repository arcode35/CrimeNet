import type { CSSProperties } from "react";
import Link from "next/link";
import {
  ArrowDown,
  ArrowRight,
  Check,
  ChevronRight,
  CircleOff,
  Cpu,
  Database,
  Globe2,
  Grid3X3,
  Hexagon,
  Layers3,
  Map,
  Network,
  Orbit,
  ShieldCheck,
  Sparkles,
  Workflow,
} from "lucide-react";

const jurisdictions = [
  ["atlanta", "Atlanta", "America/New_York"],
  ["baltimore", "Baltimore", "America/New_York"],
  ["chandler_az", "Chandler, AZ", "America/Phoenix"],
  ["chicago", "Chicago", "America/Chicago"],
  ["dallas", "Dallas", "America/Chicago"],
  ["denver", "Denver", "America/Denver"],
  ["fort_worth", "Fort Worth", "America/Chicago"],
  ["los_angeles_county_sheriff", "Los Angeles County Sheriff", "America/Los_Angeles"],
  ["marin_county_sheriff_ca", "Marin County Sheriff, CA", "America/Los_Angeles"],
  ["montgomery_county_md", "Montgomery County, MD", "America/New_York"],
  ["new_york", "New York City", "America/New_York"],
  ["san_francisco", "San Francisco", "America/Los_Angeles"],
  ["seattle", "Seattle", "America/Los_Angeles"],
  ["sonoma_county_sheriff_ca", "Sonoma County Sheriff, CA", "America/Los_Angeles"],
  ["washington_dc", "Washington, DC", "America/New_York"],
] as const;

const architecture = [
  ["01", "Raw sources", "Municipal + county records · OSM · weather · ACS · solar"],
  ["02", "Versioned lake", "S3 · Delta / Parquet · immutable snapshot lineage"],
  ["03", "Canonical spine", "Bronze → Silver → leakage-safe Gold event spine"],
  ["04", "Integration", "H3 support sampling · exposure weights · temporal support"],
  ["05", "Feature contract", "Point-in-time H3 · environment · causal event history"],
  ["06", "Training", "XGBoost point process · geographic CV · GPU acceleration"],
  ["07", "Explorer", "FastAPI · Next.js · MapLibre · deck.gl GPU surface"],
] as const;

const featureFamilies = [
  [
    "Causal history",
    "35",
    "Canonical temporal-history columns in the event-spine contract, built strictly from prior events",
  ],
  ["Baseline temporal", "06", "Local hour and day-of-week with cyclical encodings"],
  [
    "Weather + solar",
    "07",
    "Temperature, humidity, availability, solar geometry, daylight, and lighting state",
  ],
  [
    "Built environment",
    "17",
    "POI, road, intersection, building, land-use, and urban-mix densities",
  ],
  [
    "Socioeconomic",
    "08",
    "ACS population, income, age, poverty, employment, housing, tenure, and mobility context",
  ],
] as const;

const technology = [
  [
    "Data platform",
    [
      ["Dagster + Polars", "Asset orchestration and lazy columnar feature construction"],
      ["S3 + Delta / Parquet", "Immutable snapshots, partitioned storage, and reproducible lineage"],
      ["Spark / Databricks", "Used for earlier distributed ingestion and large-scale feature assembly"],
      ["DuckDB", "Local spatial joins and boundary processing"],
      ["Python + SQL", "Transformation, audit, and analytical interfaces"],
    ],
  ],
  [
    "Geospatial + sources",
    [
      ["Uber H3", "Stable cells for joins, support sampling, features, inference, and rendering"],
      ["OSM + Geofabrik", "Built-environment features"],
      ["Open-Meteo + pvlib", "Weather observations and deterministic solar state"],
      ["Census ACS / TIGER", "Socioeconomic context and jurisdiction masks"],
    ],
  ],
  [
    "Machine learning",
    [
      ["XGBoost", "Current Poisson point-process baseline and geographic-CV experiments"],
      ["Optuna", "Distributed hyperparameter search and reproducible study state"],
      ["PyTorch", "Neural marked point-process research track"],
      ["CUDA", "GPU acceleration for large training and evaluation runs"],
    ],
  ],
  [
    "Serving + interface",
    [
      ["FastAPI", "Local inference API and typed model-serving boundary"],
      ["Next.js 16 + React 19", "Application routing and analytical surfaces"],
      ["MapLibre GL + deck.gl", "Synchronized GPU map and H3 rendering"],
      ["TanStack Query", "Cancellable, jurisdiction-scoped server state"],
      ["Zustand + Zod", "Local UI state and runtime contract validation"],
    ],
  ],
] as const;

const principles = [
  [
    "Temporal correctness",
    "Historical predictions can only use information established before their model timestamp.",
  ],
  [
    "Spatial consistency",
    "Heterogeneous locations are normalized into stable H3 cells and authoritative jurisdiction masks.",
  ],
  [
    "Reproducibility",
    "Versioned configurations, deterministic samples, run IDs, and artifacts preserve experiment identity.",
  ],
  [
    "Explicit coverage",
    "Missing feature coverage can never masquerade as zero intensity or a safe region.",
  ],
  [
    "Scalable computation",
    "Lazy columnar scans, partitioned Delta tables, and GPU rendering keep large work off the DOM.",
  ],
  [
    "Typed boundaries",
    "Runtime schemas protect the interface from malformed or semantically invalid inference responses.",
  ],
] as const;

function SectionHead({
  index,
  eyebrow,
  title,
  copy,
}: {
  index: string;
  eyebrow: string;
  title: string;
  copy: string;
}) {
  return (
    <div className="landing-section-head">
      <span className="section-index">{index}</span>
      <div>
        <small>{eyebrow}</small>
        <h2>{title}</h2>
        <p>{copy}</p>
      </div>
    </div>
  );
}

function HeroSurface() {
  return (
    <div className="hero-visual" aria-label="Abstract H3 intensity surface">
      <div className="hero-map-lines" />
      <div className="hero-coordinates">
        <span>41.8781° N</span>
        <span>87.6298° W</span>
        <span>H3 · R9</span>
      </div>
      <div className="hero-hex-grid">
        {Array.from({ length: 126 }, (_, index) => {
          const x = index % 14;
          const y = Math.floor(index / 14);
          const d1 = Math.hypot(x - 5.2, y - 3.8);
          const d2 = Math.hypot(x - 9.7, y - 6.1);
          const heat = Math.max(0, Math.min(5, Math.round(5 - Math.min(d1, d2) * 1.35)));
          return (
            <i
              key={index}
              data-heat={heat}
              style={{ "--delay": `${(x + y) * 24}ms` } as CSSProperties}
            />
          );
        })}
      </div>
      <div className="hero-reticle">
        <span />
        <span />
      </div>
      <div className="surface-readout">
        <small>PREDICTED INTENSITY</small>
        <strong>λ(x,t)</strong>
        <span>events / cell / hour</span>
      </div>
    </div>
  );
}

function PublicNav() {
  return (
    <header className="public-nav">
      <Link href="/" className="brand">
        <span className="brand-mark">
          <Hexagon size={17} strokeWidth={1.5} />
        </span>
        <div>
          <strong>CRIMESENSE</strong>
          <small>SPATIOTEMPORAL INTELLIGENCE</small>
        </div>
      </Link>
      <nav aria-label="Landing navigation">
        <a href="#overview">Overview</a>
        <a href="#technology">Technology</a>
        <a href="#architecture">Architecture</a>
        <a href="#research">Research</a>
      </nav>
      <div className="public-nav-actions">
        <Link href="/model">Model</Link>
        <Link href="/explorer" className="nav-cta">
          Open explorer <ArrowRight size={13} />
        </Link>
      </div>
    </header>
  );
}

export function LandingPage() {
  return (
    <main className="landing-page">
      <PublicNav />
      <section className="landing-hero" id="overview">
        <div className="hero-copy">
          <div className="hero-kicker">
            <span /> GEOSPATIAL MACHINE-LEARNING PLATFORM
          </div>
          <h1>
            <span>SPATIOTEMPORAL</span>
            <br />
            INTELLIGENCE.
          </h1>
          <p>
            CrimeSense models how reported crime intensity changes across space and time using a
            leakage-safe event spine, sampled point-process support, and spatial, temporal,
            environmental, infrastructural, and socioeconomic context.
          </p>
          <div className="hero-actions">
            <Link href="/explorer" className="primary-action">
              Open explorer <ArrowRight size={14} />
            </Link>
            <a href="#architecture" className="secondary-action">
              View architecture <ArrowDown size={13} />
            </a>
          </div>
          <div className="hero-disclaimer">
            <ShieldCheck size={13} />
            <span>Statistical intensity modeling. Explicit coverage. No claim of certainty.</span>
          </div>
        </div>
        <HeroSurface />
        <a className="scroll-cue" href="#scale">
          <span>SCROLL TO TRACE THE SYSTEM</span>
          <ArrowDown size={13} />
        </a>
      </section>
      <section className="scale-strip" id="scale">
        <div>
          <strong>17M+</strong>
          <span>
            RAW CRIME
            <br />
            RECORDS INGESTED
          </span>
        </div>
        <div>
          <strong>15.95M</strong>
          <span>
            CANONICAL EVENT
            <br />
            SPINE RECORDS
          </span>
        </div>
        <div>
          <strong>180M+</strong>
          <span>
            POINT-PROCESS
            <br />
            MODEL EXAMPLES
          </span>
        </div>
        <div>
          <strong>74,689</strong>
          <span>
            UNIQUE EVENT-SPINE
            <br />
            H3 CELLS
          </span>
        </div>
        <div>
          <strong>15</strong>
          <span>
            SUPPORTED
            <br />
            JURISDICTIONS
          </span>
        </div>
      </section>

      <section className="landing-section definition-section">
        <SectionHead
          index="01"
          eyebrow="SYSTEM DEFINITION"
          title="More than a prediction model."
          copy="CrimeSense is the public analytical system built on CrimeNet's current 15-jurisdiction data and ML pipeline: immutable lineage, canonical geography, leakage-safe features, explicit eligibility, and GPU-backed analytical delivery."
        />
        <div className="definition-stack">
          {[
            [Database, "DATA ENGINEERING", "Municipal schemas become one canonical event spine."],
            [
              Grid3X3,
              "FEATURE SYSTEM",
              "Every cell and timestamp receives spatially aligned context.",
            ],
            [
              Cpu,
              "MACHINE LEARNING",
              "Point-process objectives estimate event intensity, not deterministic outcomes.",
            ],
            [
              ShieldCheck,
              "INFERENCE CONTRACT",
              "Coverage is established before an estimate can be displayed.",
            ],
            [
              Map,
              "GPU INTERFACE",
              "Analytical geometry flows into WebGL instead of React DOM nodes.",
            ],
          ].map(([Icon, title, copy], index) => {
            const C = Icon as typeof Database;
            return (
              <article key={String(title)}>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <C size={18} />
                <h3>{String(title)}</h3>
                <p>{String(copy)}</p>
              </article>
            );
          })}
        </div>
      </section>

      <section className="landing-section architecture-section" id="architecture">
        <SectionHead
          index="02"
          eyebrow="END-TO-END ARCHITECTURE"
          title="A traceable path from source to surface."
          copy="The current system spans immutable S3/Parquet snapshots, a canonical event spine, exposure-weighted integration sampling, point-in-time feature enrichment, XGBoost training, a FastAPI serving boundary, and the GPU Explorer."
        />
        <div className="architecture-flow">
          {architecture.map(([number, title, copy], index) => (
            <article key={title} tabIndex={0}>
              <div className="arch-node">
                <span>{number}</span>
                {index < architecture.length - 1 && <i />}
              </div>
              <small>
                {index === 6 ? "INTERFACE" : index === 5 ? "ML PLATFORM" : "DATA PLATFORM"}
              </small>
              <h3>{title}</h3>
              <p>{copy}</p>
            </article>
          ))}
        </div>
        <div className="architecture-note">
          <Workflow size={16} />
          <span>
            Dagster and Polars orchestrate the current immutable-snapshot pipeline. S3 with
            Delta/Parquet stores versioned data products, while frozen lineage ties event, integration,
            environmental, and final-model snapshots together. Typed contracts protect the interface.
          </span>
        </div>
      </section>

      <section className="landing-section data-section" id="technology">
        <SectionHead
          index="03"
          eyebrow="DATA ENGINEERING"
          title="Heterogeneous urban data, one analytical contract."
          copy="CrimeSense uses a Bronze–Silver–Gold data contract, but the current pipeline is no longer coupled to one managed compute platform. Dagster, Polars, S3, Delta/Parquet, DuckDB, Python, and SQL carry the active snapshot and feature workflow."
        />
        <div className="medallion-grid">
          <article>
            <small>BRONZE / SOURCE-FAITHFUL</small>
            <h3>Ingest with identity intact.</h3>
            <p>
              More than 17 million municipal and county crime records have been ingested across
              the expanded source footprint, with source identity, lineage, and time semantics
              preserved before canonicalization.
            </p>
            <code>_ingestion_run_id · _ingested_at_utc</code>
          </article>
          <article>
            <small>SILVER / CANONICAL</small>
            <h3>Normalize before joining.</h3>
            <p>
              Offense mappings, units, timestamps, H3 identifiers, OSM values, and socioeconomic
              periods are validated and projected into stable schemas.
            </p>
            <code>source schema → canonical event contract</code>
          </article>
          <article>
            <small>GOLD / MODEL-READY</small>
            <h3>Build context without leakage.</h3>
            <p>
              The current point-process dataset expands observed events with exposure-weighted
              integration support to more than 180 million model examples, enriched with temporal,
              weather, lighting, OSM, ACS, and causal history context.
            </p>
            <code>event rows · integration rows · exposure weights</code>
          </article>
        </div>
        <div className="source-lineage">
          <span>SOURCE LINEAGE</span>
          {[
            "Municipal + county open data",
            "OpenStreetMap / Geofabrik",
            "Open-Meteo archive",
            "Census ACS 5-year",
            "pvlib solar state",
          ].map((source) => (
            <div key={source}>
              <i />
              <strong>{source}</strong>
            </div>
          ))}
        </div>
      </section>

      <section className="landing-section temporal-section">
        <SectionHead
          index="04"
          eyebrow="TEMPORAL CORRECTNESS"
          title="The future is not a feature."
          copy="Offline scores are meaningless when future state leaks into historical examples. CrimeSense constructs causal history strictly before each model timestamp, freezes support by split, and keeps the 2025+ test partition sealed during training and validation."
        />
        <div className="leakage-visual">
          <div className="leakage-labels">
            <span>PAST / ELIGIBLE</span>
            <strong>MODEL TIME · t</strong>
            <span>FUTURE / FORBIDDEN</span>
          </div>
          <div className="leakage-track">
            <i className="past" />
            <b />
            <i className="future" />
          </div>
          <div className="leakage-events">
            {[8, 18, 29, 43, 57, 68, 82, 91].map((position, index) => (
              <i
                key={position}
                className={index > 4 ? "forbidden" : ""}
                style={{ left: `${position}%` }}
              />
            ))}
          </div>
          <div className="leakage-detail">
            <div>
              <Check size={14} />
              <span>
                Prior incidents
                <br />
                Weather observations
                <br />
                Calendar and solar state
              </span>
            </div>
            <code>FEATURE TIME ≤ PREDICTION TIME</code>
            <div>
              <CircleOff size={14} />
              <span>
                Future incidents
                <br />
                Future aggregates
                <br />
                Future feature state
              </span>
            </div>
          </div>
        </div>
      </section>

      <section className="landing-section geo-feature-section">
        <div className="geo-engine">
          <SectionHead
            index="05"
            eyebrow="GEOSPATIAL ENGINE"
            title="The city becomes a graph of stable cells."
            copy="H3 is an architectural primitive across spatial joins, aggregation, neighborhoods, model observations, inference, and rendering. The Explorer renders backend-selected H3-r4 through H3-r9 LOD cells derived from one canonical r9 inference surface."
          />
          <div className="geo-diagram">
            <div className="geo-rings">
              <Hexagon size={116} />
              <span>
                <Hexagon size={42} />
              </span>
            </div>
            <div className="geo-steps">
              <span>CITY BOUNDARY</span>
              <ChevronRight size={13} />
              <span>H3 R9 CELL</span>
              <ChevronRight size={13} />
              <span>K=1 NEIGHBORHOOD</span>
              <ChevronRight size={13} />
              <span>λ(x,t)</span>
            </div>
          </div>
        </div>
        <div className="feature-engine">
          <div className="feature-title">
            <small>FEATURE CONTRACT / CURRENT</small>
            <strong>Context at every cell and time.</strong>
          </div>
          <div className="feature-matrix">
            {featureFamilies.map(([name, count, copy]) => (
              <article key={name}>
                <span>{count}</span>
                <div>
                  <h3>{name}</h3>
                  <p>{copy}</p>
                </div>
                <i style={{ width: `${Math.max(12, Number(count) * 3.2)}%` }} />
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="landing-section model-section">
        <SectionHead
          index="06"
          eyebrow="MACHINE LEARNING"
          title="Event intensity as a point process."
          copy="The current baseline uses XGBoost with a Poisson point-process objective over exposure-weighted integration samples. The active geographic-CV baseline uses 37 numeric features plus one categorical lighting feature; richer causal-history columns remain available in the final model contract."
        />
        <div className="model-flow">
          <div className="feature-vector">
            <small>38-FEATURE BASELINE INPUT</small>
            {Array.from({ length: 38 }, (_, i) => (
              <i key={i} style={{ opacity: 0.18 + (i % 9) / 12 }} />
            ))}
          </div>
          <ArrowRight size={18} />
          <div className="model-core">
            <span>XGBOOST</span>
            <strong>POINT PROCESS</strong>
            <small>Poisson objective · CUDA hist</small>
          </div>
          <ArrowRight size={18} />
          <div className="lambda-output">
            <strong>λ(x,t)</strong>
            <span>events / cell / hour</span>
          </div>
        </div>
        <div className="validation-split">
          <div className="split-head">
            <span>CHRONOLOGICAL VALIDATION</span>
            <code>TEST ACCESS: FALSE</code>
          </div>
          <div className="split-bars">
            <div className="train">
              <span>TRAIN</span>
              <small>2014–2023</small>
            </div>
            <div className="validation">
              <span>VALIDATION</span>
              <small>2024</small>
            </div>
            <div className="test">
              <span>TEST</span>
              <small>2025–2026-07-24</small>
            </div>
          </div>
          <p>
            Current geographic-CV runs use deterministic training and validation samples across
            five held-out geographic folds. These bars show the global split policy; each source is
            further clipped to its documented temporal support. The test split remains untouched
            during model selection and is reserved for final evaluation.
          </p>
        </div>
      </section>

      <section className="landing-section coverage-section">
        <SectionHead
          index="07"
          eyebrow="INFERENCE COVERAGE"
          title="Eligibility before estimation."
          copy="A geographic coordinate is not automatically a valid model input. CrimeSense establishes required feature state before displaying intensity; the current event-spine audit reports 99.969% modeled coverage while preserving missingness explicitly."
        />
        <div className="coverage-machine">
          <div className="coverage-request">
            <Network size={18} />
            <span>LOCATION REQUEST</span>
            <code>lat · lon · time</code>
          </div>
          <ArrowRight size={16} />
          <div className="coverage-check">
            <small>FEATURE COVERAGE</small>
            {["History", "Neighbor history", "Weather", "Lighting", "OSM", "Socioeconomic"].map(
              (item) => (
                <span key={item}>
                  <Check size={10} />
                  {item}
                </span>
              ),
            )}
          </div>
          <ArrowRight size={16} />
          <div className="coverage-outcomes">
            <span className="full">
              <i />
              FULL MODEL
            </span>
            <span className="partial">
              <i />
              PARTIAL · ONLY IF DEFINED
            </span>
            <span className="unsupported">
              <i />
              UNSUPPORTED
            </span>
          </div>
        </div>
        <div className="coverage-invariants">
          <strong>MISSING DATA ≠ ZERO</strong>
          <strong>NO PREDICTION ≠ ZERO RISK</strong>
          <strong>UNSUPPORTED ≠ SAFE</strong>
        </div>
      </section>

      <section className="landing-section frontend-section">
        <SectionHead
          index="08"
          eyebrow="GPU ANALYTICAL INTERFACE"
          title="Model output enters the rendering pipeline."
          copy="MapLibre owns the geographic camera and label stack. deck.gl interleaves H3 geometry beneath those labels, keeping thousands of cells in GPU-backed layers instead of React DOM."
        />
        <div className="gpu-flow">
          <div>
            <Database size={19} />
            <span>H3 RESPONSE</span>
            <small>typed · cancellable · city-scoped</small>
          </div>
          <i />
          <div>
            <Layers3 size={19} />
            <span>DECK.GL</span>
            <small>H3HexagonLayer</small>
          </div>
          <i />
          <div>
            <Cpu size={19} />
            <span>GPU</span>
            <small>fill · extrusion · picking</small>
          </div>
          <i />
          <div>
            <Map size={19} />
            <span>MAPLIBRE</span>
            <small>camera · streets · labels</small>
          </div>
        </div>
        <div className="explorer-preview">
          <div className="preview-top">
            <span>
              <Hexagon size={14} /> CRIMESENSE
            </span>
            <small>INFERENCE EXPLORER</small>
            <i />
          </div>
          <div className="preview-panel">
            <small>SUPPORTED REGION</small>
            <strong>Chicago</strong>
            <span>MODEL JURISDICTION · H3 R9 SURFACE</span>
            <small>PREDICTION TIME</small>
            <code>AUG 21, 2024 · 17:00 CDT</code>
          </div>
          <div className="preview-map">
            {Array.from({ length: 88 }, (_, index) => (
              <i key={index} data-heat={(index * 7 + Math.floor(index / 11) * 3) % 6} />
            ))}
            <strong>Chicago</strong>
          </div>
          <div className="preview-legend">
            <span>LOW</span>
            <i />
            <span>HIGHER</span>
          </div>
          <div className="preview-rail">
            <button>▶</button>
            <span />
            <span />
            <span />
            <span />
            <span />
          </div>
          <div className="preview-callouts">
            <span>GPU H3 SURFACE</span>
            <span>TEMPORAL EXPLORATION</span>
            <span>EXPLICIT COVERAGE</span>
          </div>
        </div>
        <Link href="/explorer" className="large-explorer-link">
          Open CrimeSense Explorer <ArrowRight size={17} />
        </Link>
      </section>

      <section className="landing-section geography-section">
        <SectionHead
          index="09"
          eyebrow="SUPPORTED GEOGRAPHIES"
          title="Fifteen jurisdictions, one spatial vocabulary."
          copy="CrimeSense normalizes heterogeneous municipal and county records into one event contract while preserving source identity and local time. The current event spine spans 15 jurisdictions and 74,689 unique H3 cells; support still does not imply complete covariate coverage for every cell-time."
        />
        <div className="city-field">
          <div className="city-orbit">
            <Globe2 size={180} strokeWidth={0.45} />
            {jurisdictions.map(([id], index) => (
              <i
                key={id}
                style={{ "--angle": `${index * (360 / jurisdictions.length)}deg` } as CSSProperties}
              >
                <span />
              </i>
            ))}
          </div>
          <div className="city-list">
            {jurisdictions.map(([id, name, timezone], index) => (
              <div key={id}>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <strong>{name}</strong>
                <small>{timezone}</small>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="landing-section technology-section">
        <SectionHead
          index="10"
          eyebrow="TECHNOLOGY MATRIX"
          title="Every technology has a role."
          copy="The inventory below reflects imported dependencies and implemented code paths—not an aspirational logo cloud."
        />
        <div className="technology-matrix">
          {technology.map(([group, items]) => (
            <article key={group}>
              <h3>{group}</h3>
              {items.map(([name, reason]) => (
                <div key={name} tabIndex={0}>
                  <strong>{name}</strong>
                  <span>{reason}</span>
                </div>
              ))}
            </article>
          ))}
        </div>
      </section>

      <section className="landing-section research-section" id="research">
        <div className="research-label">
          <Sparkles size={15} /> RESEARCH ARCHITECTURE · NOT PRODUCTION
        </div>
        <div>
          <SectionHead
            index="11"
            eyebrow="CRIMENET Ω / RESEARCH TRACK"
            title="Beyond the boosted-tree baseline."
            copy="CrimeNet Ω remains a research track rather than the selected serving baseline. It explores neural marked point-process formulations over the same leakage-safe spatial, temporal, and contextual contracts."
          />
          <div className="research-spec">
            <span>
              <small>ARCHITECTURE</small>Neural marked point-process research
            </span>
            <span>
              <small>INTENSITY</small>Continuous-time event intensity
            </span>
            <span>
              <small>MARK SPACE</small>Canonical offense taxonomy
            </span>
            <span>
              <small>STATUS</small>Experimental · not serving baseline
            </span>
          </div>
          <p className="research-boundary">
            Neural history, graph structure, Hawkes-style excitation, multiscale state, and
            reporting-process components remain research directions. They are not presented as
            properties of the current XGBoost serving baseline.
          </p>
        </div>
        <div className="omega-visual">
          <Orbit size={160} strokeWidth={0.5} />
          <span>
            Ω<small>0</small>
          </span>
          <i />
          <i />
          <i />
          <i />
        </div>
      </section>

      <section className="landing-section principles-section">
        <SectionHead
          index="12"
          eyebrow="ENGINEERING PRINCIPLES"
          title="Correctness is part of the product."
          copy="CrimeSense's strongest guarantees are the ones that prevent an attractive interface from overstating what the data and model can support."
        />
        <div className="principle-grid">
          {principles.map(([title, copy], index) => (
            <article key={title}>
              <span>{String(index + 1).padStart(2, "0")}</span>
              <h3>{title}</h3>
              <p>{copy}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="final-cta">
        <div className="final-grid" aria-hidden="true">
          {Array.from({ length: 60 }, (_, index) => (
            <i key={index} />
          ))}
        </div>
        <small>CRIMESENSE / DATA → CONTEXT → INTENSITY</small>
        <h2>
          FROM RAW URBAN DATA
          <br />
          TO SPATIOTEMPORAL INFERENCE.
        </h2>
        <p>Inspect the model surface, temporal state, and coverage contract directly.</p>
        <div>
          <Link href="/explorer" className="primary-action">
            Open explorer <ArrowRight size={14} />
          </Link>
          <Link href="/model" className="secondary-action">
            View model <ChevronRight size={13} />
          </Link>
        </div>
      </section>
      <footer className="landing-footer">
        <div className="brand">
          <span className="brand-mark">
            <Hexagon size={16} />
          </span>
          <div>
            <strong>CRIMESENSE</strong>
            <small>SPATIOTEMPORAL INTELLIGENCE</small>
          </div>
        </div>
        <p>
          Statistical modeling of reported event intensity. Designed for analytical use with
          explicit data-coverage semantics.
        </p>
        <div>
          <Link href="/explorer">Explorer</Link>
          <Link href="/model">Model</Link>
          <a href="#overview">Top ↑</a>
        </div>
      </footer>
    </main>
  );
}
