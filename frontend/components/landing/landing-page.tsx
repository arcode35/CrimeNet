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
import { CITIES } from "@/lib/domain";

const architecture = [
  ["01", "Raw sources", "Municipal records · OSM · weather · ACS · solar"],
  ["02", "Lakehouse", "Databricks · Spark · Delta · Unity Catalog"],
  ["03", "Medallion", "Bronze → canonical Silver → ML-ready Gold"],
  ["04", "Features", "H3 context · spatial joins · leakage-safe history"],
  ["05", "Training", "XGBoost · Optuna · MLflow · GPU where applicable"],
  ["06", "Online path", "API · feature retrieval · coverage · inference"],
  ["07", "Explorer", "Next.js · MapLibre camera · deck.gl GPU surface"],
] as const;

const featureFamilies = [
  [
    "Historical",
    "26",
    "6h / 24h / 7d / 28d cell, city, and k=1 activity; recency and relative state",
  ],
  ["Temporal", "06", "Local hour and day-of-week with cyclical encodings"],
  ["Environment", "02", "Canonical temperature and relative humidity observations"],
  ["Lighting", "03", "Solar elevation, daylight flag, and lighting condition"],
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
      ["Databricks + Spark", "Distributed lakehouse processing at billion-row scan scale"],
      ["Delta + Unity Catalog", "Medallion tables, catalog organization, and governance"],
      ["Dagster + Polars", "Asset orchestration and lazy columnar development pipelines"],
      ["DuckDB", "Local spatial joins and boundary processing"],
      ["Parquet · Python · SQL", "Portable columnar data and transformation languages"],
    ],
  ],
  [
    "Geospatial + sources",
    [
      ["Uber H3", "Spatial joins, neighborhoods, features, inference, and rendering"],
      ["OSM + Geofabrik", "Built-environment features"],
      ["Open-Meteo + pvlib", "Weather observations and deterministic solar state"],
      ["Census ACS / TIGER", "Socioeconomic context and jurisdiction masks"],
    ],
  ],
  [
    "Machine learning",
    [
      ["XGBoost", "Current point-process baseline and mark classifiers"],
      ["Optuna", "Journal-backed distributed hyperparameter search"],
      ["MLflow", "Experiment tracking and artifact lineage"],
      ["PyTorch", "CrimeNet Omega research implementation"],
    ],
  ],
  [
    "Serving + interface",
    [
      ["FastAPI · Redis · Flink", "Broader online-serving and streaming direction"],
      ["Next.js 16 + React 19", "Application routing and analytical surfaces"],
      ["MapLibre GL + deck.gl", "Synchronized GPU map and H3 rendering"],
      ["TanStack Query", "Cancellable, city-scoped server state"],
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
          <strong>CRIMENET</strong>
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
            CrimeNet models where and when reported crime intensity changes by combining historical
            incidents with spatial, temporal, environmental, infrastructural, and socioeconomic
            context.
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
          <strong>13M+</strong>
          <span>
            CANONICAL
            <br />
            CRIME RECORDS
          </span>
        </div>
        <div>
          <strong>79M+</strong>
          <span>
            GENERATED
            <br />
            ML OBSERVATIONS
          </span>
        </div>
        <div>
          <strong>2.3–2.6B</strong>
          <span>
            ROWS SCANNED IN
            <br />
            LARGE SPARK JOBS
          </span>
        </div>
        <div>
          <strong>~200 GB</strong>
          <span>
            MATERIALIZED
            <br />
            PARQUET DATA
          </span>
        </div>
        <div>
          <strong>08</strong>
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
          copy="CrimeNet is the connected system required to make spatiotemporal modeling valid: data lineage, canonical geography, leakage-safe features, experiment tracking, explicit eligibility, and GPU-backed analytical delivery."
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
          copy="CrimeNet spans a Databricks lakehouse, offline feature and model systems, an online-serving direction, and the GPU Explorer. The current checkout contains only part of that multi-system platform."
        />
        <div className="architecture-flow">
          {architecture.map(([number, title, copy], index) => (
            <article key={title} tabIndex={0}>
              <div className="arch-node">
                <span>{number}</span>
                {index < architecture.length - 1 && <i />}
              </div>
              <small>
                {index === 5 ? "ONLINE SYSTEM" : index === 6 ? "INTERFACE" : "DATA / ML PLATFORM"}
              </small>
              <h3>{title}</h3>
              <p>{copy}</p>
            </article>
          ))}
        </div>
        <div className="architecture-note">
          <Workflow size={16} />
          <span>
            Unity Catalog organizes lakehouse assets. Spark and Delta execute distributed medallion
            processing. Dagster, Polars, and DuckDB support orchestration and development workflows.
            Typed contracts protect the final interface.
          </span>
        </div>
      </section>

      <section className="landing-section data-section" id="technology">
        <SectionHead
          index="03"
          eyebrow="DATA ENGINEERING"
          title="Heterogeneous urban data, one analytical contract."
          copy="CrimeNet's Databricks lakehouse uses Apache Spark, Delta Lake, Unity Catalog, and a Bronze–Silver–Gold design. Dagster, Polars, DuckDB, Python, SQL, and Parquet support complementary orchestration and development workflows."
        />
        <div className="medallion-grid">
          <article>
            <small>BRONZE / SOURCE-FAITHFUL</small>
            <h3>Ingest with identity intact.</h3>
            <p>
              More than 13.5 million historical source records enter source-aligned tables with
              lineage and partition identity. Weather and socioeconomic inputs retain explicit
              provenance.
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
            <code>crime_canonical_v1_3</code>
          </article>
          <article>
            <small>GOLD / MODEL-READY</small>
            <h3>Build context without leakage.</h3>
            <p>
              Gold materializations generate tens of millions of ML observations by combining
              exact-cell, neighbor, city, temporal, weather, lighting, OSM, and ACS context.
            </p>
            <code>model_row_id · cell-seconds exposure</code>
          </article>
        </div>
        <div className="source-lineage">
          <span>SOURCE LINEAGE</span>
          {[
            "Municipal open data",
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
          copy="Offline scores are meaningless when future state leaks into historical examples. CrimeNet constructs rolling history strictly before each model timestamp and keeps the 2025+ test partition sealed during training and validation."
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
            <small>FEATURE SYSTEM / FULL_V1</small>
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
          copy="The current configured baseline uses XGBoost with a Poisson point-process objective. Exposure-weighted integration samples estimate λ(x,t) in events per cell-hour; separate mark classifiers model event categories conditional on an event."
        />
        <div className="model-flow">
          <div className="feature-vector">
            <small>63-D FEATURE VECTOR</small>
            {Array.from({ length: 63 }, (_, i) => (
              <i key={i} style={{ opacity: 0.18 + (i % 9) / 12 }} />
            ))}
          </div>
          <ArrowRight size={18} />
          <div className="model-core">
            <span>XGBOOST</span>
            <strong>POINT PROCESS</strong>
            <small>hist · depth 12 configuration</small>
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
            Validation code evaluates a future year using models trained exclusively on earlier
            information. Archived mark-classifier artifacts exist; performance claims are
            intentionally deferred until an authoritative serving contract exposes the selected
            model and metrics.
          </p>
        </div>
      </section>

      <section className="landing-section coverage-section">
        <SectionHead
          index="07"
          eyebrow="INFERENCE COVERAGE"
          title="Eligibility before estimation."
          copy="A geographic coordinate is not automatically a valid model input. CrimeNet must establish every required feature group before the full model can return an intensity."
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
              <Hexagon size={14} /> CRIMENET
            </span>
            <small>INFERENCE EXPLORER</small>
            <i />
          </div>
          <div className="preview-panel">
            <small>SUPPORTED REGION</small>
            <strong>Chicago</strong>
            <span>MODEL JURISDICTION · H3 R9</span>
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
          Open CrimeNet Explorer <ArrowRight size={17} />
        </Link>
      </section>

      <section className="landing-section geography-section">
        <SectionHead
          index="09"
          eyebrow="SUPPORTED GEOGRAPHIES"
          title="Eight municipal systems, one spatial vocabulary."
          copy="CrimeNet normalizes heterogeneous municipal records into a shared schema while preserving city identity and local time. Jurisdiction support does not imply that every H3 cell has complete feature coverage."
        />
        <div className="city-field">
          <div className="city-orbit">
            <Globe2 size={180} strokeWidth={0.45} />
            {CITIES.map((city, index) => (
              <i key={city.id} style={{ "--angle": `${index * 45}deg` } as CSSProperties}>
                <span />
              </i>
            ))}
          </div>
          <div className="city-list">
            {CITIES.map((city, index) => (
              <div key={city.id}>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <strong>{city.name}</strong>
                <small>{city.timezone}</small>
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
            eyebrow="CRIMENET OMEGA"
            title="A neural marked point-process direction."
            copy="The repository contains an experimental PyTorch architecture—not a deployed replacement for the XGBoost baseline. Omega-0 conditions a continuous-time marked point process on the same contextual feature contract."
          />
          <div className="research-spec">
            <span>
              <small>ARCHITECTURE</small>Context-only marked point process
            </span>
            <span>
              <small>INTENSITY</small>Softplus · events per cell-hour
            </span>
            <span>
              <small>MARK HEAD</small>Canonical subtype code
            </span>
            <span>
              <small>TRAINING CONFIG</small>CUDA · bfloat16 · AdamW
            </span>
          </div>
          <p className="research-boundary">
            Graph state, Hawkes dynamics, raw event history, multiscale memory, and latent spatial
            components are explicitly disabled in Omega-0. They are research directions, not product
            claims.
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
          copy="CrimeNet's strongest guarantees are the ones that prevent an attractive interface from overstating what the data and model can support."
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
        <small>CRIMENET / DATA → CONTEXT → INTENSITY</small>
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
            <strong>CRIMENET</strong>
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
