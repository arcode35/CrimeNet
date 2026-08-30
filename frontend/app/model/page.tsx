import type { Metadata } from "next";
import Link from "next/link";
import {
  ArrowDownRight,
  ArrowLeft,
  ArrowRight,
  Clock3,
  CloudSun,
  Cpu,
  Database,
  Globe2,
  Grid3X3,
  Hexagon,
  Layers3,
  MoonStar,
  Server,
  ShieldCheck,
  type LucideIcon,
} from "lucide-react";
import "./model-page.css";

export const metadata: Metadata = {
  title: "Model System — CrimeSense",
  description:
    "Explore the CrimeSense data foundation, the CrimeNet two-stage XGBoost architecture, the national H3 feature system, and the rolling 24-hour forecast stack.",
};

const metrics = [
  ["15", "Cities"],
  ["12Y", "Crime data span"],
  ["16.7M", "Audited events"],
  ["180M+", "Training examples"],
  ["25.56M", "H3-r9 cells"],
  ["24H", "Forecast horizon"],
] as const;

const dataPlumbing = [
  {
    kicker: "01 / SOURCE SYSTEMS",
    title: "City crime feeds",
    copy: "Fifteen city-level public safety datasets collected across twelve years, each with its own schema, timestamps, location quality, and offense taxonomy.",
    items: ["CAD / RMS exports", "Open-data portals", "City-specific schemas"],
  },
  {
    kicker: "02 / STANDARDIZATION",
    title: "Canonical event normalization",
    copy: "Raw source records are ingested, cleaned, schema-aligned, and mapped into a unified offense vocabulary so the system can train nationally.",
    items: ["Bronze ingestion", "Schema alignment", "Canonical offense mapping"],
  },
  {
    kicker: "03 / SPINE",
    title: "Spatial-temporal event backbone",
    copy: "Every event is resolved into a consistent time basis and geospatial representation, forming the event spine that supports both modeling and auditability.",
    items: ["Timezone resolution", "H3 indexing", "Event spine + QA"],
  },
  {
    kicker: "04 / ENRICHMENT",
    title: "National feature joining",
    copy: "The event backbone is enriched with national context layers that describe place, time, weather, and solar state at the modeling hour.",
    items: ["OSM + built environment", "ACS socioeconomic", "Weather + solar + lighting"],
  },
  {
    kicker: "05 / MODEL ASSETS",
    title: "Training + serving contracts",
    copy: "The final system materializes model-ready tables for supervised learning and a serving contract used to build future-state snapshots for inference.",
    items: ["Training table", "Feature store", "Inference snapshots"],
  },
] as const;

const featureFamilies: ReadonlyArray<{
  icon: LucideIcon;
  title: string;
  count: string;
  description: string;
}> = [
  {
    icon: Grid3X3,
    title: "Built environment",
    count: "17",
    description:
      "Road density, intersections, buildings, POI mix, road composition, and land-use structure.",
  },
  {
    icon: Database,
    title: "Socioeconomic context",
    count: "8",
    description:
      "Population, income, poverty, unemployment, vacancy, tenure, age, and vehicle access.",
  },
  {
    icon: Clock3,
    title: "Local temporal state",
    count: "6",
    description: "Local hour and weekday plus cyclical encodings resolved in each cell’s timezone.",
  },
  {
    icon: CloudSun,
    title: "Forecast weather",
    count: "3",
    description:
      "Forecast temperature, relative humidity, and an explicit weather-availability signal.",
  },
  {
    icon: MoonStar,
    title: "Solar + lighting",
    count: "4",
    description:
      "Solar elevation, azimuth, daylight state, and categorical twilight or night context.",
  },
];

const infrastructure: ReadonlyArray<{
  icon: LucideIcon;
  label: string;
}> = [
  { icon: Database, label: "National feature stores" },
  { icon: CloudSun, label: "Forecast materialization" },
  { icon: Cpu, label: "GPU inference" },
  { icon: Layers3, label: "LOD generation" },
  { icon: Server, label: "Production serving" },
];

const forecastHours = Array.from({ length: 25 }, (_, hour) => hour);

function SectionHeader({
  number,
  eyebrow,
  title,
  description,
}: {
  number: string;
  eyebrow: string;
  title: string;
  description?: string;
}) {
  return (
    <header className="cs-section-header">
      <div className="cs-section-index">{number}</div>
      <div className="cs-section-heading-copy">
        <span>{eyebrow}</span>
        <h2>{title}</h2>
        {description ? <p>{description}</p> : null}
      </div>
    </header>
  );
}

function Stage({
  index,
  label,
  equation,
  title,
  description,
  facts,
}: {
  index: string;
  label: string;
  equation: string;
  title: string;
  description: string;
  facts: ReadonlyArray<[string, string]>;
}) {
  return (
    <article className="cs-stage">
      <div className="cs-stage-topline">
        <span>{index}</span>
        <span>{label}</span>
      </div>
      <div className="cs-stage-equation">{equation}</div>
      <h3>{title}</h3>
      <p>{description}</p>
      <dl>
        {facts.map(([key, value]) => (
          <div key={key}>
            <dt>{key}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </article>
  );
}

function FeatureRow({ icon: Icon, title, count, description }: (typeof featureFamilies)[number]) {
  return (
    <article className="cs-feature-row">
      <div className="cs-feature-icon">
        <Icon size={17} />
      </div>
      <div className="cs-feature-count">{count}</div>
      <div className="cs-feature-copy">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </article>
  );
}

export default function ModelPage() {
  return (
    <main className="cs-model-page">
      <div className="cs-grid-bg" aria-hidden="true" />

      <header className="cs-nav">
        <div className="cs-nav-inner">
          <Link href="/" className="cs-brand" aria-label="CrimeSense home">
            <span className="cs-brand-mark">
              <Hexagon size={17} strokeWidth={1.7} />
            </span>
            <span className="cs-brand-copy">
              <strong>CRIMESENSE</strong>
              <small>POWERED BY CRIMENET</small>
            </span>
          </Link>

          <nav aria-label="Model navigation" className="cs-nav-links">
            <Link href="/explorer">
              <ArrowLeft size={13} /> Explorer
            </Link>
            <span>Model system</span>
          </nav>

          <div className="cs-prod-state">
            <i aria-hidden="true" />
            Production architecture
          </div>
        </div>
      </header>

      <div className="cs-shell">
        <section className="cs-hero">
          <div className="cs-hero-main">
            <div className="cs-kicker">CRIMESENSE / NATIONAL RISK FORECASTING</div>
            <h1>
              Data infrastructure first,
              <span> forecasting system second.</span>
            </h1>
            <p>
              CrimeSense operationalizes twelve years of crime data across fifteen cities into a
              live 24-hour national H3 forecast. Under the interface is CrimeNet: a production data
              system that standardizes event feeds, builds model-ready feature contracts, and serves
              two-stage spatial risk inference at national scale.
            </p>

            <div className="cs-hero-actions">
              <Link href="/explorer" className="cs-primary-action">
                Open Explorer <ArrowRight size={15} />
              </Link>
              <a href="#data-foundation" className="cs-secondary-action">
                View data system <ArrowDownRight size={15} />
              </a>
            </div>
          </div>

          <aside className="cs-system-summary" aria-label="CrimeNet production system summary">
            <div className="cs-system-summary-head">
              <span>TECHNICAL ENGINE</span>
              <strong>CRIMENET</strong>
              <small>CN / PROD / R9</small>
            </div>
            <dl>
              <div>
                <dt>Coverage</dt>
                <dd>15 cities · 12 years</dd>
              </div>
              <div>
                <dt>Intensity</dt>
                <dd>XGBoost · Poisson</dd>
              </div>
              <div>
                <dt>Mark</dt>
                <dd>XGBoost · 87 classes</dd>
              </div>
              <div>
                <dt>Grid</dt>
                <dd>25.56M · H3 r9</dd>
              </div>
              <div>
                <dt>Forecast</dt>
                <dd>Live → +24H</dd>
              </div>
            </dl>
          </aside>
        </section>

        <section className="cs-metric-strip" aria-label="CrimeSense system scale">
          {metrics.map(([value, label]) => (
            <div key={label}>
              <strong>{value}</strong>
              <span>{label}</span>
            </div>
          ))}
        </section>

        <section className="cs-section" id="data-foundation">
          <SectionHeader
            number="01"
            eyebrow="DATA FOUNDATION"
            title="Data first. Modeling second."
            description="The production system starts with a long-horizon multi-city crime corpus, then progressively standardizes, audits, spatializes, enriches, and materializes it into model-ready and serving-ready assets. The chart below is the actual story of the system."
          />

          <div className="cs-plumbing-chart" aria-label="CrimeNet data plumbing diagram">
            {dataPlumbing.map((step, index) => (
              <div className="cs-plumbing-lane" key={step.title}>
                <article className="cs-plumbing-stage">
                  <span>{step.kicker}</span>
                  <h3>{step.title}</h3>
                  <p>{step.copy}</p>
                  <ul>
                    {step.items.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </article>
                {index < dataPlumbing.length - 1 ? (
                  <div className="cs-plumbing-arrow" aria-hidden="true">
                    <span />
                    <ArrowRight size={16} />
                  </div>
                ) : null}
              </div>
            ))}
          </div>

          <div className="cs-plumbing-summary">
            <div>
              <span>INPUT SCALE</span>
              <strong>15 cities · 12 years of crime data</strong>
              <small>
                Heterogeneous event feeds standardized into one national modeling system.
              </small>
            </div>
            <div>
              <span>TRAINING PRODUCT</span>
              <strong>16.7M audited events → 180M+ training examples</strong>
              <small>Event records become spatial-temporal supervised learning instances.</small>
            </div>
            <div>
              <span>SERVING PRODUCT</span>
              <strong>Feature contracts + future-state snapshots</strong>
              <small>
                Exactly the same system that builds training assets also powers inference.
              </small>
            </div>
          </div>
        </section>

        <section className="cs-section" id="architecture">
          <SectionHeader
            number="02"
            eyebrow="MODELING SYSTEM"
            title="Two models. One national risk field."
            description="Once the data system has produced consistent spatial-temporal features, CrimeNet splits the forecasting problem into expected activity and conditional event type."
          />

          <div className="cs-architecture-flow">
            <div className="cs-flow-source">
              <span>MODEL INPUT</span>
              <strong>Model-ready national training assets</strong>
              <small>
                Canonical events, aligned features, complete serving contracts, and audited
                spatial-temporal context.
              </small>
            </div>

            <div className="cs-flow-arrow" aria-hidden="true">
              <span />
              <ArrowRight size={17} />
            </div>

            <div className="cs-stage-grid">
              <Stage
                index="STAGE 01"
                label="INTENSITY"
                equation="λ(x,t)"
                title="How much activity?"
                description="Estimates expected event intensity for each canonical H3 cell under its spatial, temporal, environmental, and socioeconomic context."
                facts={[
                  ["MODEL", "XGBoost"],
                  ["OBJECTIVE", "Poisson / point-process"],
                  ["OUTPUT", "events / cell / hour"],
                ]}
              />
              <Stage
                index="STAGE 02"
                label="MARK"
                equation="P(mark | x,t)"
                title="What kind of event?"
                description="Given event context, estimates the conditional probability distribution across all modeled crime subtypes."
                facts={[
                  ["MODEL", "XGBoost"],
                  ["OBJECTIVE", "multi:softprob"],
                  ["OUTPUT", "87-class distribution"],
                ]}
              />
            </div>

            <div className="cs-flow-arrow" aria-hidden="true">
              <span />
              <ArrowRight size={17} />
            </div>

            <div className="cs-flow-output">
              <Globe2 size={18} />
              <span>MODEL OUTPUT</span>
              <strong>National H3 risk surface</strong>
              <small>
                A live spatial field of expected activity plus conditional subtype probabilities,
                refreshed over the forecast horizon.
              </small>
            </div>
          </div>
        </section>

        <section className="cs-section" id="feature-system">
          <SectionHeader
            number="03"
            eyebrow="FEATURE SYSTEM"
            title="Place context meets hourly state."
            description="Both production models share the same 38-feature serving contract: 25 static place features and 13 values rebuilt or resolved for the inference hour."
          />

          <div className="cs-feature-layout">
            <div className="cs-feature-list">
              {featureFamilies.map((feature) => (
                <FeatureRow key={feature.title} {...feature} />
              ))}
            </div>

            <div className="cs-context-diagram" aria-label="Static and dynamic context merge">
              <article className="cs-context-block">
                <span>STATIC / PLACE</span>
                <strong>Population · roads · socioeconomic context</strong>
                <p>
                  Slow-moving national layers that describe the neighborhood and the built
                  environment.
                </p>
              </article>
              <div className="cs-context-plus">+</div>
              <article className="cs-context-block">
                <span>DYNAMIC / HOUR</span>
                <strong>Time · weather · solar · lighting</strong>
                <p>
                  Fast-moving values rebuilt for the forecast hour under local time and future
                  environmental state.
                </p>
              </article>
              <div className="cs-context-arrow">→</div>
              <article className="cs-context-output">
                <span>MODEL STATE</span>
                <strong>λ(cell, hour)</strong>
                <small>
                  The same serving contract feeds intensity inference and downstream mark
                  estimation.
                </small>
              </article>
            </div>
          </div>
        </section>

        <section className="cs-section" id="spatial-inference">
          <SectionHeader
            number="04"
            eyebrow="NATIONAL SPATIAL INFERENCE"
            title="One model grid. Six serving resolutions."
            description="The model predicts at H3 resolution 9. Lower resolutions are deterministic render-time aggregations for viewing, not separate coarser models."
          />

          <div className="cs-spatial-flow">
            <div className="cs-spatial-primary">
              <span>CANONICAL MODEL GRID</span>
              <div className="cs-spatial-value">
                <strong>25.56M</strong>
                <small>H3 r9 cells covering the production inference universe.</small>
              </div>
            </div>

            <div className="cs-spatial-line" aria-hidden="true" />

            <div className="cs-spatial-inference">
              <span>CRIMENET INFERENCE</span>
              <strong>r9</strong>
              <small>Primary ML prediction resolution.</small>
            </div>

            <div className="cs-spatial-line" aria-hidden="true" />

            <div className="cs-lod-stack">
              <span>VIEWPORT SERVING</span>
              <div>
                {[9, 8, 7, 6, 5, 4].map((resolution) => (
                  <i key={resolution} className={resolution === 9 ? "active" : undefined}>
                    r{resolution}
                  </i>
                ))}
              </div>
              <small>
                Lower zoom levels are deterministic aggregates of the canonical risk field.
              </small>
            </div>
          </div>
        </section>

        <section className="cs-section" id="forecast-engine">
          <SectionHeader
            number="05"
            eyebrow="ROLLING FORECAST ENGINE"
            title="24 independently inferred future states."
            description="Every forecast hour receives a complete feature snapshot. The system is performing model inference over future environmental state, not merely interpolating one static map."
          />

          <div className="cs-forecast-panel">
            <div className="cs-forecast-topline">
              <strong>LIVE</strong>
              <span>INDEPENDENT HOURLY INFERENCE</span>
              <strong>+24H</strong>
            </div>

            <div className="cs-timeline" aria-label="Live through 24-hour forecast timeline">
              {forecastHours.map((hour) => (
                <div key={hour} className={hour === 0 ? "live" : undefined}>
                  <i />
                  <span>{hour === 0 ? "LIVE" : hour % 6 === 0 ? `+${hour}H` : ""}</span>
                </div>
              ))}
            </div>

            <div className="cs-forecast-input-grid">
              <div>
                <CloudSun size={17} />
                <span>
                  <strong>Forecast meteorology</strong>
                  <small>Temperature + humidity</small>
                </span>
              </div>
              <div>
                <Clock3 size={17} />
                <span>
                  <strong>Future local time</strong>
                  <small>Hour + weekday</small>
                </span>
              </div>
              <div>
                <MoonStar size={17} />
                <span>
                  <strong>Solar geometry</strong>
                  <small>Elevation + twilight</small>
                </span>
              </div>
              <div>
                <Grid3X3 size={17} />
                <span>
                  <strong>Static place context</strong>
                  <small>Same place, new hour</small>
                </span>
              </div>
            </div>
          </div>
        </section>

        <section className="cs-infrastructure">
          <div className="cs-infrastructure-copy">
            <span>POWERED BY CRIMENET</span>
            <h2>Infrastructure beneath the forecast.</h2>
            <p>
              The visible product is a risk map. Underneath it is a national geospatial ML stack
              that builds feature systems, materializes future-state inputs, executes inference, and
              serves multi-resolution output to the client.
            </p>
          </div>

          <div className="cs-infrastructure-pipeline" aria-label="CrimeNet infrastructure pipeline">
            {infrastructure.map(({ icon: Icon, label }, index) => (
              <div className="cs-infra-step" key={label}>
                <div>
                  <Icon size={16} />
                  <strong>{label}</strong>
                </div>
                {index < infrastructure.length - 1 ? (
                  <ArrowRight size={14} aria-hidden="true" />
                ) : null}
              </div>
            ))}
          </div>

          <footer className="cs-footer">
            <span>
              <ShieldCheck size={13} /> Statistical risk forecasting · spatial-temporal inference
            </span>
            <Link href="/explorer">
              Open CrimeSense Explorer <ArrowRight size={14} />
            </Link>
          </footer>
        </section>
      </div>
    </main>
  );
}
