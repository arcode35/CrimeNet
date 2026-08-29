"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowLeft,
  Check,
  Database,
  FileBarChart,
  Hexagon,
  Layers3,
  ShieldCheck,
} from "lucide-react";
import { getModelMetadata, isFixtureMode } from "@/lib/api";
import { CITIES } from "@/lib/domain";

export default function ModelPage() {
  const query = useQuery({
    queryKey: ["model-metadata"],
    queryFn: ({ signal }) => getModelMetadata(signal),
  });
  const model = query.data;
  return (
    <main className="model-page">
      <header className="model-nav">
        <div className="brand">
          <span className="brand-mark">
            <Hexagon size={17} />
          </span>
          <div>
            <strong>CRIMENET</strong>
            <small>MODEL OPERATIONS</small>
          </div>
        </div>
        <nav>
          <Link href="/explorer">
            <ArrowLeft size={14} /> Explorer
          </Link>
          <span>Model diagnostics</span>
        </nav>
        {isFixtureMode && (
          <span className="fixture-flag">
            <span /> CONTRACT METADATA
          </span>
        )}
      </header>
      <div className="model-grid">
        <section className="model-hero">
          <div className="eyebrow">ACTIVE MODEL / POINT PROCESS</div>
          <h1>{model?.name ?? "Loading model contract…"}</h1>
          <p>{model?.description}</p>
          <div className="model-status">
            <span>
              <i />
              <strong>{model?.status === "fixture" ? "REPOSITORY CONTRACT" : "ACTIVE"}</strong>
            </span>
            <code>{model?.version}</code>
          </div>
        </section>
        <aside className="model-identity">
          <small>MODEL IDENTITY</small>
          <dl>
            <div>
              <dt>Validation year</dt>
              <dd>{model?.validationYear ?? "—"}</dd>
            </div>
            <div>
              <dt>H3 resolution</dt>
              <dd>{model?.h3Resolution ?? "—"}</dd>
            </div>
            <div>
              <dt>Feature count</dt>
              <dd>{model?.featureCount ?? "—"}</dd>
            </div>
            <div>
              <dt>Output unit</dt>
              <dd>events / cell / hour</dd>
            </div>
          </dl>
        </aside>
        <section className="diagnostic-panel geography-panel">
          <div className="diagnostic-title">
            <span>
              <Layers3 size={15} /> SUPPORTED GEOGRAPHIES
            </span>
            <small>{CITIES.length} JURISDICTIONS</small>
          </div>
          <div className="city-matrix">
            {CITIES.map((city, index) => (
              <div key={city.id}>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <strong>{city.name}</strong>
                <small>{city.timezone}</small>
                <Check size={13} />
              </div>
            ))}
          </div>
        </section>
        <section className="diagnostic-panel contract-panel">
          <div className="diagnostic-title">
            <span>
              <Database size={15} /> FEATURE CONTRACT
            </span>
            <small>FULL_V1</small>
          </div>
          <div className="feature-groups">
            {[
              ["City identity", 1],
              ["Calendar", 6],
              ["Weather & context", 27],
              ["Lighting", 3],
              ["Crime history", 26],
            ].map(([label, count]) => (
              <div key={label}>
                <span>
                  <i style={{ width: `${Number(count) * 2.6}%` }} />
                </span>
                <strong>{label}</strong>
                <small>{count} features</small>
              </div>
            ))}
          </div>
          <p>
            <ShieldCheck size={14} /> Inference is eligible only where every required feature group
            is established. No cold-start model is defined by the repository.
          </p>
        </section>
        <section className="diagnostic-panel metrics-empty">
          <FileBarChart size={26} />
          <strong>Evaluation artifact API required</strong>
          <p>
            Training code emits validation metrics, feature importance, class metrics, and confusion
            matrices, but no service exposes them to the frontend. CrimeNet deliberately omits
            unverified charts here.
          </p>
          <code>GET /v1/model/metrics</code>
        </section>
      </div>
    </main>
  );
}
