"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { cellToLatLng } from "h3-js";
import {
  AlertTriangle,
  Check,
  Copy,
  Database,
  Info,
  MapPin,
  Minus,
  Search,
  ShieldAlert,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { serviceHealthQueryKey } from "@/lib/api";
import type { PredictionResponse } from "@/lib/domain";
import { coverageLabel, featureLabels, formatIntensity, formatTimestamp } from "@/lib/format";
import {
  cellPredictionQueryKey,
  inferenceProvider,
  type CellPrediction,
  type FamilyPrediction,
  type SubtypePrediction,
} from "@/lib/inference";
import { getCity } from "@/lib/domain";
import { isNativeMarkCell, NATIVE_MARK_RESOLUTION } from "@/lib/map/lod";
import { CRIME_FAMILIES } from "@/lib/taxonomy";
import { useExplorerStore } from "@/stores/explorer-store";

type DistributionMetric = "probability" | "intensity";

const formatProbability = (value: number) => {
  const percent = value * 100;
  if (percent < 0.01) return "<0.01%";
  if (percent < 1) return `${percent.toFixed(2)}%`;
  return `${percent.toFixed(1)}%`;
};

const formatMarkIntensity = (value: number) =>
  value < 0.001 ? value.toFixed(5) : value < 0.01 ? value.toFixed(4) : value.toFixed(3);

function valueFor(item: FamilyPrediction | SubtypePrediction, metric: DistributionMetric) {
  return metric === "probability" ? item.conditionalProbability : item.intensity;
}

function DistributionValue({
  item,
  metric,
}: {
  item: FamilyPrediction | SubtypePrediction;
  metric: DistributionMetric;
}) {
  return (
    <>
      {metric === "probability"
        ? formatProbability(item.conditionalProbability)
        : formatMarkIntensity(item.intensity)}
    </>
  );
}

function DistributionRow({
  item,
  metric,
  max,
  active = false,
  onClick,
}: {
  item: FamilyPrediction | SubtypePrediction;
  metric: DistributionMetric;
  max: number;
  active?: boolean;
  onClick?: () => void;
}) {
  const subtype = "subtypeLabel" in item;
  const label = subtype ? item.subtypeLabel : item.familyLabel;
  const code = subtype ? item.subtypeCode : item.familyCode;
  return (
    <button
      className={`distribution-row ${active ? "active" : ""}`}
      onClick={onClick}
      type="button"
    >
      <span className="distribution-label">
        <strong>{label}</strong>
        <code>{code}</code>
      </span>
      <span className="distribution-value">
        <DistributionValue item={item} metric={metric} />
      </span>
      <span className="distribution-track" aria-hidden="true">
        <i style={{ width: `${Math.max(1.5, (valueFor(item, metric) / max) * 100)}%` }} />
      </span>
    </button>
  );
}

function AllTypesPanel({
  prediction,
  open,
  onOpenChange,
  onSelectFamily,
}: {
  prediction: CellPrediction;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSelectFamily: (code: string) => void;
}) {
  const [search, setSearch] = useState("");
  const [family, setFamily] = useState("all");
  const [sort, setSort] = useState<DistributionMetric>("probability");
  const rows = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return [...prediction.subtypeDistribution]
      .filter((item) => family === "all" || item.familyCode === family)
      .filter(
        (item) =>
          !needle ||
          `${item.subtypeLabel} ${item.subtypeCode} ${item.familyLabel}`
            .toLowerCase()
            .includes(needle),
      )
      .sort((left, right) => valueFor(right, sort) - valueFor(left, sort));
  }, [prediction, search, family, sort]);
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="types-overlay" />
        <Dialog.Content className="types-dialog" aria-describedby={undefined}>
          <header>
            <div>
              <small>CANONICAL MARK DISTRIBUTION</small>
              <Dialog.Title>All 87 modeled crime types</Dialog.Title>
            </div>
            <Dialog.Close className="icon-button quiet" aria-label="Close all crime types">
              <X size={16} />
            </Dialog.Close>
          </header>
          <div className="types-toolbar">
            <label>
              <Search size={13} />
              <input
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search modeled crime type..."
              />
            </label>
            <select
              aria-label="Filter by family"
              value={family}
              onChange={(event) => setFamily(event.target.value)}
            >
              <option value="all">All families</option>
              {CRIME_FAMILIES.map((item) => (
                <option value={item.code} key={item.code}>
                  {item.label}
                </option>
              ))}
            </select>
            <div className="types-sort" aria-label="Sort crime types">
              <button
                className={sort === "probability" ? "active" : ""}
                onClick={() => setSort("probability")}
              >
                PROB.
              </button>
              <button
                className={sort === "intensity" ? "active" : ""}
                onClick={() => setSort("intensity")}
              >
                INTENSITY
              </button>
            </div>
          </div>
          <div className="types-table-head">
            <span>TYPE / CODE</span>
            <span>FAMILY</span>
            <span>PROB.</span>
            <span>INTENSITY</span>
          </div>
          <div className="types-table">
            {rows.map((item) => (
              <button
                key={item.subtypeCode}
                onClick={() => {
                  onSelectFamily(item.familyCode);
                  onOpenChange(false);
                }}
              >
                <span>
                  <strong>{item.subtypeLabel}</strong>
                  <code>{item.subtypeCode}</code>
                </span>
                <span>{item.familyLabel}</span>
                <span>{formatProbability(item.conditionalProbability)}</span>
                <span>{formatMarkIntensity(item.intensity)}</span>
              </button>
            ))}
            {rows.length === 0 && (
              <p className="types-empty">No modeled type matches this filter.</p>
            )}
          </div>
          <footer>
            <span>{rows.length} of 87 classes</span>
            <span>P(type | modeled event) · λtype = λtotal × p(type)</span>
          </footer>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

export function Inspector({
  data,
  snapshotId,
}: {
  data?: PredictionResponse;
  snapshotId?: string;
}) {
  const selectedH3 = useExplorerStore((state) => state.selectedH3);
  const close = useExplorerStore((state) => state.selectCell);
  const cityId = useExplorerStore((state) => state.cityId);
  const [metric, setMetric] = useState<DistributionMetric>("probability");
  const [selectedFamilyCode, setSelectedFamilyCode] = useState<string | null>(null);
  const [showAllFamilies, setShowAllFamilies] = useState(false);
  const [allTypesOpen, setAllTypesOpen] = useState(false);
  const queryClient = useQueryClient();
  const nativeSelection = Boolean(
    selectedH3 && data?.resolution === NATIVE_MARK_RESOLUTION && isNativeMarkCell(selectedH3),
  );
  const cell = nativeSelection
    ? data?.cells.find((candidate) => candidate.h3 === selectedH3)
    : undefined;
  const request =
    selectedH3 && data && cell
      ? {
          cityId,
          h3: selectedH3,
          timestamp: data.timestamp,
          horizonHours: data.horizonHours,
          snapshotId: snapshotId ?? data.snapshotId,
          surfaceCell: cell,
        }
      : null;
  const predictionQuery = useQuery({
    queryKey: request ? cellPredictionQueryKey(request) : ["cell-prediction", "idle"],
    queryFn: ({ signal }) => inferenceProvider.getCellPrediction({ ...request!, signal }),
    enabled: Boolean(request),
    placeholderData: (previous) =>
      previous?.h3 === selectedH3 &&
      (!request?.snapshotId || previous.snapshotId === request.snapshotId)
        ? previous
        : undefined,
  });
  const snapshotMismatch = Boolean(
    request?.snapshotId &&
      predictionQuery.data?.snapshotId &&
      request.snapshotId !== predictionQuery.data.snapshotId,
  );
  const prediction = snapshotMismatch ? undefined : predictionQuery.data;
  useEffect(() => {
    if (snapshotMismatch) {
      void queryClient.invalidateQueries({ queryKey: serviceHealthQueryKey });
    }
  }, [queryClient, snapshotMismatch]);
  const rankedFamilies = useMemo(
    () =>
      [...(prediction?.familyDistribution ?? [])].sort(
        (left, right) => valueFor(right, metric) - valueFor(left, metric),
      ),
    [prediction, metric],
  );
  if (!selectedH3) return null;
  if (data && data.resolution < NATIVE_MARK_RESOLUTION) return null;
  if (!cell || !data)
    return (
      <aside className="inspector">
        <div className="inspector-header">
          <div>
            <small>SELECTED CELL</small>
            <strong>Outside loaded surface</strong>
          </div>
          <button className="icon-button" onClick={() => close(null)}>
            <X size={15} />
          </button>
        </div>
        <div className="empty-inspector">
          <MapPin size={22} />
          <p>This H3 cell is not present in the current response.</p>
          <span>No prediction has been inferred.</span>
        </div>
      </aside>
    );
  const city = getCity(cityId);
  const [latitude, longitude] = cellToLatLng(cell.h3);
  const coverage =
    cell.coverage === "unsupported" ? "unsupported" : (prediction?.coverage ?? cell.coverage);
  const unsupported = coverage === "unsupported";
  const effectiveFamilyCode = rankedFamilies.some((item) => item.familyCode === selectedFamilyCode)
    ? selectedFamilyCode
    : rankedFamilies[0]?.familyCode;
  const selectedFamily = rankedFamilies.find((item) => item.familyCode === effectiveFamilyCode);
  const children =
    prediction && selectedFamily
      ? prediction.subtypeDistribution
          .filter((item) => item.familyCode === selectedFamily.familyCode)
          .sort((left, right) => valueFor(right, metric) - valueFor(left, metric))
      : [];
  const topSubtype = prediction
    ? [...prediction.subtypeDistribution].sort(
        (left, right) => right.conditionalProbability - left.conditionalProbability,
      )[0]
    : undefined;
  const familyMax = Math.max(...rankedFamilies.map((item) => valueFor(item, metric)), 1e-9);
  const childMax = Math.max(...children.map((item) => valueFor(item, metric)), 1e-9);
  const temporalMax = Math.max(
    ...(prediction?.temporal ?? []).map((item) => item.totalIntensity),
    1e-9,
  );

  return (
    <aside className="inspector inspector-advanced" aria-label="Selected H3 cell inspector">
      <div className="inspector-header">
        <div>
          <small>H3 CELL INSPECTOR</small>
          <strong>
            {city.name} · {cell.h3}
          </strong>
        </div>
        <div>
          <button
            className="icon-button quiet"
            onClick={() => navigator.clipboard?.writeText(cell.h3)}
            aria-label="Copy H3 ID"
          >
            <Copy size={14} />
          </button>
          <button
            className="icon-button quiet"
            onClick={() => close(null)}
            aria-label="Close inspector"
          >
            <X size={15} />
          </button>
        </div>
      </div>
      <div className={`coverage-banner ${coverage}`}>
        <span>
          {unsupported ? (
            <ShieldAlert size={15} />
          ) : coverage === "partial" ? (
            <AlertTriangle size={15} />
          ) : (
            <Check size={15} />
          )}
        </span>
        <div>
          <small>INFERENCE STATUS</small>
          <strong>{coverageLabel[coverage]}</strong>
          {cell.missingReason && <p>{cell.missingReason}</p>}
        </div>
      </div>
      {(predictionQuery.isPending || snapshotMismatch) && (
        <div className="prediction-skeleton" aria-label="Loading cell prediction">
          <i />
          <i />
          <i />
          <i />
        </div>
      )}
      {predictionQuery.isError && (
        <section className="inspector-error" role="alert">
          <strong>Cell inference unavailable</strong>
          <p>{predictionQuery.error.message}</p>
          <button onClick={() => predictionQuery.refetch()}>RETRY</button>
        </section>
      )}
      {unsupported && !predictionQuery.isPending ? (
        <section className="unsupported-callout">
          <strong>Inference unavailable</strong>
          <p>
            CrimeNet does not have sufficient feature coverage for this cell. Missing data is not
            interpreted as zero intensity.
          </p>
        </section>
      ) : (
        prediction && (
          <>
            <section className="intensity-hero">
              <div
                className="hero-intensity"
                title="CrimeNet's estimated rate of modeled events at this location and time."
              >
                <small>
                  EVENT INTENSITY <Info size={10} />
                </small>
                <strong>{formatIntensity(prediction.totalIntensity)}</strong>
                <span>events / cell / hour</span>
              </div>
              <dl>
                <div title="Integrated event intensity over the selected prediction horizon.">
                  <dt>EXPECTED +{data.horizonHours}H</dt>
                  <dd>{formatMarkIntensity(prediction.integratedIntensity ?? 0)}</dd>
                </div>
                <div>
                  <dt>P(≥1 MODELED EVENT)</dt>
                  <dd>{formatProbability(prediction.eventProbability ?? 0)}</dd>
                </div>
              </dl>
            </section>
            {topSubtype && (
              <section className="top-type">
                <small>MOST LIKELY MODELED EVENT TYPE</small>
                <div>
                  <span>
                    <strong>{topSubtype.subtypeLabel}</strong>
                    <small>
                      {topSubtype.familyLabel} · {topSubtype.subtypeCode}
                    </small>
                  </span>
                  <span>
                    <strong>{formatProbability(topSubtype.conditionalProbability)}</strong>
                    <small>conditional probability</small>
                  </span>
                </div>
                <footer>
                  <span>SUBTYPE INTENSITY</span>
                  <strong>{formatMarkIntensity(topSubtype.intensity)}</strong>
                  <small>events / cell / hour</small>
                </footer>
              </section>
            )}
            <section className="inspector-section crime-mix">
              <div className="section-title">
                <span>CRIME MIX</span>
                <span title="Predicted crime-type distribution assuming a modeled event occurs.">
                  <Info size={11} />
                </span>
              </div>
              <div className="metric-toggle" role="group" aria-label="Distribution metric">
                <button
                  className={metric === "probability" ? "active" : ""}
                  aria-pressed={metric === "probability"}
                  onClick={() => setMetric("probability")}
                >
                  PROBABILITY
                </button>
                <button
                  className={metric === "intensity" ? "active" : ""}
                  aria-pressed={metric === "intensity"}
                  onClick={() => setMetric("intensity")}
                >
                  INTENSITY
                </button>
              </div>
              <p className="metric-definition">
                {metric === "probability"
                  ? "P(family | modeled event)"
                  : "λfamily · events / cell / hour"}
              </p>
              <div className="distribution-list">
                {rankedFamilies.slice(0, showAllFamilies ? 20 : 6).map((item) => (
                  <DistributionRow
                    key={item.familyCode}
                    item={item}
                    metric={metric}
                    max={familyMax}
                    active={item.familyCode === effectiveFamilyCode}
                    onClick={() => setSelectedFamilyCode(item.familyCode)}
                  />
                ))}
              </div>
              <button className="text-action" onClick={() => setShowAllFamilies((shown) => !shown)}>
                {showAllFamilies ? "SHOW TOP FAMILIES" : "SHOW ALL 20 FAMILIES"}
              </button>
            </section>
            {selectedFamily && (
              <section className="inspector-section subtype-drilldown">
                <div className="section-title">
                  <span>{selectedFamily.familyLabel.toUpperCase()}</span>
                  <span>{children.length} SUBTYPES</span>
                </div>
                <p>
                  {formatProbability(selectedFamily.conditionalProbability)} of the modeled-event
                  distribution
                </p>
                <div className="distribution-list">
                  {children.map((item) => (
                    <DistributionRow
                      key={item.subtypeCode}
                      item={item}
                      metric={metric}
                      max={childMax}
                    />
                  ))}
                </div>
                <button className="text-action primary" onClick={() => setAllTypesOpen(true)}>
                  VIEW ALL 87 →
                </button>
              </section>
            )}
            {prediction.temporal?.length ? (
              <section className="inspector-section temporal-distribution">
                <div className="section-title">
                  <span>TEMPORAL INTENSITY</span>
                  <span>−12H · NOW · +12H</span>
                </div>
                <div className="temporal-bars">
                  {prediction.temporal.map((point, index) => (
                    <i
                      key={point.timestamp}
                      className={index === 12 ? "now" : ""}
                      style={{
                        height: `${Math.max(8, (point.totalIntensity / temporalMax) * 100)}%`,
                      }}
                      title={`${formatTimestamp(point.timestamp, city.timezone, false)} · ${formatMarkIntensity(point.totalIntensity)}`}
                    />
                  ))}
                </div>
                <div className="temporal-axis">
                  <span>−12h</span>
                  <strong>NOW</strong>
                  <span>+12h</span>
                </div>
              </section>
            ) : prediction.provider.kind === "api" ? (
              <section className="inspector-section current-hour-note">
                <div className="section-title">
                  <span>CURRENT-HOUR INFERENCE</span>
                  <span>{prediction.snapshotId}</span>
                </div>
                <p>
                  The live service exposes one current hourly rate; no temporal curve is inferred.
                </p>
              </section>
            ) : null}
            <section className="inspector-section model-strip">
              <div>
                <small>MODEL</small>
                <code>{prediction.model.version}</code>
              </div>
              <span
                className={prediction.provider.kind === "fixture" ? "fixture-data" : "live-data"}
              >
                {prediction.provider.label}
              </span>
              <div>
                <small>CENTROID</small>
                <span>
                  {latitude.toFixed(4)}, {longitude.toFixed(4)}
                </span>
              </div>
            </section>
          </>
        )
      )}
      {cell.features.length > 0 && (
        <section className="inspector-section compact-features">
          <div className="section-title">
            <span>FEATURE COVERAGE</span>
            <Database size={12} />
          </div>
          <div className="feature-pills">
            {cell.features.map((feature) => (
              <span key={feature.group} className={feature.available ? "available" : "missing"}>
                {feature.available ? <Check size={10} /> : <Minus size={10} />}
                {featureLabels[feature.group]}
              </span>
            ))}
          </div>
        </section>
      )}
      {prediction && (
        <AllTypesPanel
          prediction={prediction}
          open={allTypesOpen}
          onOpenChange={setAllTypesOpen}
          onSelectFamily={setSelectedFamilyCode}
        />
      )}
    </aside>
  );
}
