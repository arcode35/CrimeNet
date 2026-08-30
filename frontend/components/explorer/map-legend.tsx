"use client";

import { AlertTriangle, Info } from "lucide-react";
import type { PredictionResponse } from "@/lib/domain";
import { useExplorerStore } from "@/stores/explorer-store";

export function MapLegend({ data, error }: { data?: PredictionResponse; error: Error | null }) {
  const coverage = useExplorerStore((state) => state.layers.coverage);
  const hasPartial = data?.cells.some((cell) => cell.coverage === "partial");
  if (error) return null;
  return (
    <aside className="map-legend" aria-label="Map legend">
      <div className="legend-title">
        <span>
          {coverage
            ? "MODEL COVERAGE"
            : data?.source === "live"
              ? "LOCAL MODEL INTENSITY"
              : "PREDICTED INTENSITY"}
        </span>
        <Info size={12} />
      </div>
      {coverage ? (
        <>
          <div className="coverage-legend">
            <span>
              <i className="full" /> Full
            </span>
            {hasPartial && (
              <span>
                <i className="partial" /> Limited
              </span>
            )}
            <span>
              <i className="unsupported" /> Unsupported
            </span>
          </div>
          <p>
            <AlertTriangle size={11} /> Unsupported does not mean zero risk.
          </p>
        </>
      ) : (
        <>
          <div className="gradient-bar" />
          <div className="legend-scale">
            <span>Lower</span>
            <span>Higher</span>
          </div>
          <div className="legend-unit">
            <span>UNIT</span>
            <strong>
              {data?.source === "live"
                ? "events / r9-equivalent cell / hour"
                : data?.unit === "events_per_cell_hour"
                  ? "events / cell / hour"
                  : "Establishing…"}
            </strong>
          </div>
        </>
      )}
      {data && (
        <div className="legend-lod">
          <span>
            <small>SPATIAL RESOLUTION</small>
            <strong>H3-R{data.resolution}</strong>
          </span>
          <small>
            {data.resolution < 9 ? "Aggregated from modeled r9 cells" : "Native model cells"}
          </small>
        </div>
      )}
    </aside>
  );
}
