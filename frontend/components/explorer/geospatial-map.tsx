"use client";

import { MapboxOverlay } from "@deck.gl/mapbox";
import { H3HexagonLayer } from "@deck.gl/geo-layers";
import type { PickingInfo } from "@deck.gl/core";
import maplibregl, { type Map as MapLibreMap } from "maplibre-gl";
import { useEffect, useRef, useState } from "react";
import type { City, PredictionCell, PredictionResponse } from "@/lib/domain";
import { formatIntensity } from "@/lib/format";
import { applyBasemapMode, ensureSatelliteLayer, SATELLITE_SOURCE_ID } from "@/lib/map/basemaps";
import { useExplorerStore } from "@/stores/explorer-store";

const RISK_STOPS = [
  [21, 31, 40],
  [35, 72, 86],
  [35, 118, 127],
  [75, 158, 134],
  [151, 190, 117],
  [229, 194, 100],
  [241, 133, 80],
  [218, 74, 72],
] as const;

function riskColor(percentile: number | null): [number, number, number, number] {
  const value = Math.max(0, Math.min(0.999, percentile ?? 0));
  const index = Math.min(RISK_STOPS.length - 1, Math.floor(value * RISK_STOPS.length));
  const [r, g, b] = RISK_STOPS[index];
  return [r, g, b, 186];
}

const coverageColor = (cell: PredictionCell): [number, number, number, number] =>
  cell.coverage === "full"
    ? [69, 148, 151, 138]
    : cell.coverage === "partial"
      ? [213, 171, 88, 165]
      : [105, 109, 116, 118];

export function GeospatialMap({
  city,
  data,
  isFetching,
}: {
  city: City;
  data?: PredictionResponse;
  isFetching: boolean;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const overlayRef = useRef<MapboxOverlay | null>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [styleRevision, setStyleRevision] = useState(0);
  const [satelliteError, setSatelliteError] = useState(false);
  const selectedH3 = useExplorerStore((state) => state.selectedH3);
  const layers = useExplorerStore((state) => state.layers);
  const mode = useExplorerStore((state) => state.mode);
  const basemapMode = useExplorerStore((state) => state.basemapMode);
  const basemapModeRef = useRef(basemapMode);
  const mapTilerKey = process.env.NEXT_PUBLIC_MAPTILER_KEY;
  useEffect(() => {
    basemapModeRef.current = basemapMode;
  }, [basemapMode]);
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style:
        process.env.NEXT_PUBLIC_MAP_STYLE_URL ||
        "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
      center: city.center,
      zoom: city.zoom,
      pitch: 0,
      bearing: -8,
      attributionControl: false,
      maxPitch: 68,
      minZoom: 3,
    });
    map.addControl(new maplibregl.AttributionControl({ compact: true }), "bottom-right");
    let fallbackApplied = false;
    const handleStyleReady = () => {
      ensureSatelliteLayer(map, mapTilerKey);
      applyBasemapMode(map, basemapModeRef.current);
      setStyleRevision((revision) => revision + 1);
      containerRef.current?.setAttribute("data-map-loaded", "true");
    };
    const handleMapError = (event: maplibregl.ErrorEvent) => {
      const sourceId = (event as maplibregl.ErrorEvent & { sourceId?: string }).sourceId;
      const message = event.error?.message ?? "";
      if (
        basemapModeRef.current === "satellite" &&
        (sourceId === SATELLITE_SOURCE_ID || message.includes("satellite-v4"))
      ) {
        applyBasemapMode(map, "dark");
        useExplorerStore.getState().setBasemapMode("dark");
        setSatelliteError(true);
        return;
      }
      if (fallbackApplied || map.isStyleLoaded()) return;
      fallbackApplied = true;
      containerRef.current?.setAttribute("data-map-fallback", "true");
      map.setStyle({
        version: 8,
        sources: {},
        layers: [
          {
            id: "fallback-background",
            type: "background",
            paint: { "background-color": "#0b1114" },
          },
        ],
      });
    };
    const handleMapIdle = () => {
      if (basemapModeRef.current === "satellite") {
        containerRef.current?.setAttribute("data-satellite-loaded", "true");
      }
    };
    map.on("style.load", handleStyleReady);
    map.on("error", handleMapError);
    map.on("idle", handleMapIdle);
    mapRef.current = map;
    return () => {
      map.off("style.load", handleStyleReady);
      map.off("error", handleMapError);
      map.off("idle", handleMapIdle);
      overlayRef.current?.finalize();
      overlayRef.current = null;
      map.remove();
      mapRef.current = null;
    };
    // Map intentionally mounts once; camera updates are handled separately.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || styleRevision === 0) return;
    const installed = ensureSatelliteLayer(map, mapTilerKey);
    const resolvedMode = basemapMode === "satellite" && installed ? "satellite" : "dark";
    if (basemapMode === "satellite" && !installed) {
      useExplorerStore.getState().setBasemapMode("dark");
    }
    containerRef.current?.removeAttribute("data-satellite-loaded");
    applyBasemapMode(map, resolvedMode);
    if (resolvedMode === "satellite") setSatelliteError(false);
    containerRef.current?.setAttribute("data-basemap-mode", resolvedMode);
    containerRef.current?.setAttribute(
      "data-satellite-layer-visible",
      String(resolvedMode === "satellite"),
    );
  }, [basemapMode, mapTilerKey, styleRevision]);

  useEffect(() => {
    mapRef.current?.flyTo({
      center: city.center,
      zoom: city.zoom,
      pitch: mode === "3d" ? 48 : 0,
      bearing: mode === "3d" ? -18 : -8,
      duration: 1050,
      essential: true,
    });
  }, [city, mode]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || styleRevision === 0 || !data) return;
    const firstLabelLayer = map.getStyle().layers?.find((layer) => layer.type === "symbol")?.id;
    const buildLayer = () =>
      new H3HexagonLayer<PredictionCell>({
        id: `crime-surface-${data.city}-${data.timestamp}-${layers.coverage}-${mode}`,
        // Resolve the active style's label stack after style.load. This avoids
        // a race where deck.gl targets a layer that MapLibre has not created.
        ...(firstLabelLayer ? { beforeId: firstLabelLayer } : {}),
        data: data.cells,
        getHexagon: (cell) => cell.h3,
        getFillColor: (cell) =>
          layers.coverage
            ? coverageColor(cell)
            : cell.coverage === "unsupported"
              ? [78, 82, 89, 85]
              : riskColor(cell.percentile),
        getLineColor: (cell) =>
          cell.h3 === selectedH3
            ? basemapMode === "satellite"
              ? [255, 255, 255, 255]
              : [225, 244, 241, 255]
            : cell.coverage === "unsupported"
              ? [143, 148, 156, 155]
              : basemapMode === "satellite"
                ? [4, 9, 12, 176]
                : [10, 18, 23, 92],
        getLineWidth: (cell) =>
          cell.h3 === selectedH3
            ? basemapMode === "satellite"
              ? 3.6
              : 2.8
            : cell.coverage === "unsupported"
              ? 0.7
              : basemapMode === "satellite"
                ? 0.62
                : 0.35,
        lineWidthUnits: "pixels",
        filled: layers.prediction || layers.coverage,
        stroked: true,
        extruded: mode === "3d" && !layers.coverage,
        getElevation: (cell) => (cell.intensity === null ? 0 : cell.intensity * 2450),
        elevationScale: mode === "3d" ? 1 : 0,
        opacity: isFetching ? 0.64 : basemapMode === "satellite" ? 0.78 : 0.92,
        pickable: true,
        autoHighlight: true,
        highlightColor: [225, 244, 241, 72],
        transitions: { getFillColor: 320, getElevation: 420, opacity: 220 },
        updateTriggers: {
          getFillColor: [layers.coverage, data.timestamp],
          getLineColor: [selectedH3, basemapMode],
          getLineWidth: [selectedH3, basemapMode],
          getElevation: [data.timestamp, mode],
        },
        onHover: (info: PickingInfo<PredictionCell>) => {
          const tooltip = tooltipRef.current;
          useExplorerStore.getState().hoverCell(info.object?.h3 ?? null);
          if (!tooltip) return;
          if (!info.object) {
            tooltip.style.opacity = "0";
            return;
          }
          const cell = info.object;
          tooltip.style.opacity = "1";
          tooltip.style.transform = `translate3d(${info.x + 14}px, ${info.y + 14}px, 0)`;
          tooltip.innerHTML = `<div class="tooltip-kicker">${cell.coverage === "full" ? "FULL MODEL" : cell.coverage === "partial" ? "LIMITED COVERAGE" : "INFERENCE UNAVAILABLE"}</div><strong>${formatIntensity(cell.intensity)}</strong><span>${cell.intensity === null ? "No prediction returned" : "events / cell / hour"}</span><code>${cell.h3}</code>`;
        },
        onClick: (info: PickingInfo<PredictionCell>) =>
          useExplorerStore.getState().selectCell(info.object?.h3 ?? null),
      });
    if (!overlayRef.current) {
      overlayRef.current = new MapboxOverlay({ interleaved: true, layers: [buildLayer()] });
      map.addControl(overlayRef.current);
    } else overlayRef.current.setProps({ layers: [buildLayer()] });
    containerRef.current?.setAttribute("data-prediction-layer", "ready");
  }, [data, selectedH3, layers, mode, isFetching, styleRevision, basemapMode]);

  return (
    <div className="map-stage">
      <div ref={containerRef} className="map-canvas" />
      <div ref={tooltipRef} className="map-tooltip" aria-hidden="true" />
      {satelliteError && (
        <div className="basemap-alert" role="status">
          Satellite imagery unavailable. Using standard basemap.
        </div>
      )}
    </div>
  );
}
