"use client";

import { MapboxOverlay } from "@deck.gl/mapbox";
import { H3HexagonLayer } from "@deck.gl/geo-layers";
import type { PickingInfo } from "@deck.gl/core";
import maplibregl, { type Map as MapLibreMap } from "maplibre-gl";
import { useEffect, useRef, useState } from "react";
import type { ViewportBounds } from "@/lib/api";
import type { City, PredictionCell, PredictionResponse } from "@/lib/domain";
import { formatDetailedIntensity, formatIntensity } from "@/lib/format";
import { applyBasemapMode, ensureSatelliteLayer, SATELLITE_SOURCE_ID } from "@/lib/map/basemaps";
import {
  getCellVisualizationIntensity,
  getCellVisualizationScore,
  resolveMapCellClick,
} from "@/lib/map/lod";
import type { MapNavigation } from "@/lib/map/navigation";
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
  onViewportChange,
  onMapReady,
}: {
  city: City;
  data?: PredictionResponse;
  isFetching: boolean;
  onViewportChange?: (viewport: ViewportBounds) => void;
  onMapReady?: (navigation: MapNavigation | null) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const overlayRef = useRef<MapboxOverlay | null>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const destinationMarkerRef = useRef<maplibregl.Marker | null>(null);
  const [styleRevision, setStyleRevision] = useState(0);
  const [satelliteError, setSatelliteError] = useState(false);
  const selectedH3 = useExplorerStore((state) => state.selectedH3);
  const layers = useExplorerStore((state) => state.layers);
  const mode = useExplorerStore((state) => state.mode);
  const basemapMode = useExplorerStore((state) => state.basemapMode);
  const basemapModeRef = useRef(basemapMode);
  const onViewportChangeRef = useRef(onViewportChange);
  const onMapReadyRef = useRef(onMapReady);
  const mapTilerKey = process.env.NEXT_PUBLIC_MAPTILER_KEY;
  useEffect(() => {
    basemapModeRef.current = basemapMode;
  }, [basemapMode]);
  useEffect(() => {
    onViewportChangeRef.current = onViewportChange;
  }, [onViewportChange]);
  useEffect(() => {
    onMapReadyRef.current = onMapReady;
  }, [onMapReady]);
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style:
        process.env.NEXT_PUBLIC_MAP_STYLE_URL ||
        "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
      center: city.center,
      zoom: city.zoom,
      pitch: mode === "3d" ? 48 : 0,
      bearing: mode === "3d" ? -18 : -8,
      attributionControl: false,
      maxPitch: 68,
      minZoom: 3,
    });
    map.addControl(new maplibregl.AttributionControl({ compact: true }), "bottom-right");
    let fallbackApplied = false;
    const emitViewport = () => {
      const bounds = map.getBounds();
      const center = map.getCenter();
      containerRef.current?.setAttribute("data-map-center-longitude", center.lng.toFixed(5));
      containerRef.current?.setAttribute("data-map-center-latitude", center.lat.toFixed(5));
      onViewportChangeRef.current?.({
        west: bounds.getWest(),
        south: bounds.getSouth(),
        east: bounds.getEast(),
        north: bounds.getNorth(),
        zoom: map.getZoom(),
      });
    };
    const handleStyleReady = () => {
      ensureSatelliteLayer(map, mapTilerKey);
      applyBasemapMode(map, basemapModeRef.current);
      setStyleRevision((revision) => revision + 1);
      containerRef.current?.setAttribute("data-map-loaded", "true");
      emitViewport();
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
    map.on("moveend", emitViewport);
    mapRef.current = map;
    const placeDestinationMarker = (center: [number, number]) => {
      destinationMarkerRef.current?.remove();
      const markerElement = document.createElement("div");
      markerElement.className = "search-destination-marker";
      markerElement.setAttribute("aria-hidden", "true");
      destinationMarkerRef.current = new maplibregl.Marker({
        element: markerElement,
        anchor: "center",
      })
        .setLngLat(center)
        .addTo(map);
    };
    onMapReadyRef.current?.({
      getCenter: () => {
        const center = map.getCenter();
        return { longitude: center.lng, latitude: center.lat };
      },
      flyToLocation: ({ center, zoom, duration }) => {
        placeDestinationMarker(center);
        map.flyTo({ center, zoom, duration, essential: true });
      },
      fitToBounds: ({ bounds, maxZoom, duration }) => {
        const center: [number, number] = [
          (bounds[0][0] + bounds[1][0]) / 2,
          (bounds[0][1] + bounds[1][1]) / 2,
        ];
        placeDestinationMarker(center);
        map.fitBounds(bounds, { padding: 84, maxZoom, duration, essential: true });
      },
    });
    return () => {
      onMapReadyRef.current?.(null);
      map.off("style.load", handleStyleReady);
      map.off("error", handleMapError);
      map.off("idle", handleMapIdle);
      map.off("moveend", emitViewport);
      overlayRef.current?.finalize();
      overlayRef.current = null;
      destinationMarkerRef.current?.remove();
      destinationMarkerRef.current = null;
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
    if (!map || styleRevision === 0) return;
    if (!data) {
      overlayRef.current?.setProps({ layers: [] });
      containerRef.current?.removeAttribute("data-prediction-layer");
      containerRef.current?.removeAttribute("data-h3-resolution");
      containerRef.current?.removeAttribute("data-snapshot-timestamp");
      return;
    }
    const firstLabelLayer = map.getStyle().layers?.find((layer) => layer.type === "symbol")?.id;
    const buildLayer = () =>
      new H3HexagonLayer<PredictionCell>({
        id: `crime-surface-${data.city}-${data.timestamp}-r${data.resolution}-${layers.coverage}-${mode}`,
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
              : data.source === "live"
                ? riskColor(getCellVisualizationScore(cell))
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
        getElevation: (cell) => (getCellVisualizationIntensity(cell) ?? 0) * 2450,
        elevationScale: mode === "3d" ? 1 : 0,
        opacity: isFetching ? 0.64 : basemapMode === "satellite" ? 0.78 : 0.92,
        pickable: true,
        autoHighlight: true,
        highlightColor: [225, 244, 241, 72],
        transitions: { getFillColor: 320, getElevation: 420, opacity: 220 },
        updateTriggers: {
          getFillColor: [layers.coverage, data.timestamp, data.resolution],
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
          tooltip.innerHTML =
            data.resolution < 9
              ? `<div class="tooltip-kicker">AGGREGATED MODEL CELL · H3-R${data.resolution}</div><strong>${formatIntensity(cell.intensity)}</strong><span>expected modeled events / hour</span><div class="tooltip-secondary"><b>${formatDetailedIntensity(cell.visualIntensity ?? null)}</b><span>mean r9-cell intensity</span></div><span>${(cell.modeledR9Cells ?? 0).toLocaleString()} modeled r9 cells</span><code>${cell.h3}</code>`
              : `<div class="tooltip-kicker">${cell.coverage === "full" ? "FULL MODEL · H3-R9" : cell.coverage === "partial" ? "LIMITED COVERAGE · H3-R9" : "INFERENCE UNAVAILABLE · H3-R9"}</div><strong>${formatIntensity(cell.intensity)}</strong><span>${cell.intensity === null ? "No prediction returned" : "events / cell / hour"}</span><code>${cell.h3}</code>`;
        },
        onClick: (info: PickingInfo<PredictionCell>) => {
          if (!info.object) return;
          const action = resolveMapCellClick(info.object.h3, data.resolution, map.getZoom());
          if (action.kind === "select") {
            useExplorerStore.getState().selectCell(action.h3);
            return;
          }
          useExplorerStore.getState().selectCell(null);
          map.easeTo({
            center: action.center,
            zoom: action.zoom,
            duration: 650,
            essential: true,
          });
        },
      });
    if (!overlayRef.current) {
      overlayRef.current = new MapboxOverlay({ interleaved: true, layers: [buildLayer()] });
      map.addControl(overlayRef.current);
    } else overlayRef.current.setProps({ layers: [buildLayer()] });
    containerRef.current?.setAttribute("data-prediction-layer", "ready");
    containerRef.current?.setAttribute("data-h3-resolution", String(data.resolution));
    containerRef.current?.setAttribute("data-snapshot-timestamp", data.timestamp);
  }, [data, selectedH3, layers, mode, isFetching, styleRevision, basemapMode]);

  return (
    <div className="map-stage">
      <div ref={containerRef} className="map-canvas" data-map-mode={mode} />
      <div ref={tooltipRef} className="map-tooltip" aria-hidden="true" />
      {satelliteError && (
        <div className="basemap-alert" role="status">
          Satellite imagery unavailable. Using standard basemap.
        </div>
      )}
    </div>
  );
}
