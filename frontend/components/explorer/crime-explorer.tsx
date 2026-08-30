"use client";

import dynamic from "next/dynamic";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useState } from "react";
import {
  CrimeNetApiError,
  getLiveViewport,
  getPredictions,
  getServiceHealth,
  isFixtureMode,
  isLiveMode,
  liveViewportQueryKey,
  predictionQueryKey,
  roundViewportBounds,
  serviceHealthQueryKey,
  type ViewportBounds,
} from "@/lib/api";
import { getCity } from "@/lib/domain";
import { shouldClearSelectionForResolution } from "@/lib/map/lod";
import type { MapNavigation } from "@/lib/map/navigation";
import { parseExplorerUrl, serializeExplorerUrl } from "@/lib/url-state";
import { useExplorerStore } from "@/stores/explorer-store";
import { CommandPalette } from "./command-palette";
import { ControlPanel } from "./control-panel";
import { Inspector } from "./inspector";
import { MapLegend } from "./map-legend";
import { Timeline } from "./timeline";
import { TopBar } from "./top-bar";

const GeospatialMap = dynamic(
  () => import("./geospatial-map").then((module) => module.GeospatialMap),
  {
    ssr: false,
    loading: () => (
      <div className="map-loading" aria-label="Loading geospatial renderer">
        <span />
        <span />
        <span />
        <p>INITIALIZING SPATIAL RENDERER</p>
      </div>
    ),
  },
);

export function CrimeExplorer() {
  const cityId = useExplorerStore((state) => state.cityId);
  const timestamp = useExplorerStore((state) => state.timestamp);
  const horizonHours = useExplorerStore((state) => state.horizonHours);
  const selectedH3 = useExplorerStore((state) => state.selectedH3);
  const basemapMode = useExplorerStore((state) => state.basemapMode);
  const [viewport, setViewport] = useState<ViewportBounds | null>(null);
  const [mapNavigation, setMapNavigation] = useState<MapNavigation | null>(null);
  const queryClient = useQueryClient();
  const city = getCity(cityId);

  const healthQuery = useQuery({
    queryKey: serviceHealthQueryKey,
    queryFn: ({ signal }) => getServiceHealth(signal),
    enabled: isLiveMode,
    refetchInterval: 60_000,
    retry: 2,
  });
  const snapshotId = healthQuery.data?.snapshot_id;
  const liveReady = Boolean(isLiveMode && snapshotId && viewport);

  const predictionQuery = useQuery({
    queryKey: isLiveMode
      ? liveReady
        ? liveViewportQueryKey(snapshotId!, viewport!)
        : (["live-intensity", "waiting"] as const)
      : predictionQueryKey(cityId, timestamp, horizonHours),
    queryFn: ({ signal }) =>
      isLiveMode
        ? getLiveViewport(cityId, viewport!, signal)
        : getPredictions(cityId, timestamp, horizonHours, signal),
    enabled: !isLiveMode || liveReady,
    placeholderData: (previous) => {
      if (isLiveMode) return previous?.snapshotId === snapshotId ? previous : undefined;
      return previous?.city === cityId ? previous : undefined;
    },
    retry: (failures, error) =>
      error instanceof CrimeNetApiError &&
      (error.kind === "viewport-too-large" || error.kind === "not-found")
        ? false
        : failures < 2,
  });

  const handleViewportChange = useCallback((next: ViewportBounds) => {
    const rounded = roundViewportBounds(next);
    setViewport((current) =>
      current &&
      current.west === rounded.west &&
      current.south === rounded.south &&
      current.east === rounded.east &&
      current.north === rounded.north &&
      current.zoom === rounded.zoom
        ? current
        : rounded,
    );
  }, []);

  useEffect(() => {
    if (!isLiveMode || !healthQuery.data) return;
    const store = useExplorerStore.getState();
    const liveTimestamp = new Date(healthQuery.data.valid_utc_hour).toISOString();
    if (store.timestamp !== liveTimestamp) store.setTimestamp(liveTimestamp);
    if (store.horizonHours !== 1) store.setHorizon(1);
    if (store.playing) store.setPlaying(false);
  }, [healthQuery.data]);

  useEffect(() => {
    if (
      isLiveMode &&
      predictionQuery.data?.snapshotId &&
      snapshotId &&
      predictionQuery.data.snapshotId !== snapshotId
    ) {
      void queryClient.invalidateQueries({ queryKey: serviceHealthQueryKey });
    }
  }, [predictionQuery.data?.snapshotId, queryClient, snapshotId]);

  useEffect(() => {
    if (shouldClearSelectionForResolution(predictionQuery.data?.resolution, selectedH3)) {
      useExplorerStore.getState().selectCell(null);
    }
  }, [predictionQuery.data?.resolution, selectedH3]);

  useEffect(() => {
    const restored = parseExplorerUrl(window.location.search);
    const store = useExplorerStore.getState();
    if (restored.cityId) store.setCity(restored.cityId);
    if (!isLiveMode && restored.timestamp) store.setTimestamp(restored.timestamp);
    if (!isLiveMode && restored.horizonHours) store.setHorizon(restored.horizonHours);
    if (restored.selectedH3) store.selectCell(restored.selectedH3);
    if (restored.basemapMode) store.setBasemapMode(restored.basemapMode);
  }, []);

  useEffect(() => {
    window.history.replaceState(
      null,
      "",
      serializeExplorerUrl({ cityId, timestamp, horizonHours, selectedH3, basemapMode }),
    );
  }, [cityId, timestamp, horizonHours, selectedH3, basemapMode]);

  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (target.matches("input, textarea, select, [contenteditable=true]")) return;
      const store = useExplorerStore.getState();
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        store.setCommandOpen(true);
      } else if (!isLiveMode && event.key === "ArrowLeft") store.stepTime(-1);
      else if (!isLiveMode && event.key === "ArrowRight") store.stepTime(1);
      else if (!isLiveMode && event.code === "Space") {
        event.preventDefault();
        store.setPlaying(!store.playing);
      } else if (event.key === "Escape") {
        store.selectCell(null);
        store.setCommandOpen(false);
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, []);

  useEffect(() => {
    if (isLiveMode) return;
    const nextTime = new Date(new Date(timestamp).getTime() + 3_600_000).toISOString();
    void queryClient.prefetchQuery({
      queryKey: predictionQueryKey(cityId, nextTime, horizonHours),
      queryFn: ({ signal }) => getPredictions(cityId, nextTime, horizonHours, signal),
    });
  }, [cityId, timestamp, horizonHours, queryClient]);

  const viewportError =
    predictionQuery.error instanceof CrimeNetApiError ? predictionQuery.error : null;
  const zoomState =
    viewportError?.kind === "viewport-too-large"
      ? "Zoom in to load live H3 predictions"
      : viewportError?.kind === "not-found"
        ? "No live prediction coverage in this view"
        : null;
  const data = zoomState ? undefined : predictionQuery.data;
  const genericPredictionError = predictionQuery.isError && !zoomState;

  return (
    <main className="app-shell">
      <GeospatialMap
        city={city}
        data={data}
        isFetching={predictionQuery.isFetching}
        onViewportChange={handleViewportChange}
        onMapReady={setMapNavigation}
      />
      <div className="map-vignette" aria-hidden="true" />
      <TopBar
        fixtureMode={isFixtureMode}
        snapshotId={snapshotId}
        serviceDegraded={
          isLiveMode &&
          (healthQuery.isError ||
            Boolean(
              healthQuery.data &&
                (healthQuery.data.status !== "ok" ||
                  healthQuery.data.mark_model.status !== "ready"),
            ))
        }
      />
      <ControlPanel />
      <MapLegend data={data} error={genericPredictionError ? predictionQuery.error : null} />
      <Inspector data={data} snapshotId={snapshotId} />
      <Timeline data={data} isFetching={predictionQuery.isFetching} liveMode={isLiveMode} />
      <CommandPalette mapNavigation={mapNavigation} />
      {zoomState && <div className="viewport-notice">{zoomState}</div>}
      {genericPredictionError && (
        <div className="service-alert" role="alert">
          <strong>Inference temporarily unavailable</strong>
          <span>Map exploration remains available.</span>
          <button onClick={() => predictionQuery.refetch()}>Retry</button>
        </div>
      )}
      {isLiveMode && healthQuery.isError && (
        <div className="service-alert health-alert" role="alert">
          <strong>Live snapshot status unavailable</strong>
          <span>Check that the local serving API is running.</span>
          <button onClick={() => healthQuery.refetch()}>Retry</button>
        </div>
      )}
    </main>
  );
}
