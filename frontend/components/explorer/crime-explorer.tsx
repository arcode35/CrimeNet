"use client";

import dynamic from "next/dynamic";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  CrimeNetApiError,
  fetchIntensityTimeline,
  getLiveViewport,
  getPredictions,
  getServiceHealth,
  isFixtureMode,
  isLiveMode,
  intensityTimelineQueryKey,
  liveViewportQueryKey,
  predictionQueryKey,
  roundViewportBounds,
  serviceHealthQueryKey,
  type ViewportBounds,
} from "@/lib/api";
import { getCity } from "@/lib/domain";
import { resolveForecastSelection } from "@/lib/forecast";
import { isAbortError, LatestRequest } from "@/lib/interaction";
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
  const [preferredForecastHour, setPreferredForecastHour] = useState<string | null>(null);
  const timelineRequest = useRef(new LatestRequest());
  const viewportRequest = useRef(new LatestRequest());
  const queryClient = useQueryClient();
  const city = getCity(cityId);

  const healthQuery = useQuery({
    queryKey: serviceHealthQueryKey,
    queryFn: ({ signal }) => getServiceHealth(signal),
    enabled: isLiveMode,
    refetchInterval: 60_000,
    retry: 2,
  });
  const timelineQuery = useQuery({
    queryKey: intensityTimelineQueryKey,
    queryFn: ({ signal }) =>
      timelineRequest.current.run((latestSignal) => fetchIntensityTimeline(latestSignal), signal),
    enabled: isLiveMode,
    refetchInterval: 60_000,
    retry: 2,
  });
  const fallbackSnapshots = useMemo(
    () =>
      healthQuery.data
        ? [
            {
              snapshot_id: healthQuery.data.snapshot_id,
              valid_utc_hour: healthQuery.data.valid_utc_hour,
              horizon_hours: 0,
              kind: "live" as const,
            },
          ]
        : [],
    [healthQuery.data],
  );
  const snapshots = timelineQuery.data?.snapshots ?? fallbackSnapshots;
  const resolvedForecast = useMemo(
    () => resolveForecastSelection(snapshots, preferredForecastHour),
    [preferredForecastHour, snapshots],
  );
  const selectedSnapshot = resolvedForecast?.snapshot;
  const selectedForecastIndex = resolvedForecast?.index ?? 0;
  const asOfUtcHour =
    timelineQuery.data?.as_of_utc_hour ?? healthQuery.data?.valid_utc_hour ?? null;
  const liveReady = Boolean(isLiveMode && asOfUtcHour && selectedSnapshot && viewport);

  const predictionQuery = useQuery({
    queryKey: isLiveMode
      ? liveReady
        ? liveViewportQueryKey(asOfUtcHour!, selectedSnapshot!.valid_utc_hour, viewport!)
        : (["intensity-viewport", "waiting"] as const)
      : predictionQueryKey(cityId, timestamp, horizonHours),
    queryFn: ({ signal }) =>
      viewportRequest.current.run(
        (latestSignal) =>
          isLiveMode
            ? getLiveViewport(cityId, viewport!, selectedSnapshot!.valid_utc_hour, latestSignal)
            : getPredictions(cityId, timestamp, horizonHours, latestSignal),
        signal,
      ),
    enabled: !isLiveMode || liveReady,
    placeholderData: (previous) => {
      if (isLiveMode) return previous;
      return previous?.city === cityId ? previous : undefined;
    },
    retry: (failures, error) =>
      isAbortError(error) ||
      (error instanceof CrimeNetApiError &&
        (error.kind === "viewport-too-large" ||
          error.kind === "not-found" ||
          error.kind === "busy"))
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
      current.north === rounded.north
        ? current
        : rounded,
    );
  }, []);

  useEffect(
    () => () => {
      timelineRequest.current.cancel();
      viewportRequest.current.cancel();
    },
    [],
  );

  const handleForecastIndexChange = useCallback(
    (index: number) => {
      const snapshot = snapshots[index];
      if (!snapshot) return;
      setPreferredForecastHour(snapshot.kind === "live" ? null : snapshot.valid_utc_hour);
      useExplorerStore.getState().hoverCell(null);
    },
    [snapshots],
  );

  useEffect(() => {
    if (!isLiveMode || !selectedSnapshot) return;
    const store = useExplorerStore.getState();
    const selectedTimestamp = new Date(selectedSnapshot.valid_utc_hour).toISOString();
    if (store.timestamp !== selectedTimestamp) store.setTimestamp(selectedTimestamp);
    if (store.horizonHours !== 1) store.setHorizon(1);
  }, [selectedSnapshot]);

  useEffect(() => {
    if (
      !isLiveMode ||
      selectedSnapshot?.kind !== "live" ||
      predictionQuery.isPlaceholderData ||
      !predictionQuery.data?.snapshotId ||
      predictionQuery.data.snapshotId === selectedSnapshot.snapshot_id
    ) {
      return;
    }
    void queryClient.invalidateQueries({ queryKey: intensityTimelineQueryKey });
    void queryClient.invalidateQueries({ queryKey: serviceHealthQueryKey });
  }, [
    predictionQuery.data?.snapshotId,
    predictionQuery.isPlaceholderData,
    queryClient,
    selectedSnapshot,
  ]);

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
  const forecastHourUnavailable = Boolean(
    viewportError?.kind === "not-found" && selectedSnapshot?.kind === "forecast",
  );
  const zoomState =
    viewportError?.kind === "viewport-too-large"
      ? "Zoom in to load live H3 predictions"
      : viewportError?.kind === "not-found" && !forecastHourUnavailable
        ? "No live prediction coverage in this view"
        : null;
  const data = zoomState ? undefined : predictionQuery.data;
  const genericPredictionError = predictionQuery.isError && !zoomState && !forecastHourUnavailable;

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
        snapshotId={selectedSnapshot?.snapshot_id}
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
      <Inspector
        data={data}
        selectedSnapshot={selectedSnapshot}
        asOfUtcHour={asOfUtcHour ?? undefined}
      />
      <Timeline
        data={data}
        isFetching={predictionQuery.isFetching}
        liveMode={isLiveMode}
        snapshots={snapshots}
        selectedIndex={selectedForecastIndex}
        onSelectedIndexChange={handleForecastIndexChange}
        forecastUnavailable={timelineQuery.isError}
        forecastLoading={timelineQuery.isPending}
      />
      <CommandPalette mapNavigation={mapNavigation} />
      {zoomState && <div className="viewport-notice">{zoomState}</div>}
      {forecastHourUnavailable && (
        <div className="viewport-notice forecast-notice">
          Forecast hour unavailable. Showing the previous surface.
        </div>
      )}
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
          <span>The public CrimeSense API could not be reached.</span>
          <button onClick={() => healthQuery.refetch()}>Retry</button>
        </div>
      )}
    </main>
  );
}
