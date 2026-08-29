"use client";

import dynamic from "next/dynamic";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { getPredictions, isFixtureMode, predictionQueryKey } from "@/lib/api";
import { getCity } from "@/lib/domain";
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
  const queryClient = useQueryClient();
  const city = getCity(cityId);

  const predictionQuery = useQuery({
    queryKey: predictionQueryKey(cityId, timestamp, horizonHours),
    queryFn: ({ signal }) => getPredictions(cityId, timestamp, horizonHours, signal),
    placeholderData: (previous, previousQuery) =>
      previousQuery?.queryKey[1] === cityId ? previous : undefined,
  });

  useEffect(() => {
    const restored = parseExplorerUrl(window.location.search);
    const store = useExplorerStore.getState();
    if (restored.cityId) store.setCity(restored.cityId);
    if (restored.timestamp) store.setTimestamp(restored.timestamp);
    if (restored.horizonHours) store.setHorizon(restored.horizonHours);
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
      } else if (event.key === "ArrowLeft") store.stepTime(-1);
      else if (event.key === "ArrowRight") store.stepTime(1);
      else if (event.code === "Space") {
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
    const nextTime = new Date(new Date(timestamp).getTime() + 3_600_000).toISOString();
    void queryClient.prefetchQuery({
      queryKey: predictionQueryKey(cityId, nextTime, horizonHours),
      queryFn: ({ signal }) => getPredictions(cityId, nextTime, horizonHours, signal),
    });
  }, [cityId, timestamp, horizonHours, queryClient]);

  return (
    <main className="app-shell">
      <GeospatialMap
        city={city}
        data={predictionQuery.data}
        isFetching={predictionQuery.isFetching}
      />
      <div className="map-vignette" aria-hidden="true" />
      <TopBar fixtureMode={isFixtureMode} />
      <ControlPanel />
      <MapLegend data={predictionQuery.data} error={predictionQuery.error} />
      <Inspector data={predictionQuery.data} />
      <Timeline data={predictionQuery.data} isFetching={predictionQuery.isFetching} />
      <CommandPalette />
      {predictionQuery.isError && (
        <div className="service-alert" role="alert">
          <strong>Inference temporarily unavailable</strong>
          <span>Map exploration remains available.</span>
          <button onClick={() => predictionQuery.refetch()}>Retry</button>
        </div>
      )}
    </main>
  );
}
