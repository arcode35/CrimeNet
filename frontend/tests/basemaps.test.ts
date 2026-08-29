import { describe, expect, it, vi } from "vitest";
import type { Map as MapLibreMap, StyleSpecification } from "maplibre-gl";
import {
  applyBasemapMode,
  ensureSatelliteLayer,
  SATELLITE_LAYER_ID,
  SATELLITE_SOURCE_ID,
} from "@/lib/map/basemaps";

function createMap() {
  const layers: StyleSpecification["layers"] = [
    { id: "Background", type: "background", paint: { "background-color": "#000" } },
    { id: "Water", type: "fill", source: "vector", "source-layer": "water" },
    { id: "Country border", type: "line", source: "vector", "source-layer": "boundary" },
    { id: "City labels", type: "symbol", source: "vector", "source-layer": "place" },
  ];
  const sources = new Map<string, unknown>();
  const visibility = new Map<string, string>();
  const map = {
    getStyle: () => ({ version: 8, sources: {}, layers }),
    getSource: (id: string) => sources.get(id),
    addSource: vi.fn((id: string, source: unknown) => sources.set(id, source)),
    getLayer: (id: string) => layers.find((layer) => layer.id === id),
    addLayer: vi.fn((layer: StyleSpecification["layers"][number], beforeId?: string) => {
      const index = beforeId ? layers.findIndex((candidate) => candidate.id === beforeId) : -1;
      if (index >= 0) layers.splice(index, 0, layer);
      else layers.push(layer);
    }),
    setLayoutProperty: vi.fn((id: string, _property: string, value: string) => {
      visibility.set(id, value);
    }),
  };
  return { map: map as unknown as MapLibreMap, mock: map, visibility, layers };
}

describe("satellite basemap layers", () => {
  it("does nothing when MapTiler configuration is missing", () => {
    const { map, mock } = createMap();
    expect(ensureSatelliteLayer(map)).toBe(false);
    expect(mock.addSource).not.toHaveBeenCalled();
    expect(mock.addLayer).not.toHaveBeenCalled();
  });

  it("installs one hidden raster below reference layers", () => {
    const { map, mock, layers } = createMap();
    expect(ensureSatelliteLayer(map, "test-key")).toBe(true);
    expect(ensureSatelliteLayer(map, "test-key")).toBe(true);
    expect(mock.addSource).toHaveBeenCalledTimes(1);
    expect(mock.addSource).toHaveBeenCalledWith(
      SATELLITE_SOURCE_ID,
      expect.objectContaining({ type: "raster", tileSize: 512 }),
    );
    expect(mock.addLayer).toHaveBeenCalledTimes(1);
    expect(layers.findIndex((layer) => layer.id === SATELLITE_LAYER_ID)).toBeLessThan(
      layers.findIndex((layer) => layer.id === "Country border"),
    );
  });

  it("switches only base visibility and keeps references visible", () => {
    const { map, visibility } = createMap();
    ensureSatelliteLayer(map, "test-key");
    applyBasemapMode(map, "satellite");
    expect(visibility.get("Background")).toBe("none");
    expect(visibility.get("Water")).toBe("none");
    expect(visibility.get("Country border")).toBe("visible");
    expect(visibility.get("City labels")).toBe("visible");
    expect(visibility.get(SATELLITE_LAYER_ID)).toBe("visible");

    applyBasemapMode(map, "dark");
    expect(visibility.get("Background")).toBe("visible");
    expect(visibility.get("Water")).toBe("visible");
    expect(visibility.get(SATELLITE_LAYER_ID)).toBe("none");
  });
});
