import type { Map as MapLibreMap, StyleSpecification } from "maplibre-gl";
import type { BasemapMode } from "@/stores/explorer-store";

export const SATELLITE_SOURCE_ID = "crimenet-satellite";
export const SATELLITE_LAYER_ID = "crimenet-satellite-layer";

type StyleLayer = NonNullable<StyleSpecification["layers"]>[number];

type LayerGroups = {
  signature: string;
  darkBaseLayers: Array<{ id: string; visibility: "visible" | "none" }>;
  referenceLayerIds: string[];
  firstReferenceLayerId?: string;
};

const layerGroups = new WeakMap<MapLibreMap, LayerGroups>();
const REFERENCE_LAYER_PATTERN = /label|border|boundary|admin|shield/i;

function isReferenceLayer(layer: StyleLayer) {
  return layer.type === "symbol" || REFERENCE_LAYER_PATTERN.test(layer.id);
}

function inspectLayerGroups(map: MapLibreMap): LayerGroups {
  const layers = (map.getStyle().layers ?? []).filter((layer) => layer.id !== SATELLITE_LAYER_ID);
  const signature = layers.map((layer) => `${layer.id}:${layer.type}`).join("|");
  const cached = layerGroups.get(map);
  if (cached?.signature === signature) return cached;

  const referenceLayers = layers.filter(isReferenceLayer);
  const groups: LayerGroups = {
    signature,
    darkBaseLayers: layers
      .filter((layer) => !isReferenceLayer(layer))
      .map((layer) => ({
        id: layer.id,
        visibility: layer.layout?.visibility === "none" ? "none" : "visible",
      })),
    referenceLayerIds: referenceLayers.map((layer) => layer.id),
    firstReferenceLayerId: referenceLayers[0]?.id,
  };
  layerGroups.set(map, groups);
  return groups;
}

export function ensureSatelliteLayer(map: MapLibreMap, mapTilerKey?: string): boolean {
  if (!mapTilerKey) return false;
  const groups = inspectLayerGroups(map);

  if (!map.getSource(SATELLITE_SOURCE_ID)) {
    map.addSource(SATELLITE_SOURCE_ID, {
      type: "raster",
      url: `https://api.maptiler.com/maps/satellite-v4/tiles.json?key=${mapTilerKey}`,
      tileSize: 512,
      attribution:
        '<a href="https://www.maptiler.com/copyright/" target="_blank">&copy; MapTiler</a>',
    });
  }

  if (!map.getLayer(SATELLITE_LAYER_ID)) {
    map.addLayer(
      {
        id: SATELLITE_LAYER_ID,
        type: "raster",
        source: SATELLITE_SOURCE_ID,
        layout: { visibility: "none" },
        paint: {
          "raster-opacity": 0.74,
          "raster-brightness-min": 0.03,
          "raster-brightness-max": 0.68,
          "raster-contrast": 0.12,
          "raster-saturation": -0.26,
          "raster-fade-duration": 180,
        },
      },
      groups.firstReferenceLayerId,
    );
  }
  return true;
}

export function applyBasemapMode(map: MapLibreMap, mode: BasemapMode): void {
  const groups = inspectLayerGroups(map);
  const satellite = mode === "satellite" && Boolean(map.getLayer(SATELLITE_LAYER_ID));

  for (const layer of groups.darkBaseLayers) {
    if (map.getLayer(layer.id)) {
      map.setLayoutProperty(layer.id, "visibility", satellite ? "none" : layer.visibility);
    }
  }
  for (const id of groups.referenceLayerIds) {
    if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", "visible");
  }
  if (map.getLayer(SATELLITE_LAYER_ID)) {
    map.setLayoutProperty(SATELLITE_LAYER_ID, "visibility", satellite ? "visible" : "none");
  }
}
