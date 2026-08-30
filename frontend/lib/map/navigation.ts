import type { GeocodingResult } from "@/lib/geocoding";

export type MapCenter = { longitude: number; latitude: number };

export type MapNavigation = {
  getCenter: () => MapCenter;
  flyToLocation: (options: {
    center: [longitude: number, latitude: number];
    zoom: number;
    duration: number;
  }) => void;
  fitToBounds: (options: {
    bounds: [
      [west: number, south: number],
      [east: number, north: number],
    ];
    maxZoom: number;
    duration: number;
  }) => void;
};

export function targetZoomForResult(type: GeocodingResult["type"]) {
  if (type === "address" || type === "poi" || type === "coordinate") return 15.5;
  if (type === "road") return 14;
  if (type === "neighbourhood" || type === "locality") return 13;
  if (type === "postal_code") return 12.5;
  if (type === "place" || type === "municipality") return 11;
  return 12;
}

export function navigateToGeocodingResult(
  navigation: MapNavigation,
  result: GeocodingResult,
) {
  if (result.bbox) {
    const [west, south, east, north] = result.bbox;
    navigation.fitToBounds({
      bounds: [
        [west, south],
        [east, north],
      ],
      maxZoom: 16,
      duration: 1_200,
    });
    return "fitBounds" as const;
  }
  navigation.flyToLocation({
    center: [result.longitude, result.latitude],
    zoom: targetZoomForResult(result.type),
    duration: 1_200,
  });
  return "flyTo" as const;
}
