import type { GeocodingFeature, GeocodingSearchResult } from "@maptiler/client";
import { describe, expect, it, vi } from "vitest";
import {
  adaptMapTilerFeature,
  buildGeocodingOptions,
  parseCoordinateQuery,
  searchLocations,
} from "@/lib/geocoding";
import { navigateToGeocodingResult, type MapNavigation } from "@/lib/map/navigation";

function feature(overrides: Partial<GeocodingFeature> = {}): GeocodingFeature {
  return {
    type: "Feature",
    id: "address.123",
    text: "Westheimer Road",
    address: "1234",
    place_name: "1234 Westheimer Road, Houston, Texas 77006, United States",
    place_type: ["address"],
    place_type_name: ["Address"],
    center: [-95.3936, 29.7411],
    bbox: [-95.394, 29.7408, -95.3932, 29.7414],
    geometry: { type: "Point", coordinates: [-95.3936, 29.7411] },
    properties: { ref: "test", country_code: "us" },
    relevance: 1,
    ...overrides,
  };
}

function navigation(): MapNavigation {
  return {
    getCenter: vi.fn(() => ({ longitude: -95.37, latitude: 29.76 })),
    flyToLocation: vi.fn(),
    fitToBounds: vi.fn(),
  };
}

describe("MapTiler geocoding adapter", () => {
  it("adapts GeoJSON coordinates in longitude, latitude order and formats labels", () => {
    const result = adaptMapTilerFeature(feature());
    expect(result).toMatchObject({
      longitude: -95.3936,
      latitude: 29.7411,
      primaryLabel: "1234 Westheimer Road",
      secondaryLabel: "Houston, Texas 77006, United States",
      type: "address",
    });
    expect(result.bbox).toEqual([-95.394, 29.7408, -95.3932, 29.7414]);
  });

  it("uses US autocomplete, fuzzy matching, navigation types, English, and proximity", async () => {
    const response: GeocodingSearchResult = {
      type: "FeatureCollection",
      features: [feature()],
      query: ["westheimer"],
      attribution: "MapTiler",
    };
    const forward = vi.fn(async () => response);
    await searchLocations("  Westheimer   Rd Houston  ", [-95.37, 29.76], forward);
    expect(forward).toHaveBeenCalledWith(
      "Westheimer Rd Houston",
      expect.objectContaining({
        autocomplete: true,
        fuzzyMatch: true,
        country: ["us"],
        language: ["en"],
        limit: 6,
        proximity: [-95.37, 29.76],
        types: expect.arrayContaining(["address", "road", "place", "poi", "postal_code"]),
      }),
    );
    expect(buildGeocodingOptions()).not.toHaveProperty("bbox");
  });

  it("strictly parses latitude, longitude input without swapping values", () => {
    expect(parseCoordinateQuery("29.7604, -95.3698")).toMatchObject({
      latitude: 29.7604,
      longitude: -95.3698,
      type: "coordinate",
    });
    expect(parseCoordinateQuery("1234 Westheimer Rd, Houston")).toBeNull();
    expect(parseCoordinateQuery("95, -200")).toBeNull();
  });
});

describe("geocoding map navigation", () => {
  it("uses fitBounds for a useful provider bbox", () => {
    const map = navigation();
    expect(navigateToGeocodingResult(map, adaptMapTilerFeature(feature()))).toBe("fitBounds");
    expect(map.fitToBounds).toHaveBeenCalledWith({
      bounds: [
        [-95.394, 29.7408],
        [-95.3932, 29.7414],
      ],
      maxZoom: 16,
      duration: 1_200,
    });
    expect(map.flyToLocation).not.toHaveBeenCalled();
  });

  it("uses type-aware flyTo for a point result without a bbox", () => {
    const map = navigation();
    const result = adaptMapTilerFeature(feature({ bbox: undefined }));
    expect(navigateToGeocodingResult(map, result)).toBe("flyTo");
    expect(map.flyToLocation).toHaveBeenCalledWith({
      center: [-95.3936, 29.7411],
      zoom: 15.5,
      duration: 1_200,
    });
  });
});
