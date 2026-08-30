import {
  config,
  geocoding,
  type GeocodingFeature,
  type GeocodingOptions,
  type GeocodingPlaceType,
  type GeocodingSearchResult,
} from "@maptiler/client";

export const MIN_GEOCODING_QUERY_LENGTH = 3;
export const GEOCODING_RESULT_LIMIT = 6;

export type GeocodingResult = {
  id: string;
  label: string;
  primaryLabel: string;
  secondaryLabel?: string;
  longitude: number;
  latitude: number;
  bbox?: [west: number, south: number, east: number, north: number];
  type?: GeocodingPlaceType | "coordinate";
};

export type GeocodingErrorKind = "missing-key" | "provider";

export class GeocodingError extends Error {
  constructor(
    message: string,
    readonly kind: GeocodingErrorKind,
  ) {
    super(message);
    this.name = "GeocodingError";
  }
}

export const GEOCODING_TYPES: GeocodingPlaceType[] = [
  "address",
  "road",
  "place",
  "municipality",
  "locality",
  "neighbourhood",
  "poi",
  "postal_code",
];

const mapTilerApiKey = process.env.NEXT_PUBLIC_MAPTILER_KEY?.trim() ?? "";
config.apiKey = mapTilerApiKey;

export function normalizeGeocodingQuery(query: string) {
  return query.trim().replace(/\s+/g, " ");
}

export function buildGeocodingOptions(
  proximity?: [longitude: number, latitude: number],
): GeocodingOptions {
  return {
    autocomplete: true,
    fuzzyMatch: true,
    country: ["us"],
    language: ["en"],
    limit: GEOCODING_RESULT_LIMIT,
    types: GEOCODING_TYPES,
    ...(proximity ? { proximity } : {}),
  };
}

function finitePosition(value: unknown): value is [number, number] {
  return (
    Array.isArray(value) &&
    value.length >= 2 &&
    Number.isFinite(value[0]) &&
    Number.isFinite(value[1])
  );
}

function adaptBbox(value: unknown): GeocodingResult["bbox"] {
  if (
    !Array.isArray(value) ||
    value.length < 4 ||
    !value.slice(0, 4).every((part) => Number.isFinite(part))
  ) {
    return undefined;
  }
  const [west, south, east, north] = value as number[];
  return west < east && south < north ? [west, south, east, north] : undefined;
}

export function adaptMapTilerFeature(feature: GeocodingFeature): GeocodingResult {
  const geometryPosition =
    feature.geometry.type === "Point" && finitePosition(feature.geometry.coordinates)
      ? feature.geometry.coordinates
      : undefined;
  const position = geometryPosition ?? (finitePosition(feature.center) ? feature.center : undefined);
  if (!position) throw new GeocodingError("MapTiler returned an invalid location.", "provider");

  const [longitude, latitude] = position;
  const primaryLabel = feature.address ? `${feature.address} ${feature.text}` : feature.text;
  const label = feature.place_name?.trim() || primaryLabel;
  const prefix = label.toLocaleLowerCase().startsWith(primaryLabel.toLocaleLowerCase())
    ? label.slice(primaryLabel.length).replace(/^,\s*/, "")
    : label;

  return {
    id: feature.id,
    label,
    primaryLabel,
    ...(prefix && prefix !== primaryLabel ? { secondaryLabel: prefix } : {}),
    longitude,
    latitude,
    ...(adaptBbox(feature.bbox) ? { bbox: adaptBbox(feature.bbox) } : {}),
    ...(feature.place_type[0] ? { type: feature.place_type[0] } : {}),
  };
}

export function parseCoordinateQuery(query: string): GeocodingResult | null {
  const match = normalizeGeocodingQuery(query).match(
    /^([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*,\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))$/,
  );
  if (!match) return null;
  const latitude = Number(match[1]);
  const longitude = Number(match[2]);
  if (latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180) return null;
  const label = `${latitude.toFixed(5)}, ${longitude.toFixed(5)}`;
  return {
    id: `coordinate:${latitude},${longitude}`,
    label,
    primaryLabel: label,
    secondaryLabel: "Latitude, longitude",
    longitude,
    latitude,
    type: "coordinate",
  };
}

type ForwardGeocoder = (
  query: string,
  options?: GeocodingOptions,
) => Promise<GeocodingSearchResult>;

export async function searchLocations(
  query: string,
  proximity?: [longitude: number, latitude: number],
  forward: ForwardGeocoder = geocoding.forward,
): Promise<GeocodingResult[]> {
  const normalized = normalizeGeocodingQuery(query);
  if (normalized.length < MIN_GEOCODING_QUERY_LENGTH) return [];
  if (!mapTilerApiKey && forward === geocoding.forward) {
    throw new GeocodingError("MapTiler API key is not configured.", "missing-key");
  }

  try {
    const response = await forward(normalized, buildGeocodingOptions(proximity));
    return response.features.flatMap((feature) => {
      try {
        return [adaptMapTilerFeature(feature)];
      } catch {
        return [];
      }
    });
  } catch (error) {
    if (error instanceof GeocodingError) throw error;
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new GeocodingError("Location search unavailable.", "provider");
  }
}
