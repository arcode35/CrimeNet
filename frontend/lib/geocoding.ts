import {
  config,
  geocoding,
  type GeocodingFeature,
  type GeocodingOptions,
  type GeocodingPlaceType,
  type GeocodingSearchResult,
  type ReverseGeocodingOptions,
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

export type CellLocation = {
  label: string;
  primaryLabel: string;
  secondaryLabel?: string;
  longitude: number;
  latitude: number;
  source: "reverse-geocoder" | "coordinates";
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

export const CELL_LOCATION_TYPES: GeocodingPlaceType[] = [
  "municipality",
  "place",
  "locality",
  "region",
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
  const position =
    geometryPosition ?? (finitePosition(feature.center) ? feature.center : undefined);
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

type ReverseGeocoder = (
  position: [longitude: number, latitude: number],
  options?: ReverseGeocodingOptions,
) => Promise<GeocodingSearchResult>;

function coordinateLabel(latitude: number, longitude: number) {
  const latitudeDirection = latitude >= 0 ? "N" : "S";
  const longitudeDirection = longitude >= 0 ? "E" : "W";
  return `${Math.abs(latitude).toFixed(4)}° ${latitudeDirection}, ${Math.abs(longitude).toFixed(4)}° ${longitudeDirection}`;
}

export function coordinateCellLocation(longitude: number, latitude: number): CellLocation {
  const label = coordinateLabel(latitude, longitude);
  return {
    label,
    primaryLabel: label,
    longitude,
    latitude,
    source: "coordinates",
  };
}

function hierarchyEntry(
  feature: GeocodingFeature,
  type: "municipality" | "place" | "locality" | "region",
) {
  if (feature.place_type.includes(type)) return feature;
  return feature.context?.find((entry) => entry.id.startsWith(`${type}.`));
}

export function adaptReverseGeocodingResult(
  response: GeocodingSearchResult,
  longitude: number,
  latitude: number,
): CellLocation {
  for (const feature of response.features) {
    const municipality = hierarchyEntry(feature, "municipality");
    const place = hierarchyEntry(feature, "place");
    const locality = hierarchyEntry(feature, "locality");
    const region = hierarchyEntry(feature, "region");
    const primary = municipality?.text ?? place?.text ?? locality?.text;
    if (!primary) continue;

    const secondary = region?.text && region.text !== primary ? region.text : undefined;
    return {
      label: secondary ? `${primary}, ${secondary}` : primary,
      primaryLabel: primary,
      ...(secondary ? { secondaryLabel: secondary } : {}),
      longitude,
      latitude,
      source: "reverse-geocoder",
    };
  }
  return coordinateCellLocation(longitude, latitude);
}

export async function reverseCellLocation(
  longitude: number,
  latitude: number,
  reverse: ReverseGeocoder = geocoding.reverse,
): Promise<CellLocation> {
  if (!mapTilerApiKey && reverse === geocoding.reverse) {
    return coordinateCellLocation(longitude, latitude);
  }

  try {
    const response = await reverse([longitude, latitude], {
      language: ["en"],
      limit: 1,
      types: CELL_LOCATION_TYPES,
    });
    return adaptReverseGeocodingResult(response, longitude, latitude);
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    return coordinateCellLocation(longitude, latitude);
  }
}

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
