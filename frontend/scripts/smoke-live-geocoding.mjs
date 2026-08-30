import { readFileSync } from "node:fs";
import { config, geocoding } from "@maptiler/client";

function localPublicKey() {
  if (process.env.NEXT_PUBLIC_MAPTILER_KEY?.trim()) {
    return process.env.NEXT_PUBLIC_MAPTILER_KEY.trim();
  }
  const env = readFileSync(new URL("../.env.local", import.meta.url), "utf8");
  const entry = env
    .split(/\r?\n/)
    .find((line) => line.trim().startsWith("NEXT_PUBLIC_MAPTILER_KEY="));
  return entry?.split("=").slice(1).join("=").trim() ?? "";
}

function distanceKm([longitudeA, latitudeA], [longitudeB, latitudeB]) {
  const radians = (degrees) => (degrees * Math.PI) / 180;
  const latitudeDelta = radians(latitudeB - latitudeA);
  const longitudeDelta = radians(longitudeB - longitudeA);
  const a =
    Math.sin(latitudeDelta / 2) ** 2 +
    Math.cos(radians(latitudeA)) *
      Math.cos(radians(latitudeB)) *
      Math.sin(longitudeDelta / 2) ** 2;
  return 6_371 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

const apiKey = localPublicKey();
if (!apiKey) throw new Error("NEXT_PUBLIC_MAPTILER_KEY is not configured.");
config.apiKey = apiKey;

const houston = [-95.3698, 29.7604];
const chicago = [-87.6298, 41.8781];
const queries = [
  "1600 Pennsylvania Ave NW, Washington, DC",
  "1234 Westheimer Rd, Houston, TX",
  "350 5th Ave, New York, NY",
  "Chicago IL",
  "Times Square",
  "JFK Airport",
  "77005",
  "Washington Ave",
];
const results = [];
for (const query of queries) {
  const response = await geocoding.forward(query, {
    autocomplete: true,
    fuzzyMatch: true,
    country: ["us"],
    language: ["en"],
    limit: 6,
    proximity: houston,
    types: [
      "address",
      "road",
      "place",
      "municipality",
      "locality",
      "neighbourhood",
      "poi",
      "postal_code",
    ],
  });
  const first = response.features[0];
  if (!first || !Array.isArray(first.center) || first.center.length < 2) {
    throw new Error(`No usable MapTiler result for ${query}`);
  }
  results.push({
    query,
    first: first.place_name,
    type: first.place_type[0],
    center: first.center.slice(0, 2),
    resultCount: response.features.length,
  });
}

const washingtonAvenue = results.find((result) => result.query === "Washington Ave");
if (!washingtonAvenue || distanceKm(washingtonAvenue.center, houston) > 160) {
  throw new Error("Houston proximity did not produce a nearby Washington Ave result.");
}
const chicagoResult = results.find((result) => result.query === "Chicago IL");
if (!chicagoResult || distanceKm(chicagoResult.center, chicago) > 100) {
  throw new Error("Nationwide Chicago search was incorrectly constrained by Houston proximity.");
}

console.log(
  JSON.stringify(
    {
      status: "ok",
      provider: "MapTiler",
      proximity: "Houston",
      queries: results,
      houstonBiasKm: Number(distanceKm(washingtonAvenue.center, houston).toFixed(1)),
      chicagoFromHoustonBiasKm: Number(distanceKm(chicagoResult.center, chicago).toFixed(1)),
    },
    null,
    2,
  ),
);
