import { describe, expect, it } from "vitest";
import { parseExplorerUrl, serializeExplorerUrl } from "@/lib/url-state";

describe("shareable explorer state", () => {
  it("restores valid city, time, horizon, and H3 selection", () => {
    const url = serializeExplorerUrl({
      cityId: "baltimore",
      timestamp: "2024-08-21T22:00:00.000Z",
      horizonHours: 6,
      selectedH3: "892664c1a8fffff",
      basemapMode: "satellite",
    });
    expect(parseExplorerUrl(url)).toEqual({
      cityId: "baltimore",
      timestamp: "2024-08-21T22:00:00.000Z",
      horizonHours: 6,
      selectedH3: "892664c1a8fffff",
      basemapMode: "satellite",
    });
  });

  it("ignores invalid analytical state", () => {
    expect(parseExplorerUrl("?city=atlantis&time=never&horizon=72&basemap=terrain")).toEqual({});
  });
});
