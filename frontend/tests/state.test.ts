import { beforeEach, describe, expect, it } from "vitest";
import { predictionQueryKey } from "@/lib/api";
import { useExplorerStore } from "@/stores/explorer-store";

describe("city-scoped state", () => {
  beforeEach(() =>
    useExplorerStore.setState({
      cityId: "chicago",
      selectedH3: null,
      timestamp: "2024-08-21T22:00:00.000Z",
      basemapMode: "dark",
    }),
  );

  it("defaults to the dark basemap", () => {
    expect(useExplorerStore.getState().basemapMode).toBe("dark");
  });

  it("changes basemap without changing analytical state", () => {
    const timestamp = useExplorerStore.getState().timestamp;
    useExplorerStore.getState().selectCell("892664c1a8fffff");
    useExplorerStore.getState().setBasemapMode("satellite");
    expect(useExplorerStore.getState()).toMatchObject({
      basemapMode: "satellite",
      timestamp,
      selectedH3: "892664c1a8fffff",
    });
  });

  it("clears city-specific selections during city changes", () => {
    useExplorerStore.getState().selectCell("892664c1a8fffff");
    useExplorerStore.getState().setCity("seattle");
    expect(useExplorerStore.getState().selectedH3).toBeNull();
  });

  it("isolates prediction cache entries by city", () => {
    expect(predictionQueryKey("chicago", "2024-01-01T00:00:00.000Z", 1)).not.toEqual(
      predictionQueryKey("seattle", "2024-01-01T00:00:00.000Z", 1),
    );
  });
});
