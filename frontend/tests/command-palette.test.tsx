import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CommandPalette } from "@/components/explorer/command-palette";
import * as geocoding from "@/lib/geocoding";
import type { MapNavigation } from "@/lib/map/navigation";
import { useExplorerStore } from "@/stores/explorer-store";

vi.mock("@/lib/geocoding", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/geocoding")>()),
  searchLocations: vi.fn(),
}));

const searchLocations = vi.mocked(geocoding.searchLocations);

function renderPalette(mapNavigation: MapNavigation) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return render(
    <QueryClientProvider client={client}>
      <CommandPalette mapNavigation={mapNavigation} />
    </QueryClientProvider>,
  );
}

describe("location command palette", () => {
  beforeEach(() => {
    searchLocations.mockReset();
    useExplorerStore.setState({
      commandOpen: true,
      selectedH3: "892664c1a8fffff",
      hoveredH3: "892664c1a8fffff",
    });
  });

  it("debounces rapid typing and keyboard selection clears stale H3 before flyTo", async () => {
    searchLocations.mockResolvedValue([
      {
        id: "address.1",
        label: "1234 Westheimer Road, Houston, Texas",
        primaryLabel: "1234 Westheimer Road",
        secondaryLabel: "Houston, Texas",
        longitude: -95.3936,
        latitude: 29.7411,
        type: "address",
      },
    ]);
    const mapNavigation: MapNavigation = {
      getCenter: vi.fn(() => ({ longitude: -95.37, latitude: 29.76 })),
      flyToLocation: vi.fn(),
      fitToBounds: vi.fn(),
    };
    renderPalette(mapNavigation);
    const input = screen.getByRole("combobox");
    fireEvent.change(input, { target: { value: "Hou" } });
    fireEvent.change(input, { target: { value: "Houston" } });
    expect(searchLocations).not.toHaveBeenCalled();

    await waitFor(() => expect(searchLocations).toHaveBeenCalledTimes(1));
    expect(searchLocations).toHaveBeenCalledWith("Houston", [-95.37, 29.76]);
    expect(await screen.findByText("1234 Westheimer Road")).toBeInTheDocument();

    fireEvent.keyDown(input, { key: "ArrowDown" });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(useExplorerStore.getState()).toMatchObject({
      selectedH3: null,
      hoveredH3: null,
      commandOpen: false,
    });
    expect(mapNavigation.flyToLocation).toHaveBeenCalledWith({
      center: [-95.3936, 29.7411],
      zoom: 15.5,
      duration: 1_200,
    });
  });

  it("keeps provider failures inside the search UI", async () => {
    searchLocations.mockRejectedValue(new geocoding.GeocodingError("No service", "provider"));
    const mapNavigation: MapNavigation = {
      getCenter: vi.fn(() => ({ longitude: -87.63, latitude: 41.88 })),
      flyToLocation: vi.fn(),
      fitToBounds: vi.fn(),
    };
    renderPalette(mapNavigation);
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "Chicago IL" } });
    expect(await screen.findByText("Location search unavailable", {}, { timeout: 3_000 })).toBeVisible();
    expect(screen.queryByText("Inference temporarily unavailable")).not.toBeInTheDocument();
  });
});
