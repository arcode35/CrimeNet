"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  Box,
  Clock3,
  Command,
  CornerDownLeft,
  Layers3,
  LoaderCircle,
  MapPin,
  Search,
  X,
} from "lucide-react";
import { useEffect, useId, useState } from "react";
import { CITIES } from "@/lib/domain";
import {
  MIN_GEOCODING_QUERY_LENGTH,
  normalizeGeocodingQuery,
  parseCoordinateQuery,
  searchLocations,
  type GeocodingResult,
} from "@/lib/geocoding";
import { navigateToGeocodingResult, type MapNavigation } from "@/lib/map/navigation";
import { useExplorerStore } from "@/stores/explorer-store";

const GEOCODING_DEBOUNCE_MS = 300;

export function CommandPalette({ mapNavigation }: { mapNavigation: MapNavigation | null }) {
  const open = useExplorerStore((state) => state.commandOpen);
  const setOpen = useExplorerStore((state) => state.setCommandOpen);
  const store = useExplorerStore();
  const listboxId = useId();
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(-1);
  const [suggestionsDismissed, setSuggestionsDismissed] = useState(false);
  const normalizedQuery = normalizeGeocodingQuery(query);
  const coordinateResult = parseCoordinateQuery(normalizedQuery);
  const isLocationQuery = normalizedQuery.length >= MIN_GEOCODING_QUERY_LENGTH;
  const mapCenter = open && mapNavigation ? mapNavigation.getCenter() : undefined;
  const proximity: [number, number] | undefined = mapCenter
    ? [mapCenter.longitude, mapCenter.latitude]
    : undefined;

  useEffect(() => {
    const timer = window.setTimeout(
      () => setDebouncedQuery(normalizedQuery),
      GEOCODING_DEBOUNCE_MS,
    );
    return () => window.clearTimeout(timer);
  }, [normalizedQuery]);

  const updateQuery = (nextQuery: string) => {
    setQuery(nextQuery);
    setActiveIndex(-1);
    setSuggestionsDismissed(false);
  };

  const roundedProximity: [number, number] | undefined = proximity
    ? [Number(proximity[0].toFixed(3)), Number(proximity[1].toFixed(3))]
    : undefined;
  const geocodeQuery = useQuery({
    queryKey: ["geocode", debouncedQuery, ...(roundedProximity ?? [])],
    queryFn: () => searchLocations(debouncedQuery, proximity),
    enabled:
      open &&
      debouncedQuery.length >= MIN_GEOCODING_QUERY_LENGTH &&
      !parseCoordinateQuery(debouncedQuery),
    staleTime: 5 * 60 * 1_000,
    retry: 1,
  });
  const queryIsCurrent = debouncedQuery === normalizedQuery;
  const results = suggestionsDismissed
    ? []
    : coordinateResult
      ? [coordinateResult]
      : queryIsCurrent
        ? (geocodeQuery.data ?? [])
        : [];
  const searching =
    isLocationQuery &&
    !coordinateResult &&
    (!queryIsCurrent || (geocodeQuery.isFetching && !geocodeQuery.data));
  const cities = CITIES.filter((city) =>
    city.name.toLowerCase().includes(normalizedQuery.toLowerCase()),
  );

  const run = (action: () => void) => {
    action();
    setOpen(false);
    setQuery("");
  };

  const selectLocation = (result: GeocodingResult) => {
    if (!mapNavigation) return;
    const explorer = useExplorerStore.getState();
    explorer.selectCell(null);
    explorer.hoverCell(null);
    navigateToGeocodingResult(mapNavigation, result);
    setQuery(result.label);
    setDebouncedQuery(result.label);
    setActiveIndex(-1);
    setSuggestionsDismissed(true);
    setOpen(false);
  };

  const handleSearchKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "ArrowDown" && results.length > 0) {
      event.preventDefault();
      event.stopPropagation();
      setActiveIndex((current) => (current + 1) % results.length);
    } else if (event.key === "ArrowUp" && results.length > 0) {
      event.preventDefault();
      event.stopPropagation();
      setActiveIndex((current) => (current <= 0 ? results.length - 1 : current - 1));
    } else if (event.key === "Enter" && activeIndex >= 0 && results[activeIndex]) {
      event.preventDefault();
      event.stopPropagation();
      selectLocation(results[activeIndex]);
    } else if (event.key === "Escape" && results.length > 0) {
      event.preventDefault();
      event.stopPropagation();
      setSuggestionsDismissed(true);
      setActiveIndex(-1);
    }
  };

  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className="command-dialog" aria-describedby={undefined}>
          <Dialog.Title className="sr-only">Search locations and CrimeNet commands</Dialog.Title>
          <div className="command-search">
            <Search size={17} />
            <input
              autoFocus
              aria-label="Search addresses, places, and CrimeNet commands"
              aria-autocomplete="list"
              aria-controls={results.length > 0 ? listboxId : undefined}
              aria-expanded={results.length > 0}
              aria-activedescendant={
                activeIndex >= 0 ? `${listboxId}-option-${activeIndex}` : undefined
              }
              role="combobox"
              value={query}
              onChange={(event) => updateQuery(event.target.value)}
              onKeyDown={handleSearchKeyDown}
              placeholder="Search addresses, places, and commands…"
            />
            {searching ? <LoaderCircle className="command-loading" size={14} /> : null}
            <kbd>
              <Command size={11} />K
            </kbd>
            <Dialog.Close aria-label="Close search">
              <X size={15} />
            </Dialog.Close>
          </div>
          <div className="command-body">
            {isLocationQuery && (
              <div className="command-group location-results">
                <small>LOCATIONS</small>
                {results.length > 0 && (
                  <div id={listboxId} role="listbox" aria-label="Location suggestions">
                    {results.map((result, index) => (
                      <button
                        id={`${listboxId}-option-${index}`}
                        key={result.id}
                        type="button"
                        role="option"
                        aria-selected={index === activeIndex}
                        className={index === activeIndex ? "active" : undefined}
                        onMouseMove={() => setActiveIndex(index)}
                        onClick={() => selectLocation(result)}
                      >
                        <MapPin size={15} />
                        <span>
                          <strong>{result.primaryLabel}</strong>
                          {result.secondaryLabel && <small>{result.secondaryLabel}</small>}
                        </span>
                        <CornerDownLeft size={13} />
                      </button>
                    ))}
                  </div>
                )}
                {searching && <p className="command-search-state">Searching locations…</p>}
                {queryIsCurrent && geocodeQuery.isError && !coordinateResult && (
                  <p className="command-search-state error" role="status">
                    Location search unavailable
                  </p>
                )}
                {queryIsCurrent &&
                  !searching &&
                  !geocodeQuery.isError &&
                  results.length === 0 &&
                  !suggestionsDismissed && (
                    <p className="command-search-state">No locations found</p>
                  )}
              </div>
            )}
            <div className="command-group">
              <small>JUMP TO REGION</small>
              {cities.map((city) => (
                <button key={city.id} onClick={() => run(() => store.setCity(city.id))}>
                  <MapPin size={15} />
                  <span>
                    <strong>{city.name}</strong>
                    <small>{city.timezone}</small>
                  </span>
                  <CornerDownLeft size={13} />
                </button>
              ))}
            </div>
            {!isLocationQuery && (
              <>
                <div className="command-group">
                  <small>DISPLAY</small>
                  <button onClick={() => run(() => store.toggleLayer("coverage"))}>
                    <Layers3 size={15} />
                    <span>
                      <strong>Toggle model coverage</strong>
                      <small>Show feature eligibility states</small>
                    </span>
                  </button>
                  <button
                    onClick={() => run(() => store.setMode(store.mode === "2d" ? "3d" : "2d"))}
                  >
                    <Box size={15} />
                    <span>
                      <strong>Toggle {store.mode === "2d" ? "3D" : "2D"} surface</strong>
                      <small>Change analytical elevation</small>
                    </span>
                  </button>
                  <button
                    onClick={() => run(() => store.setTimestamp("2024-08-21T22:00:00.000Z"))}
                  >
                    <Clock3 size={15} />
                    <span>
                      <strong>Reset model time</strong>
                      <small>Return to fixture reference time</small>
                    </span>
                  </button>
                </div>
                <div className="command-group shortcuts">
                  <small>KEYBOARD</small>
                  <p>
                    <span>Step through time</span>
                    <kbd>←</kbd>
                    <kbd>→</kbd>
                  </p>
                  <p>
                    <span>Play / pause</span>
                    <kbd>Space</kbd>
                  </p>
                  <p>
                    <span>Close inspector</span>
                    <kbd>Esc</kbd>
                  </p>
                </div>
              </>
            )}
          </div>
          <div className="command-footer">
            <span>
              <Activity size={12} /> CrimeNet command interface
            </span>
            <span>
              <kbd>↑↓</kbd> Navigate <kbd>↵</kbd> Select
            </span>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
