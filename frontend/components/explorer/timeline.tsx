"use client";

import * as Slider from "@radix-ui/react-slider";
import { ChevronLeft, ChevronRight, Pause, Play, RotateCcw } from "lucide-react";
import { useEffect } from "react";
import type { IntensityTimelineSnapshot } from "@/lib/api";
import { getCity, type PredictionResponse } from "@/lib/domain";
import { forecastHorizonLabel } from "@/lib/forecast";
import { formatTimestamp } from "@/lib/format";
import { useExplorerStore } from "@/stores/explorer-store";

function localForecastParts(validUtcHour: string) {
  const parts = new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZoneName: "short",
  }).formatToParts(new Date(validUtcHour));
  const value = (type: Intl.DateTimeFormatPartTypes) =>
    parts.find((part) => part.type === type)?.value ?? "";
  return {
    time: `${value("hour")}:${value("minute")} ${value("dayPeriod")}`.trim(),
    date: `${value("month")} ${value("day")}`,
    zone: value("timeZoneName"),
  };
}

function timelineLabelIndexes(snapshotCount: number) {
  if (snapshotCount <= 1) return [0];
  return [...new Set([0, 0.25, 0.5, 0.75, 1].map((ratio) => Math.round((snapshotCount - 1) * ratio)))];
}

export function Timeline({
  data,
  isFetching,
  liveMode = false,
  snapshots = [],
  selectedIndex = 0,
  onSelectedIndexChange,
  forecastUnavailable = false,
  forecastLoading = false,
}: {
  data?: PredictionResponse;
  isFetching: boolean;
  liveMode?: boolean;
  snapshots?: readonly IntensityTimelineSnapshot[];
  selectedIndex?: number;
  onSelectedIndexChange?: (index: number) => void;
  forecastUnavailable?: boolean;
  forecastLoading?: boolean;
}) {
  const state = useExplorerStore();
  const playing = state.playing;
  const stepTime = state.stepTime;
  const city = getCity(state.cityId);
  const selectedSnapshot = snapshots[selectedIndex];
  const forecastEnabled = liveMode && snapshots.length > 1 && !forecastUnavailable;

  useEffect(() => {
    if (!playing) return;
    if (liveMode) {
      if (!forecastEnabled || !onSelectedIndexChange) return;
      const timer = window.setInterval(() => {
        if (selectedIndex >= snapshots.length - 1) {
          state.setPlaying(false);
          return;
        }
        onSelectedIndexChange(selectedIndex + 1);
      }, 1_150);
      return () => window.clearInterval(timer);
    }
    const timer = window.setInterval(() => stepTime(1), 1_150);
    return () => window.clearInterval(timer);
  }, [
    forecastEnabled,
    liveMode,
    onSelectedIndexChange,
    playing,
    selectedIndex,
    snapshots.length,
    state,
    stepTime,
  ]);

  const fixtureDate = new Date(state.timestamp);
  const fixtureHour = fixtureDate.getUTCHours();
  const localTime = selectedSnapshot
    ? localForecastParts(selectedSnapshot.valid_utc_hour)
    : undefined;
  const cells = data?.cells.filter((cell) => cell.intensity !== null) ?? [];
  const total = cells.reduce((sum, cell) => sum + (cell.intensity ?? 0), 0);

  const setFixtureHour = ([hour]: number[]) => {
    const next = new Date(state.timestamp);
    next.setUTCHours(hour, 0, 0, 0);
    state.setTimestamp(next.toISOString());
  };
  const selectForecast = ([index]: number[]) => onSelectedIndexChange?.(index);
  const togglePlayback = () => {
    if (liveMode && selectedIndex >= snapshots.length - 1) onSelectedIndexChange?.(0);
    state.setPlaying(!state.playing);
  };

  return (
    <section
      className={`timeline ${liveMode ? "live-timeline forecast-timeline" : ""}`}
      aria-label="Prediction time controls"
    >
      <div className="timeline-controls">
        <button
          className="play-button"
          onClick={togglePlayback}
          aria-label={state.playing ? "Pause forecast playback" : "Play forecast timeline"}
          disabled={liveMode && !forecastEnabled}
        >
          {state.playing ? <Pause size={14} /> : <Play size={14} fill="currentColor" />}
        </button>
        <button
          onClick={() =>
            liveMode
              ? onSelectedIndexChange?.(Math.max(0, selectedIndex - 1))
              : state.stepTime(-1)
          }
          aria-label="Previous hour"
          disabled={liveMode && (!forecastEnabled || selectedIndex === 0)}
        >
          <ChevronLeft size={15} />
        </button>
        <button
          onClick={() =>
            liveMode
              ? onSelectedIndexChange?.(Math.min(snapshots.length - 1, selectedIndex + 1))
              : state.stepTime(1)
          }
          aria-label="Next hour"
          disabled={liveMode && (!forecastEnabled || selectedIndex >= snapshots.length - 1)}
        >
          <ChevronRight size={15} />
        </button>
      </div>
      <div className="timeline-time">
        <small>
          {liveMode
            ? selectedSnapshot?.kind === "forecast"
              ? `+${selectedSnapshot.horizon_hours}H FORECAST`
              : "LIVE · NOW"
            : `MODEL TIME · ${city.timezone.split("/")[1].replace("_", " ").toUpperCase()}`}
        </small>
        <strong>
          {liveMode && localTime
            ? `${localTime.time} · ${localTime.zone}`
            : formatTimestamp(state.timestamp, city.timezone)}
        </strong>
        {liveMode && localTime && <span className="timeline-date">{localTime.date}</span>}
      </div>
      {liveMode ? (
        forecastEnabled && selectedSnapshot ? (
          <div className="rail-wrap forecast-rail">
            <div className="forecast-ticks" aria-hidden="true">
              {snapshots.map((snapshot, index) => (
                <i
                  key={snapshot.valid_utc_hour}
                  className={index === selectedIndex ? "active" : undefined}
                />
              ))}
            </div>
            <Slider.Root
              className="slider"
              min={0}
              max={snapshots.length - 1}
              step={1}
              value={[selectedIndex]}
              onValueChange={selectForecast}
            >
              <Slider.Track className="slider-track">
                <Slider.Range className="slider-range" />
              </Slider.Track>
              <Slider.Thumb
                className="slider-thumb"
                aria-label="Forecast hour"
                aria-valuetext={`${forecastHorizonLabel(selectedSnapshot)} ${localTime!.time} ${localTime!.date}`}
              />
            </Slider.Root>
            <div className="rail-labels forecast-labels">
              {timelineLabelIndexes(snapshots.length).map((index) => (
                <span key={snapshots[index].valid_utc_hour}>
                  {forecastHorizonLabel(snapshots[index])}
                </span>
              ))}
            </div>
          </div>
        ) : (
          <div className="rail-wrap live-hour-rail" role="status">
            <span />
            <strong>
              {forecastUnavailable
                ? "FORECAST UNAVAILABLE · LIVE DATA REMAINS ACTIVE"
                : forecastLoading
                  ? "LOADING ROLLING FORECAST"
                  : "LIVE HOURLY RISK"}
            </strong>
          </div>
        )
      ) : (
        <div className="rail-wrap">
          <div className="sparkline" aria-hidden="true">
            {Array.from({ length: 48 }, (_, index) => (
              <i key={index} style={{ height: `${18 + ((index * 17 + fixtureHour * 9) % 42)}%` }} />
            ))}
          </div>
          <Slider.Root
            className="slider"
            min={0}
            max={23}
            step={1}
            value={[fixtureHour]}
            onValueChange={setFixtureHour}
          >
            <Slider.Track className="slider-track">
              <Slider.Range className="slider-range" />
            </Slider.Track>
            <Slider.Thumb className="slider-thumb" aria-label="Hour" />
          </Slider.Root>
          <div className="rail-labels">
            <span>00:00</span>
            <span>06:00</span>
            <span>12:00</span>
            <span>18:00</span>
            <span>24:00</span>
          </div>
        </div>
      )}
      <div className="timeline-summary">
        <span className={isFetching ? "updating" : ""}>
          <small>VISIBLE MODEL MASS</small>
          <strong>{isFetching ? "Updating" : total.toFixed(2)}</strong>
        </span>
        <span>
          <small>{selectedSnapshot?.kind === "forecast" ? "RISK FORECAST" : "HORIZON"}</small>
          <strong>
            {liveMode && selectedSnapshot
              ? forecastHorizonLabel(selectedSnapshot)
              : `+${state.horizonHours}h`}
          </strong>
        </span>
        <button
          onClick={() =>
            liveMode
              ? onSelectedIndexChange?.(0)
              : state.setTimestamp("2024-08-21T22:00:00.000Z")
          }
          aria-label={liveMode ? "Return to live" : "Reset time"}
          disabled={liveMode && (!selectedSnapshot || selectedIndex === 0)}
        >
          <RotateCcw size={13} />
        </button>
      </div>
    </section>
  );
}
