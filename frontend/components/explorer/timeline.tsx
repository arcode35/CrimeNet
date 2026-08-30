"use client";

import * as Slider from "@radix-ui/react-slider";
import { ChevronLeft, ChevronRight, Pause, Play, RotateCcw } from "lucide-react";
import { useEffect } from "react";
import { getCity, type PredictionResponse } from "@/lib/domain";
import { formatTimestamp } from "@/lib/format";
import { useExplorerStore } from "@/stores/explorer-store";

export function Timeline({
  data,
  isFetching,
  liveMode = false,
}: {
  data?: PredictionResponse;
  isFetching: boolean;
  liveMode?: boolean;
}) {
  const state = useExplorerStore();
  const playing = state.playing;
  const stepTime = state.stepTime;
  const city = getCity(state.cityId);
  useEffect(() => {
    if (!playing || liveMode) return;
    const timer = window.setInterval(() => stepTime(1), 1150);
    return () => window.clearInterval(timer);
  }, [liveMode, playing, stepTime]);
  const date = new Date(state.timestamp);
  const value = date.getUTCHours();
  const setHour = ([hour]: number[]) => {
    const next = new Date(state.timestamp);
    next.setUTCHours(hour, 0, 0, 0);
    state.setTimestamp(next.toISOString());
  };
  const cells = data?.cells.filter((cell) => cell.intensity !== null) ?? [];
  const total = cells.reduce((sum, cell) => sum + (cell.intensity ?? 0), 0);
  return (
    <section
      className={`timeline ${liveMode ? "live-timeline" : ""}`}
      aria-label="Prediction time controls"
    >
      <div className="timeline-controls">
        <button
          className="play-button"
          onClick={() => state.setPlaying(!state.playing)}
          aria-label={state.playing ? "Pause playback" : "Play timeline"}
          disabled={liveMode}
        >
          {state.playing ? <Pause size={14} /> : <Play size={14} fill="currentColor" />}
        </button>
        <button onClick={() => state.stepTime(-1)} aria-label="Previous hour" disabled={liveMode}>
          <ChevronLeft size={15} />
        </button>
        <button onClick={() => state.stepTime(1)} aria-label="Next hour" disabled={liveMode}>
          <ChevronRight size={15} />
        </button>
      </div>
      <div className="timeline-time">
        <small>
          {liveMode
            ? "LIVE SNAPSHOT · UTC"
            : `MODEL TIME · ${city.timezone.split("/")[1].replace("_", " ").toUpperCase()}`}
        </small>
        <strong>{formatTimestamp(state.timestamp, liveMode ? "UTC" : city.timezone)}</strong>
      </div>
      {liveMode ? (
        <div className="rail-wrap live-hour-rail">
          <span />
          <strong>CURRENT-HOUR RATE · NO HISTORICAL OR FUTURE INFERENCE</strong>
        </div>
      ) : (
        <div className="rail-wrap">
          <div className="sparkline" aria-hidden="true">
            {Array.from({ length: 48 }, (_, index) => (
              <i key={index} style={{ height: `${18 + ((index * 17 + value * 9) % 42)}%` }} />
            ))}
          </div>
          <Slider.Root
            className="slider"
            min={0}
            max={23}
            step={1}
            value={[value]}
            onValueChange={setHour}
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
          <small>HORIZON</small>
          <strong>{liveMode ? "CURRENT +1h" : `+${state.horizonHours}h`}</strong>
        </span>
        <button
          onClick={() => state.setTimestamp("2024-08-21T22:00:00.000Z")}
          aria-label="Reset time"
          disabled={liveMode}
        >
          <RotateCcw size={13} />
        </button>
      </div>
    </section>
  );
}
