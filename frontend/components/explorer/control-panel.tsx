"use client";

import * as Select from "@radix-ui/react-select";
import * as Switch from "@radix-ui/react-switch";
import {
  Box,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronsUpDown,
  CircleDot,
  Layers3,
  MapPin,
  PanelLeftClose,
  PanelLeftOpen,
} from "lucide-react";
import { CITIES, getCity } from "@/lib/domain";
import { formatTimestamp } from "@/lib/format";
import { type LayerKey, useExplorerStore } from "@/stores/explorer-store";

function FieldSelect({
  value,
  onChange,
  children,
  label,
}: {
  value: string;
  onChange: (value: string) => void;
  children: React.ReactNode;
  label: string;
}) {
  return (
    <Select.Root value={value} onValueChange={onChange}>
      <Select.Trigger className="select-trigger" aria-label={label}>
        <Select.Value />
        <Select.Icon>
          <ChevronsUpDown size={13} />
        </Select.Icon>
      </Select.Trigger>
      <Select.Portal>
        <Select.Content className="select-content" position="popper" sideOffset={6}>
          <Select.Viewport>{children}</Select.Viewport>
        </Select.Content>
      </Select.Portal>
    </Select.Root>
  );
}

function LayerToggle({
  id,
  label,
  detail,
  disabled,
}: {
  id: LayerKey;
  label: string;
  detail: string;
  disabled?: boolean;
}) {
  const checked = useExplorerStore((state) => state.layers[id]);
  const toggle = useExplorerStore((state) => state.toggleLayer);
  return (
    <div className={`layer-row ${disabled ? "is-disabled" : ""}`}>
      <span className="layer-symbol">
        {id === "prediction" ? (
          <CircleDot size={14} />
        ) : id === "coverage" ? (
          <Layers3 size={14} />
        ) : (
          <MapPin size={14} />
        )}
      </span>
      <label htmlFor={`layer-${id}`}>
        <strong>{label}</strong>
        <small>{detail}</small>
      </label>
      <Switch.Root
        id={`layer-${id}`}
        className="switch"
        checked={checked}
        onCheckedChange={() => toggle(id)}
        disabled={disabled}
      >
        <Switch.Thumb className="switch-thumb" />
      </Switch.Root>
    </div>
  );
}

export function ControlPanel() {
  const state = useExplorerStore();
  const city = getCity(state.cityId);
  const satelliteConfigured = Boolean(process.env.NEXT_PUBLIC_MAPTILER_KEY);
  if (state.controlsCollapsed)
    return (
      <button className="controls-reopen" onClick={() => state.setControlsCollapsed(false)}>
        <PanelLeftOpen size={16} />
        <span>Controls</span>
      </button>
    );
  return (
    <aside className="control-panel" aria-label="Prediction controls">
      <div className="panel-heading">
        <div>
          <small>INFERENCE EXPLORER</small>
          <strong>Prediction surface</strong>
        </div>
        <button
          className="icon-button quiet"
          onClick={() => state.setControlsCollapsed(true)}
          aria-label="Collapse controls"
        >
          <PanelLeftClose size={15} />
        </button>
      </div>
      <section className="control-section">
        <label className="field-label">SUPPORTED REGION</label>
        <FieldSelect value={state.cityId} onChange={state.setCity} label="Supported city">
          {CITIES.map((item) => (
            <Select.Item className="select-item" value={item.id} key={item.id}>
              <Select.ItemText>{item.name}</Select.ItemText>
              <Select.ItemIndicator>
                <Check size={13} />
              </Select.ItemIndicator>
            </Select.Item>
          ))}
        </FieldSelect>
        <div className="coverage-line">
          <span className="coverage-dot" /> MODEL JURISDICTION <small>H3 · R9</small>
        </div>
      </section>
      <section className="control-section">
        <label className="field-label">MODEL OUTPUT</label>
        <button className="locked-field">
          <span>
            <strong>All reported events</strong>
            <small>Point-process intensity</small>
          </span>
          <ChevronDown size={13} />
        </button>
      </section>
      <section className="control-section">
        <div className="field-label split">
          <span>PREDICTION TIME</span>
          <span>LOCAL</span>
        </div>
        <div className="time-field">
          <span>{formatTimestamp(state.timestamp, city.timezone)}</span>
          <small>Model time is shareable via URL</small>
        </div>
        <label className="field-label section-gap">HORIZON</label>
        <div className="segment-control">
          {[1, 6, 12, 24].map((hours) => (
            <button
              key={hours}
              className={state.horizonHours === hours ? "active" : ""}
              onClick={() => state.setHorizon(hours)}
            >
              {hours}h
            </button>
          ))}
        </div>
      </section>
      <section className="control-section layer-section">
        <div className="field-label split">
          <span>MAP LAYERS</span>
          <Layers3 size={12} />
        </div>
        <LayerToggle id="prediction" label="Predicted intensity" detail="GPU H3 surface" />
        <LayerToggle id="coverage" label="Model coverage" detail="Feature eligibility" />
        <LayerToggle
          id="historical"
          label="Historical events"
          detail="Endpoint unavailable"
          disabled
        />
      </section>
      <section className="control-section display-section">
        <label className="field-label">DISPLAY</label>
        <div className="segment-control">
          <button
            className={state.mode === "2d" ? "active" : ""}
            onClick={() => state.setMode("2d")}
          >
            <Box size={13} /> 2D
          </button>
          <button
            className={state.mode === "3d" ? "active" : ""}
            onClick={() => state.setMode("3d")}
          >
            <Box size={13} /> 3D
          </button>
        </div>
        <label className="field-label section-gap">BASEMAP</label>
        <div className="segment-control" role="group" aria-label="Basemap">
          <button
            className={state.basemapMode === "dark" ? "active" : ""}
            aria-pressed={state.basemapMode === "dark"}
            onClick={() => state.setBasemapMode("dark")}
          >
            DARK
          </button>
          <button
            className={state.basemapMode === "satellite" ? "active" : ""}
            aria-pressed={state.basemapMode === "satellite"}
            disabled={!satelliteConfigured}
            title={
              satelliteConfigured ? "MapTiler satellite imagery" : "MapTiler key not configured"
            }
            onClick={() => state.setBasemapMode("satellite")}
          >
            SATELLITE
          </button>
        </div>
        {!satelliteConfigured && (
          <small className="basemap-config-note">Satellite key not configured</small>
        )}
      </section>
      <div className="panel-coordinate">
        <ChevronLeft size={12} /> {city.center[1].toFixed(4)}°N&nbsp;{" "}
        {Math.abs(city.center[0]).toFixed(4)}°W
      </div>
    </aside>
  );
}
