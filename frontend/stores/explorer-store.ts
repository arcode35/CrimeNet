import { create } from "zustand";

export type MapMode = "2d" | "3d";
export type BasemapMode = "dark" | "satellite";
export type LayerKey = "prediction" | "coverage" | "historical";

type ExplorerState = {
  cityId: string;
  timestamp: string;
  horizonHours: number;
  selectedH3: string | null;
  hoveredH3: string | null;
  mode: MapMode;
  basemapMode: BasemapMode;
  layers: Record<LayerKey, boolean>;
  playing: boolean;
  controlsCollapsed: boolean;
  commandOpen: boolean;
  setCity: (cityId: string) => void;
  setTimestamp: (timestamp: string) => void;
  stepTime: (hours: number) => void;
  setHorizon: (hours: number) => void;
  selectCell: (h3: string | null) => void;
  hoverCell: (h3: string | null) => void;
  setMode: (mode: MapMode) => void;
  setBasemapMode: (mode: BasemapMode) => void;
  toggleLayer: (layer: LayerKey) => void;
  setPlaying: (playing: boolean) => void;
  setControlsCollapsed: (collapsed: boolean) => void;
  setCommandOpen: (open: boolean) => void;
};

const initialTime = "2024-08-21T22:00:00.000Z";

export const useExplorerStore = create<ExplorerState>((set) => ({
  cityId: "chicago",
  timestamp: initialTime,
  horizonHours: 1,
  selectedH3: null,
  hoveredH3: null,
  mode: "3d",
  basemapMode: "dark",
  layers: { prediction: true, coverage: false, historical: false },
  playing: false,
  controlsCollapsed: false,
  commandOpen: false,
  setCity: (cityId) => set({ cityId, selectedH3: null, hoveredH3: null }),
  setTimestamp: (timestamp) => set({ timestamp }),
  stepTime: (hours) =>
    set((state) => ({
      timestamp: new Date(new Date(state.timestamp).getTime() + hours * 3_600_000).toISOString(),
    })),
  setHorizon: (horizonHours) => set({ horizonHours }),
  selectCell: (selectedH3) => set({ selectedH3 }),
  hoverCell: (hoveredH3) => set({ hoveredH3 }),
  setMode: (mode) => set({ mode }),
  setBasemapMode: (basemapMode) => set({ basemapMode }),
  toggleLayer: (layer) =>
    set((state) => ({ layers: { ...state.layers, [layer]: !state.layers[layer] } })),
  setPlaying: (playing) => set({ playing }),
  setControlsCollapsed: (controlsCollapsed) => set({ controlsCollapsed }),
  setCommandOpen: (commandOpen) => set({ commandOpen }),
}));
