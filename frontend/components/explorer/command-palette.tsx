"use client";

import * as Dialog from "@radix-ui/react-dialog";
import {
  Activity,
  Box,
  Clock3,
  Command,
  CornerDownLeft,
  Layers3,
  MapPin,
  Search,
  X,
} from "lucide-react";
import { useState } from "react";
import { CITIES } from "@/lib/domain";
import { useExplorerStore } from "@/stores/explorer-store";

export function CommandPalette() {
  const open = useExplorerStore((state) => state.commandOpen);
  const setOpen = useExplorerStore((state) => state.setCommandOpen);
  const store = useExplorerStore();
  const [query, setQuery] = useState("");
  const cities = CITIES.filter((city) => city.name.toLowerCase().includes(query.toLowerCase()));
  const run = (action: () => void) => {
    action();
    setOpen(false);
    setQuery("");
  };
  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className="command-dialog" aria-describedby={undefined}>
          <Dialog.Title className="sr-only">CrimeNet commands</Dialog.Title>
          <div className="command-search">
            <Search size={17} />
            <input
              autoFocus
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search cities and commands…"
            />
            <kbd>
              <Command size={11} />K
            </kbd>
            <Dialog.Close>
              <X size={15} />
            </Dialog.Close>
          </div>
          <div className="command-body">
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
            <div className="command-group">
              <small>DISPLAY</small>
              <button onClick={() => run(() => store.toggleLayer("coverage"))}>
                <Layers3 size={15} />
                <span>
                  <strong>Toggle model coverage</strong>
                  <small>Show feature eligibility states</small>
                </span>
              </button>
              <button onClick={() => run(() => store.setMode(store.mode === "2d" ? "3d" : "2d"))}>
                <Box size={15} />
                <span>
                  <strong>Toggle {store.mode === "2d" ? "3D" : "2D"} surface</strong>
                  <small>Change analytical elevation</small>
                </span>
              </button>
              <button onClick={() => run(() => store.setTimestamp("2024-08-21T22:00:00.000Z"))}>
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
