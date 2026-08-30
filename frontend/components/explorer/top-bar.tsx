"use client";

import Link from "next/link";
import { Activity, Command, Cpu, Hexagon, Search } from "lucide-react";
import { useExplorerStore } from "@/stores/explorer-store";

export function TopBar({
  fixtureMode,
  snapshotId,
  serviceDegraded = false,
}: {
  fixtureMode: boolean;
  snapshotId?: string;
  serviceDegraded?: boolean;
}) {
  const setCommandOpen = useExplorerStore((state) => state.setCommandOpen);
  return (
    <header className="top-bar">
      <div className="brand">
        <span className="brand-mark">
          <Hexagon size={17} strokeWidth={1.5} />
        </span>
        <div>
          <strong>CRIMESENSE</strong>
          <small>POWERED BY CRIMENET</small>
        </div>
      </div>
      <nav aria-label="Primary navigation">
        <Link href="/explorer" className="nav-active">
          Explorer
        </Link>
        <Link href="/model">Model</Link>
      </nav>
      <div className="top-actions">
        {fixtureMode && (
          <span className="fixture-flag">
            <span /> DEVELOPMENT FIXTURE
          </span>
        )}
        {!fixtureMode && (
          <span className="live-flag">
            <span /> LIVE API
          </span>
        )}
        <span className={`system-state ${serviceDegraded ? "degraded" : ""}`}>
          <Activity size={13} />
          {fixtureMode
            ? "CONTRACT PREVIEW"
            : serviceDegraded
              ? "SERVICE DEGRADED"
              : snapshotId
                ? `SNAPSHOT ${snapshotId}`
                : "CONNECTING"}
        </span>
        <button
          className="command-trigger"
          onClick={() => setCommandOpen(true)}
          aria-label="Open command palette"
        >
          <Search size={14} />
          <span>Search or command</span>
          <kbd>
            <Command size={11} />K
          </kbd>
        </button>
        <button className="icon-button" aria-label="Model runtime">
          <Cpu size={15} />
        </button>
      </div>
    </header>
  );
}
