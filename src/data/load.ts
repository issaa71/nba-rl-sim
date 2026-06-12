// Async loaders for the static JSON payloads under /public/data.
//
// Weights are large (≈430 KB + 397 KB) so we load them lazily and in parallel
// with the lighter possession/player payloads.

import { loadNetwork, type LoadedNetwork, type WeightsFile } from "../engine/network";
import { BASKET_X, BASKET_Y } from "../engine/features";
import type { PlayersFile, PossessionsFile } from "./types";

// Vite injects `import.meta.env.BASE_URL`; read it defensively so this module
// also type-checks under the test tsconfig (which doesn't load `vite/client`).
const BASE =
  (import.meta as { env?: { BASE_URL?: string } }).env?.BASE_URL ?? "/";

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}data/${path}`);
  if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
  return (await res.json()) as T;
}

export interface AppData {
  possessions: PossessionsFile;
  players: PlayersFile;
  dueling: LoadedNetwork;
  dqn: LoadedNetwork;
  /** zone FG% by compact id, plus a league-average fallback row. */
  zoneFg: ZoneFgTable;
}

export interface ZoneFgTable {
  byCompactId: Record<string, Record<string, number>>;
  leagueAvg: Record<string, number>;
}

interface ZoneFgFile {
  zone_definitions: Record<string, string>;
  special_entries: { __league_avg__: Record<string, number> };
  by_compact_id: Record<string, Record<string, number>>;
}

export async function loadAppData(): Promise<AppData> {
  const [possessions, players, duelingFile, dqnFile, zoneFgFile] =
    await Promise.all([
      getJson<PossessionsFile>("possessions.json"),
      getJson<PlayersFile>("players.json"),
      getJson<WeightsFile>("model_weights.dueling.json"),
      getJson<WeightsFile>("model_weights.dqn.json"),
      getJson<ZoneFgFile>("zone_fg.json"),
    ]);

  return {
    possessions,
    players,
    dueling: loadNetwork(duelingFile),
    dqn: loadNetwork(dqnFile),
    zoneFg: {
      byCompactId: zoneFgFile.by_compact_id,
      leagueAvg: zoneFgFile.special_entries.__league_avg__,
    },
  };
}

// ---------------------------------------------------------------------------
// Zone-FG lookup (distance -> 5-ft bucket -> FG%) for what-if BH recompute.
// Zones: 0:0-5, 1:5-10, 2:10-15, 3:15-20, 4:20-25, 5:25-30, 6:30+ ft.
// ---------------------------------------------------------------------------

export function distanceToZoneIdx(dist: number): number {
  const z = Math.floor(dist / 5);
  return z > 6 ? 6 : z;
}

/**
 * Make a `(x, y) -> FG%` lookup for a given ball-handler compact id, falling
 * back to league average when a player has no data in that zone.
 */
export function makeZoneFgLookup(
  table: ZoneFgTable,
  compactId: number,
): (x: number, y: number) => number {
  const row = table.byCompactId[String(compactId)];
  return (x: number, y: number) => {
    const dx = x - BASKET_X;
    const dy = y - BASKET_Y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    const zone = String(distanceToZoneIdx(dist));
    const v = row?.[zone];
    if (v !== undefined && v > 0) return v;
    return table.leagueAvg[zone] ?? 0;
  };
}
