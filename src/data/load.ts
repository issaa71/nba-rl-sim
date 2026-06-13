// Async loaders for the static JSON payloads under /public/data.
//
// Weights are large (≈430 KB + 397 KB) so we load them lazily and in parallel
// with the lighter possession/player payloads.

import { loadNetwork, type LoadedNetwork, type WeightsFile } from "../engine/network";
import { BASKET_X, BASKET_Y } from "../engine/features";
import type {
  PlayersFile,
  Possession,
  PossessionsFile,
  TrackingFile,
  TrackingPossession,
} from "./types";

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
  /**
   * Real high-frequency tracking for the curated 40, keyed by possession id.
   * Loaded upfront alongside the explorer payloads (it drives the headline
   * real-time playback). A possession with NO entry falls back to the old
   * stepped, low-rate view.
   */
  tracking: Map<string, TrackingPossession>;
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
  const [possessions, players, duelingFile, dqnFile, zoneFgFile, trackingFile] =
    await Promise.all([
      getJson<PossessionsFile>("possessions.json"),
      getJson<PlayersFile>("players.json"),
      getJson<WeightsFile>("model_weights.dueling.json"),
      getJson<WeightsFile>("model_weights.dqn.json"),
      getJson<ZoneFgFile>("zone_fg.json"),
      // Curated tracking ships with the explorer — it IS the headline playback.
      // If it 404s the explorer still works via the stepped fallback view.
      getJson<TrackingFile>("tracking_curated.json").catch(
        () => null as TrackingFile | null,
      ),
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
    tracking: indexTracking(trackingFile),
  };
}

/** Index a tracking file's possessions by id (empty map when absent). */
export function indexTracking(
  file: TrackingFile | null,
): Map<string, TrackingPossession> {
  const map = new Map<string, TrackingPossession>();
  if (file) for (const t of file.possessions) map.set(t.id, t);
  return map;
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

// ---------------------------------------------------------------------------
// Watch-mode rotation: lazily fetched stream possessions merged with the
// curated 40 into a single shuffled rotation.
//
// The stream payload (~1.8 MB) is fetched ONLY when watch mode first starts —
// the landing/browser view never pays for it. Callers should cache the promise
// so a second visit to watch mode reuses the already-fetched data.
// ---------------------------------------------------------------------------

interface StreamFile {
  note: string;
  model: string;
  possessions: Possession[];
}

let streamPromise: Promise<Possession[]> | null = null;

/**
 * Dynamically fetch the stream possessions. Memoized: the network request fires
 * at most once per session. Not part of `loadAppData` — the browser/landing
 * view must not pay for this payload.
 */
export function loadStreamPossessions(): Promise<Possession[]> {
  if (!streamPromise) {
    streamPromise = getJson<StreamFile>("possessions_stream.json")
      .then((f) => f.possessions)
      .catch((e: unknown) => {
        // Allow a later retry if the fetch failed.
        streamPromise = null;
        throw e;
      });
  }
  return streamPromise;
}

let streamTrackingPromise: Promise<Map<string, TrackingPossession>> | null =
  null;

/**
 * Lazily fetch the stream tracking (the ~9 MB high-frequency motion for the
 * watch-mode rotation), indexed by possession id. Memoized; fetched ONLY when
 * watch mode first asks for it — the landing/browser/explorer views never pay
 * for it. A failed fetch resolves to an empty map (those possessions fall back
 * to the stepped low-rate view) and is retried on the next call.
 */
export function loadStreamTracking(): Promise<Map<string, TrackingPossession>> {
  if (!streamTrackingPromise) {
    streamTrackingPromise = getJson<TrackingFile>("tracking_stream.json")
      .then(indexTracking)
      .catch(() => {
        streamTrackingPromise = null;
        return new Map<string, TrackingPossession>();
      });
  }
  return streamTrackingPromise;
}

/**
 * Deterministic (seeded) Fisher–Yates shuffle — a small LCG keeps the rotation
 * stable across reloads so the "possession N of M" counter is reproducible.
 */
export function seededShuffle<T>(items: readonly T[], seed: number): T[] {
  const out = items.slice();
  // LCG (Numerical Recipes constants); avoids pulling in a dependency.
  let s = (seed >>> 0) || 1;
  const next = () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 0x100000000;
  };
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(next() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

/** Build the shuffled watch rotation: curated possessions + stream possessions. */
export function buildWatchRotation(
  curated: readonly Possession[],
  stream: readonly Possession[],
  seed = 42,
): Possession[] {
  return seededShuffle([...curated, ...stream], seed);
}
