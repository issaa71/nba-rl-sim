// Real-time tracking playback engine — TRUE wall-clock playback over the
// high-frequency SportVU frames (public/data/tracking_*.json).
//
// Playback is wall-clock against the tracking timeline: each frame carries a
// real time offset `t` (s), frames are ~80 ms apart (12.5 fps after dedup), and
// the head linearly interpolates between the two ADJACENT tracking frames that
// bracket the current time. Because adjacent frames are genuinely 80 ms apart,
// that interpolation is honest motion (no dwell/glide/cut fiction). The speed
// control scales the clock.
//
// COORDINATE FRAMES. Tracking positions have the basket on the LEFT at
// (5.25, 25); x in [0, 47], y in [0, 50]. The feature/possessions frame (the
// one engine/features.ts + the Court renderer use) has the basket on the RIGHT
// at x = 47. The two are related by a single reflection in x that PRESERVES
// distance-to-basket exactly (verified against possessions.json: tracking
// distance-to-(5.25,25) == recorded distance_to_basket to 1e-2):
//
//   x_feature = (BASKET_X + TRACK_BASKET_X) - x_track = 52.25 - x_track
//   y_feature = y_track
//
// Velocity, being a displacement rate, transforms the same way:
//   vx_feature = -vx_track,  vy_feature = vy_track
//
// VELOCITY CONVENTION. The env derives entity velocity as displacement between
// consecutive tracking samples over the sample period: v = Δpos / Δt, with
// Δt = 1 / fps. We use a backward difference (forward at frame 0) so the
// velocity at frame i reflects the motion that just happened, then map it into
// the feature frame. (The recorded decision-frame vx/vy in possessions.json
// were computed in the env from the original pre-dedup stream and are NOT
// reproducible from this curated export — so at snap indices the UI uses the
// parity-exact recorded path, never tracking-derived velocity.)

import {
  BASKET_X,
  deriveContext,
  distanceToBasket,
  type RawState,
} from "./features";
import type { TrackingPossession } from "../data/types";

// Tracking-frame basket x. The reflection axis is (BASKET_X + TRACK_BASKET_X).
export const TRACK_BASKET_X = 5.25;
const X_REFLECT = BASKET_X + TRACK_BASKET_X; // 52.25

// Court render bounds (must match Court.tsx COURT_X / COURT_Y).
const RENDER_X_MAX = 47;
const RENDER_Y_MAX = 50;

// Teleport guard. The dedup'd SportVU export still carries occasional entity
// ID-swaps / tracking glitches: ~6% of adjacent frame-steps move an entity
// >20 ft in one 80 ms step (non-physical — NBA top speed is ~3 ft / 80 ms).
// Linearly interpolating across one of those would streak a player across the
// court. When an entity's adjacent-frame displacement exceeds this many feet we
// SNAP it at the frame boundary instead of gliding, and treat its velocity as
// zero (so the spurious ~500 ft/s does not poison the live agent's features).
const TELEPORT_FT = 6;

/** Map a tracking x into the feature/render frame (basket on the right). */
export function trackXToFeature(xTrack: number): number {
  return X_REFLECT - xTrack;
}

/** A position in the feature/render frame plus its derived velocity. */
export interface TrackEntity {
  x: number;
  y: number;
  vx: number;
  vy: number;
}

/** A fully resolved playback snapshot in the feature/render frame. */
export interface TrackSnapshot {
  /** 10 entities [BH, tm1..4, def1..5] in network order, feature frame. */
  players: TrackEntity[];
  /** ball [x, y] in the feature/render frame. */
  ball: { x: number; y: number };
  /** raw ball height (ft) — drives the shot-arc render. */
  ballZ: number;
  /** interpolated shot clock (s) or null when unavailable. */
  shotClock: number | null;
  /** fractional tracking-frame position (continuous). */
  framePos: number;
}

const lerp = (a: number, b: number, f: number): number => a + (b - a) * f;
const clamp = (v: number, lo: number, hi: number): number =>
  v < lo ? lo : v > hi ? hi : v;

/**
 * Velocity at tracking-frame index `i`, in the FEATURE frame, via the env
 * convention v = Δpos / Δt (backward difference; forward at i == 0).
 *
 * `xy` is `[x, y]` in the TRACKING frame; the result is mapped to the feature
 * frame (vx negated, vy kept). Exposed for golden-vector testing.
 */
export function deriveVelocityFeature(
  prev: readonly [number, number],
  cur: readonly [number, number],
  dt: number,
): { vx: number; vy: number } {
  const vxTrack = (cur[0] - prev[0]) / dt;
  const vyTrack = (cur[1] - prev[1]) / dt;
  return { vx: -vxTrack, vy: vyTrack };
}

/** True when the displacement between two tracking points is a teleport glitch. */
function isTeleport(
  a: readonly [number, number],
  b: readonly [number, number],
): boolean {
  return Math.hypot(b[0] - a[0], b[1] - a[1]) > TELEPORT_FT;
}

/** Per-entity velocity at frame `i` for all 10 players, in the feature frame. */
function velocitiesAt(
  poss: TrackingPossession,
  i: number,
  dt: number,
): { vx: number; vy: number }[] {
  const cur = poss.frames[i].players;
  const ref = i > 0 ? poss.frames[i - 1].players : poss.frames[i].players;
  // At i == 0 prev == cur -> zero velocity (no prior sample to difference).
  // Across a teleport glitch -> zero velocity (the displacement is non-physical).
  return cur.map((c, k) =>
    isTeleport(ref[k], c)
      ? { vx: 0, vy: 0 }
      : deriveVelocityFeature(ref[k], c, dt),
  );
}

/**
 * Resolve the playback head at wall-time `time` (s) to a feature-frame
 * snapshot. Interpolates POSITIONS linearly between the two adjacent tracking
 * frames bracketing `time`; velocities are taken at the lower bracket frame
 * (Δpos/Δt is already a per-interval rate, so it is constant across the
 * interval). Times outside the span clamp to the first/last frame.
 */
export function snapshotAt(
  poss: TrackingPossession,
  time: number,
): TrackSnapshot {
  const frames = poss.frames;
  const n = frames.length;
  const dt = 1 / (poss.fps || 12.5);

  if (n === 0) {
    return {
      players: [],
      ball: { x: 0, y: 0 },
      ballZ: 0,
      shotClock: null,
      framePos: 0,
    };
  }
  if (n === 1) {
    return frameSnapshot(poss, 0, dt, 0);
  }

  const clampedT = clamp(time, frames[0].t, frames[n - 1].t);
  // Find the lower bracket frame (largest i with frames[i].t <= clampedT).
  // Linear scan is fine (a few hundred frames) and stays allocation-free.
  let lo = 0;
  for (let i = 1; i < n; i++) {
    if (frames[i].t <= clampedT) lo = i;
    else break;
  }
  const hi = Math.min(n - 1, lo + 1);
  const span = frames[hi].t - frames[lo].t;
  const f = span > 0 ? clamp((clampedT - frames[lo].t) / span, 0, 1) : 0;
  return interpolatedSnapshot(poss, lo, hi, f, dt);
}

/** Snapshot exactly on tracking-frame `i` (no interpolation). */
export function frameSnapshot(
  poss: TrackingPossession,
  i: number,
  dt: number,
  frameFractionBias = 0,
): TrackSnapshot {
  const fr = poss.frames[i];
  const vel = velocitiesAt(poss, i, dt);
  const players: TrackEntity[] = fr.players.map((p, k) => ({
    x: clamp(trackXToFeature(p[0]), 0, RENDER_X_MAX),
    y: clamp(p[1], 0, RENDER_Y_MAX),
    vx: vel[k].vx,
    vy: vel[k].vy,
  }));
  return {
    players,
    ball: {
      x: clamp(trackXToFeature(fr.ball[0]), 0, RENDER_X_MAX),
      y: clamp(fr.ball[1], 0, RENDER_Y_MAX),
    },
    ballZ: fr.ball[2],
    shotClock: fr.sc,
    framePos: i + frameFractionBias,
  };
}

/** Snapshot interpolated between adjacent frames `lo`..`hi` at fraction `f`. */
function interpolatedSnapshot(
  poss: TrackingPossession,
  lo: number,
  hi: number,
  f: number,
  dt: number,
): TrackSnapshot {
  const a = poss.frames[lo];
  const b = poss.frames[hi];
  // Velocity is computed at `hi` (so it reflects the motion INTO the displayed
  // interval and is already teleport-suppressed by velocitiesAt).
  const vel = velocitiesAt(poss, hi, dt);
  const players: TrackEntity[] = a.players.map((pa, k) => {
    const pb = b.players[k] ?? pa;
    // SNAP across a teleport glitch: hold `a` for the first half of the interval
    // then jump to `b` — no cross-court streak — rather than gliding through it.
    if (isTeleport(pa, pb)) {
      const p = f < 0.5 ? pa : pb;
      return {
        x: clamp(trackXToFeature(p[0]), 0, RENDER_X_MAX),
        y: clamp(p[1], 0, RENDER_Y_MAX),
        vx: vel[k].vx,
        vy: vel[k].vy,
      };
    }
    return {
      x: clamp(trackXToFeature(lerp(pa[0], pb[0], f)), 0, RENDER_X_MAX),
      y: clamp(lerp(pa[1], pb[1], f), 0, RENDER_Y_MAX),
      vx: vel[k].vx,
      vy: vel[k].vy,
    };
  });
  const ballA: [number, number] = [a.ball[0], a.ball[1]];
  const ballB: [number, number] = [b.ball[0], b.ball[1]];
  const ballSnap = isTeleport(ballA, ballB);
  const ballSrc = ballSnap ? (f < 0.5 ? a.ball : b.ball) : null;
  return {
    players,
    ball: ballSrc
      ? {
          x: clamp(trackXToFeature(ballSrc[0]), 0, RENDER_X_MAX),
          y: clamp(ballSrc[1], 0, RENDER_Y_MAX),
        }
      : {
          x: clamp(trackXToFeature(lerp(a.ball[0], b.ball[0], f)), 0, RENDER_X_MAX),
          y: clamp(lerp(a.ball[1], b.ball[1], f), 0, RENDER_Y_MAX),
        },
    ballZ: ballSrc ? ballSrc[2] : lerp(a.ball[2], b.ball[2], f),
    shotClock:
      a.sc != null && b.sc != null ? lerp(a.sc, b.sc, f) : (a.sc ?? b.sc),
    framePos: lo + f,
  };
}

// ---------------------------------------------------------------------------
// Live RawState from a tracking snapshot.
//
// Mirrors the existing live path (engine/explorer.ts::liveFrameToRawState):
// re-sort entities to canonical order (defenders closest-first to the BH,
// teammates dist-to-BH ascending), re-lookup each teammate's zone FG% at its
// interpolated spot, recompute the BH zone FG% + distance-to-basket, then
// rebuild context via deriveContext. Identity for the per-teammate FG lookup is
// the entity's network index (tm slot 1..4 == players[1..4]).
// ---------------------------------------------------------------------------

function distSq(p: { x: number; y: number }, q: { x: number; y: number }): number {
  const dx = p.x - q.x;
  const dy = p.y - q.y;
  return dx * dx + dy * dy;
}

/**
 * Build a parity-faithful `RawState` from a live tracking snapshot.
 *
 * @param snap        feature-frame snapshot (10 players in network order)
 * @param shotClock   shot clock to feed the network (carried from the snapshot
 *                    or a fallback when the tracking clock is missing)
 * @param bhZoneFgFor BH compact-id zone-FG lookup
 * @param tmZoneFgFor per-teammate (network slot 1..4) zone-FG lookups
 */
export function trackSnapshotToRawState(
  snap: TrackSnapshot,
  shotClock: number,
  bhZoneFgFor: (x: number, y: number) => number,
  tmZoneFgFor: ((x: number, y: number) => number)[],
): RawState {
  const bh = snap.players[0];
  const tmRaw = snap.players.slice(1, 5).map((p, i) => ({ ...p, srcIndex: i }));
  const dfRaw = snap.players.slice(5, 10).map((p) => ({ ...p }));

  // Re-sort defenders closest-first to the (interpolated) ball-handler.
  const defenders = [...dfRaw].sort((p, q) => distSq(p, bh) - distSq(q, bh));
  // Re-sort teammates by distance-to-BH ascending (teammate slot i == pass i).
  const teammates = [...tmRaw].sort((p, q) => distSq(p, bh) - distSq(q, bh));

  // Re-lookup zone FG% per teammate at its interpolated position, keyed by the
  // teammate's network identity (srcIndex) — NOT its post-sort slot.
  const tmWithFg = teammates.map((t) => ({
    x: t.x,
    y: t.y,
    zone_fg_pct: (tmZoneFgFor[t.srcIndex] ?? (() => 0))(t.x, t.y),
  }));

  const ctx = deriveContext(
    { x: bh.x, y: bh.y },
    tmWithFg,
    defenders.map((d) => ({ x: d.x, y: d.y })),
  );

  return {
    ball_handler: {
      compact_id: 0,
      x: bh.x,
      y: bh.y,
      vx: bh.vx,
      vy: bh.vy,
      distance_to_basket: distanceToBasket(bh.x, bh.y),
      zone_fg_pct: bhZoneFgFor(bh.x, bh.y),
    },
    teammates: teammates.map((t, i) => ({
      slot: i + 1,
      compact_id: 0,
      x: t.x,
      y: t.y,
      vx: t.vx,
      vy: t.vy,
    })),
    defenders: defenders.map((d, i) => ({
      slot: i + 1,
      x: d.x,
      y: d.y,
      vx: d.vx,
      vy: d.vy,
    })),
    shot_clock: shotClock,
    context: ctx,
  };
}

// ---------------------------------------------------------------------------
// Timeline helpers (wall-clock span + decision-snap mapping).
// ---------------------------------------------------------------------------

/** First/last wall-time (s) of a possession's tracking span. */
export function trackSpan(poss: TrackingPossession): { start: number; end: number } {
  const n = poss.frames.length;
  if (n === 0) return { start: 0, end: 0 };
  return { start: poss.frames[0].t, end: poss.frames[n - 1].t };
}

/** Wall-time (s) of a tracking frame index (clamped to the span). */
export function timeForTrackFrame(poss: TrackingPossession, idx: number): number {
  const n = poss.frames.length;
  if (n === 0) return 0;
  const i = clamp(Math.round(idx), 0, n - 1);
  return poss.frames[i].t;
}

/**
 * The tracking-frame index nearest a given wall-time (used to detect when the
 * head is sitting ON a recorded decision snap so the UI can hand off to the
 * parity-exact recorded path).
 */
export function nearestTrackFrame(poss: TrackingPossession, time: number): number {
  const frames = poss.frames;
  const n = frames.length;
  if (n === 0) return 0;
  let best = 0;
  let bestD = Infinity;
  for (let i = 0; i < n; i++) {
    const d = Math.abs(frames[i].t - time);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

// ===========================================================================
// FULL-COURT RENDER PATH (additive — does NOT touch the feature/agent path).
//
// The renderer consumes RAW SportVU coordinates (x in [0,94], BOTH baskets,
// real per-possession orientation) from tracking_*_fullcourt.json, while the
// live agent keeps consuming the half-court feature snapshot above, UNCHANGED.
// The two frames are related by `foldRawToFeature`, which is clip-free — and
// exactly equal to the half-court export (verified to rounding) — in the
// frontcourt; what-if drag uses it to map a dragged full-court position back
// into the agent's feature frame.
// ===========================================================================

/** Raw full-court bounds — must match Court.tsx full-court COURT_X_FULL/COURT_Y. */
const RAW_X_MAX = 94;
const RAW_Y_MAX = 50;
// x-offset of the right-attacking fold: feature_x = rawX - (94 - X_REFLECT).
const FOLD_RIGHT_DX = RAW_X_MAX - X_REFLECT; // 41.75

/**
 * Fold a RAW full-court position into the basket-right feature frame
 * ([0,47] x [0,50]), clamped. In the frontcourt this reproduces the half-court
 * export EXACTLY (verified, |delta| <= rounding); in the backcourt it clamps
 * like the export's clip. `attackingRight` is the possession's real orientation:
 *   true  -> offense attacks the RIGHT raw basket (88.75,25)
 *   false -> offense attacks the LEFT  raw basket (5.25,25)
 */
export function foldRawToFeature(
  x: number,
  y: number,
  attackingRight: boolean,
): { x: number; y: number } {
  const fx = attackingRight ? x - FOLD_RIGHT_DX : X_REFLECT - x;
  const fy = attackingRight ? RAW_Y_MAX - y : y;
  return { x: clamp(fx, 0, RENDER_X_MAX), y: clamp(fy, 0, RENDER_Y_MAX) };
}

/** A render-only snapshot in RAW full-court coordinates (no velocity, no fold). */
export interface RawTrackSnapshot {
  /** 10 entities [BH, tm1..4, def1..5], raw [x,y] in [0,94] x [0,50]. */
  players: { x: number; y: number }[];
  /** ball [x,y] raw. */
  ball: { x: number; y: number };
  ballZ: number;
  shotClock: number | null;
  framePos: number;
}

/**
 * Resolve the playback head at wall-time `time` (s) to a RAW full-court render
 * snapshot. Same wall-clock interpolation + teleport guard as `snapshotAt`, but
 * NO feature fold and NO velocity (render only). Positions clamp to the raw
 * court so out-of-bounds tracking never streaks off-canvas.
 */
export function rawSnapshotAt(
  poss: TrackingPossession,
  time: number,
): RawTrackSnapshot {
  const frames = poss.frames;
  const n = frames.length;
  if (n === 0) {
    return { players: [], ball: { x: 0, y: 0 }, ballZ: 0, shotClock: null, framePos: 0 };
  }
  if (n === 1) return rawFrameSnapshot(poss, 0, 0);

  const clampedT = clamp(time, frames[0].t, frames[n - 1].t);
  let lo = 0;
  for (let i = 1; i < n; i++) {
    if (frames[i].t <= clampedT) lo = i;
    else break;
  }
  const hi = Math.min(n - 1, lo + 1);
  const span = frames[hi].t - frames[lo].t;
  const f = span > 0 ? clamp((clampedT - frames[lo].t) / span, 0, 1) : 0;
  return rawInterpolatedSnapshot(poss, lo, hi, f);
}

function rawFrameSnapshot(
  poss: TrackingPossession,
  i: number,
  frameFractionBias: number,
): RawTrackSnapshot {
  const fr = poss.frames[i];
  return {
    players: fr.players.map((p) => ({
      x: clamp(p[0], 0, RAW_X_MAX),
      y: clamp(p[1], 0, RAW_Y_MAX),
    })),
    ball: { x: clamp(fr.ball[0], 0, RAW_X_MAX), y: clamp(fr.ball[1], 0, RAW_Y_MAX) },
    ballZ: fr.ball[2],
    shotClock: fr.sc,
    framePos: i + frameFractionBias,
  };
}

function rawInterpolatedSnapshot(
  poss: TrackingPossession,
  lo: number,
  hi: number,
  f: number,
): RawTrackSnapshot {
  const a = poss.frames[lo];
  const b = poss.frames[hi];
  const players = a.players.map((pa, k) => {
    const pb = b.players[k] ?? pa;
    // teleport glitch: hold `a` then jump to `b` rather than streak the court.
    const px = isTeleport(pa, pb) ? (f < 0.5 ? pa[0] : pb[0]) : lerp(pa[0], pb[0], f);
    const py = isTeleport(pa, pb) ? (f < 0.5 ? pa[1] : pb[1]) : lerp(pa[1], pb[1], f);
    return { x: clamp(px, 0, RAW_X_MAX), y: clamp(py, 0, RAW_Y_MAX) };
  });
  const ballA: [number, number] = [a.ball[0], a.ball[1]];
  const ballB: [number, number] = [b.ball[0], b.ball[1]];
  const ballSnap = isTeleport(ballA, ballB);
  const pick = (ai: number, bi: number, lerped: number) =>
    ballSnap ? (f < 0.5 ? ai : bi) : lerped;
  return {
    players,
    ball: {
      x: clamp(pick(a.ball[0], b.ball[0], lerp(a.ball[0], b.ball[0], f)), 0, RAW_X_MAX),
      y: clamp(pick(a.ball[1], b.ball[1], lerp(a.ball[1], b.ball[1], f)), 0, RAW_Y_MAX),
    },
    ballZ: pick(a.ball[2], b.ball[2], lerp(a.ball[2], b.ball[2], f)),
    shotClock: a.sc != null && b.sc != null ? lerp(a.sc, b.sc, f) : (a.sc ?? b.sc),
    framePos: lo + f,
  };
}
