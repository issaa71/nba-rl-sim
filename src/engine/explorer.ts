// App-facing glue over the (immutable) features + network engines.
//
// Responsibilities:
//   - build a `RawState` from an exported possession frame,
//   - run a live forward pass through either loaded model,
//   - rebuild context from dragged entity positions for what-if mode.
//
// Everything here delegates to engine/features.ts + engine/network.ts so the
// golden-vector parity guarantee carries through to the live UI.

import {
  BASKET_X,
  BASKET_Y,
  buildFeatures,
  deriveContext,
  distanceToBasket,
  type ContextInput,
  type RawState,
} from "./features";
import { forward, type LoadedNetwork } from "./network";
import type { Frame, FrameContext } from "../data/types";

export interface QResult {
  /** [shoot, pass_1, pass_2, pass_3, pass_4] */
  q: number[];
  /** argmax over q. */
  best: number;
}

function frameContextToInput(c: FrameContext): ContextInput {
  return {
    grid_zone: c.grid_zone,
    is_three_point_zone: c.is_three_point_zone,
    closest_defender_dist: c.closest_defender_dist,
    help_defender_dist: c.help_defender_dist,
    num_defenders_within_6ft: c.num_defenders_within_6ft,
    best_teammate_openness: c.best_teammate_openness,
    num_open_teammates: c.num_open_teammates,
    teammate_openness: c.teammate_openness,
    teammate_zone_fg: c.teammate_zone_fg,
    teammate_dist_to_basket: c.teammate_dist_to_basket,
    pass_lane_features: c.pass_lane_features,
  };
}

/**
 * Build the engine's RawState from a recorded frame, using the frame's stored
 * context verbatim (the golden path — reproduces the recorded Q-values).
 */
export function frameToRawState(frame: Frame): RawState {
  return {
    ball_handler: {
      compact_id: frame.ball_handler.compact_id,
      x: frame.ball_handler.x,
      y: frame.ball_handler.y,
      vx: frame.ball_handler.vx,
      vy: frame.ball_handler.vy,
      distance_to_basket: frame.ball_handler.distance_to_basket,
      zone_fg_pct: frame.ball_handler.zone_fg_pct,
    },
    teammates: frame.teammates.map((t) => ({
      slot: t.slot,
      compact_id: t.compact_id,
      x: t.x,
      y: t.y,
      vx: t.vx,
      vy: t.vy,
    })),
    defenders: frame.defenders.map((d) => ({
      slot: d.slot,
      x: d.x,
      y: d.y,
      vx: d.vx,
      vy: d.vy,
    })),
    shot_clock: frame.shot_clock,
    context: frameContextToInput(frame.context),
  };
}

/**
 * Positions for a what-if scenario (a draggable snapshot of one frame).
 * Velocities are recomputed by the caller from frame-to-frame deltas where
 * available; here we carry them so feature recompute stays faithful.
 */
export interface WhatIfState {
  ballHandler: { x: number; y: number; vx: number; vy: number };
  teammates: { x: number; y: number; vx: number; vy: number; zone_fg_pct: number }[];
  defenders: { x: number; y: number; vx: number; vy: number }[];
}

/**
 * Build a RawState from dragged positions, recomputing the geometric context
 * (distances, openness, pass lanes, grid zone, 3PT flag) via the engine's
 * `deriveContext`. distance_to_basket / zone_fg_pct for the BH are recomputed
 * too so the network sees a self-consistent state after a drag.
 */
export function whatIfToRawState(
  wi: WhatIfState,
  bhZoneFgFor: (x: number, y: number) => number,
): RawState {
  const ctx = deriveContext(
    { x: wi.ballHandler.x, y: wi.ballHandler.y },
    wi.teammates.map((t) => ({ x: t.x, y: t.y, zone_fg_pct: t.zone_fg_pct })),
    wi.defenders.map((d) => ({ x: d.x, y: d.y })),
  );
  const dx = wi.ballHandler.x - BASKET_X;
  const dy = wi.ballHandler.y - BASKET_Y;
  const distanceToBasket = Math.sqrt(dx * dx + dy * dy);
  return {
    ball_handler: {
      compact_id: 0,
      x: wi.ballHandler.x,
      y: wi.ballHandler.y,
      vx: wi.ballHandler.vx,
      vy: wi.ballHandler.vy,
      distance_to_basket: distanceToBasket,
      zone_fg_pct: bhZoneFgFor(wi.ballHandler.x, wi.ballHandler.y),
    },
    teammates: wi.teammates.map((t, i) => ({
      slot: i + 1,
      compact_id: 0,
      x: t.x,
      y: t.y,
      vx: t.vx,
      vy: t.vy,
    })),
    defenders: wi.defenders.map((d, i) => ({
      slot: i + 1,
      x: d.x,
      y: d.y,
      vx: d.vx,
      vy: d.vy,
    })),
    shot_clock: 0, // overwritten by caller (carries the real shot clock)
    context: ctx,
  };
}

/** Run a forward pass on a raw state + the possession's embedding ids. */
export function runState(
  net: LoadedNetwork,
  state: RawState,
  playerIds: number[],
): QResult {
  const feats = buildFeatures(state);
  const { q } = forward(net, feats, playerIds);
  const arr = Array.from(q);
  return { q: arr, best: argmax(arr) };
}

export function argmax(xs: number[]): number {
  let bi = 0;
  let bv = -Infinity;
  for (let i = 0; i < xs.length; i++) {
    if (xs[i] > bv) {
      bv = xs[i];
      bi = i;
    }
  }
  return bi;
}

/**
 * "What the player did" pseudo-model: surfaces the player's recorded action as
 * the recommendation. Q-values are not meaningful here; we return the live
 * Dueling Q for the bars (so the bars stay informative) but force `best` to the
 * player's action. The UI labels this mode clearly.
 */
export function playerChoice(
  duelingQ: number[],
  playerAction: number,
): QResult {
  return { q: duelingQ, best: playerAction };
}

// ---------------------------------------------------------------------------
// Live (continuous-playback) interpolation glue.
//
// Recorded frames are ~2 Hz samples. For smooth ~30 fps playback we linearly
// interpolate positions, velocities and the shot clock between two adjacent
// recorded frames, then run the FULL live path on the interpolated state:
//   interpolate -> re-sort to canonical order -> re-lookup zone FG%
//   -> deriveContext -> buildFeatures -> forward.
//
// Re-sorting + zone-FG re-lookup are MANDATORY before deriveContext (see
// features.ts::deriveContext — it consumes entities in canonical sorted order
// and treats teammate_zone_fg as a per-player table lookup, not geometry).
//
// At integer playback time `t` (i.e. landing exactly on a recorded frame) the
// caller should use the parity-exact recorded path instead (frameToRawState),
// so stored vs live Q-values never drift at sample points.
// ---------------------------------------------------------------------------

/** A position/velocity sample shared by both endpoints of an interpolation. */
interface PV {
  x: number;
  y: number;
  vx: number;
  vy: number;
}

/** One interpolated offensive/defensive entity, carrying identity for re-sort. */
export interface LiveEntity extends PV {
  /** stable index into the recorded teammate/defender array (for FG lookups). */
  srcIndex: number;
}

/** A fully interpolated frame snapshot, pre-sort. */
export interface LiveFrame {
  ballHandler: PV;
  teammates: LiveEntity[];
  defenders: LiveEntity[];
  shotClock: number;
}

const lerp = (a: number, b: number, f: number): number => a + (b - a) * f;

function lerpPV(a: PV, b: PV, f: number): PV {
  return {
    x: lerp(a.x, b.x, f),
    y: lerp(a.y, b.y, f),
    vx: lerp(a.vx, b.vx, f),
    vy: lerp(a.vy, b.vy, f),
  };
}

/**
 * Linearly interpolate every entity (and the shot clock) between two adjacent
 * recorded frames. `f` is the fractional position in [0, 1] from `a` to `b`.
 *
 * Teammates/defenders are matched by array index — both endpoints share the
 * recorded canonical ordering, so index i in frame `a` and frame `b` is the
 * same logical player for the span of one 2 Hz interval. (Identity for FG
 * lookups is taken from frame `a` via `srcIndex`.)
 */
export function interpolateFrame(a: Frame, b: Frame, f: number): LiveFrame {
  return {
    ballHandler: lerpPV(a.ball_handler, b.ball_handler, f),
    teammates: a.teammates.map((t, i) => ({
      ...lerpPV(t, b.teammates[i] ?? t, f),
      srcIndex: i,
    })),
    defenders: a.defenders.map((d, i) => ({
      ...lerpPV(d, b.defenders[i] ?? d, f),
      srcIndex: i,
    })),
    shotClock: lerp(a.shot_clock, b.shot_clock, f),
  };
}

/**
 * Build a parity-faithful `RawState` from an interpolated `LiveFrame`.
 *
 * Re-sorts entities to canonical order (defenders closest-first to the BH,
 * teammates dist-to-BH ascending — matching environment.py), re-looks-up each
 * teammate's zone FG% at its interpolated spot, recomputes the BH zone FG% +
 * distance-to-basket, then rebuilds context via the engine's `deriveContext`.
 *
 * @param live        interpolated snapshot
 * @param bhZoneFgFor BH compact-id zone-FG lookup
 * @param tmZoneFgFor per-recorded-teammate zone-FG lookups, indexed by srcIndex
 */
export function liveFrameToRawState(
  live: LiveFrame,
  bhZoneFgFor: (x: number, y: number) => number,
  tmZoneFgFor: ((x: number, y: number) => number)[],
): RawState {
  const bh = live.ballHandler;

  // Re-sort defenders closest-first to the (interpolated) ball-handler.
  const defenders = [...live.defenders].sort(
    (p, q) => distSq(p, bh) - distSq(q, bh),
  );
  // Re-sort teammates by distance-to-BH ascending (teammate slot i == pass i).
  const teammates = [...live.teammates].sort(
    (p, q) => distSq(p, bh) - distSq(q, bh),
  );

  // Re-lookup zone FG% per teammate at its interpolated position, keyed by the
  // teammate's recorded identity (srcIndex) — NOT its post-sort slot.
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
    shot_clock: live.shotClock,
    context: ctx,
  };
}

function distSq(p: { x: number; y: number }, q: { x: number; y: number }): number {
  const dx = p.x - q.x;
  const dy = p.y - q.y;
  return dx * dx + dy * dy;
}

// ---------------------------------------------------------------------------
// Dwell / glide / cut TIMELINE (honest playback).
//
// Recorded frames are DECISION POINTS at irregular real intervals — NOT uniform
// 2 Hz samples. Treating them as uniform (and linearly interpolating every gap)
// makes long real gaps play ~20x too fast and renders shot-clock RESETS (an
// offensive rebound, a new possession-clock) as smooth glides through the
// middle of the court. Both are dishonest.
//
// Instead we lay out, per possession, an explicit timeline of segments:
//   - DWELL  at frame i:  playT is held EXACTLY at the integer i for
//     DWELL_MS — this is the parity-exact recorded path. Decision moments are
//     held so they can be read.
//   - GLIDE  i -> i+1:    playT animates i..i+1 with a duration PROPORTIONAL to
//     the real time elapsed (shotClock[i] - shotClock[i+1]) but compressed —
//     bigger real gaps read as longer movements, without playing in real time.
//   - CUT    i -> i+1:    when the shot clock RESETS (delta <= 0), the real gap
//     is too large (> CUT_GAP_S), or a clock is missing, we do NOT interpolate
//     positions. We hold frame i, then a brief fade/dissolve reveals frame i+1.
//     The jump is marked as a "beat" so the discontinuity is honest.
//
// Playback is driven by elapsed wall-clock against this timeline; the speed
// control scales the whole timeline (durations are divided by the speed). The
// scrubber maps linearly to timeline time, and decision-frame markers map back
// to the start of that frame's dwell so clicking a marker snaps to it.
// ---------------------------------------------------------------------------

/** Frame held in place (parity-exact recorded path) for DWELL_MS at 1x. */
export const DWELL_MS = 850;
/** A real gap at or below this (seconds) glides; above it, we CUT. */
export const CUT_GAP_S = 8;
/** glideMs = clamp(realGap * GLIDE_MS_PER_S, GLIDE_MIN_MS, GLIDE_MAX_MS). */
export const GLIDE_MS_PER_S = 350;
export const GLIDE_MIN_MS = 300;
export const GLIDE_MAX_MS = 1300;
/** Length of the dissolve shown across a CUT (fade frame i -> i+1) at 1x. */
export const CUT_MS = 200;

export type TimelineSegmentKind = "dwell" | "glide" | "cut";

export interface TimelineSegment {
  kind: TimelineSegmentKind;
  /** recorded frame index this segment starts on. */
  from: number;
  /** glide/cut land on this frame; equals `from` for a dwell. */
  to: number;
  /** segment start in timeline ms at 1x (cumulative). */
  start: number;
  /** segment duration in timeline ms at 1x. */
  duration: number;
}

export interface Timeline {
  segments: TimelineSegment[];
  /** total timeline length in ms at 1x. */
  total: number;
  /** number of recorded frames the timeline spans (decisionFrame + 1). */
  nFrames: number;
}

const clamp = (v: number, lo: number, hi: number): number =>
  v < lo ? lo : v > hi ? hi : v;

/**
 * Map the real time elapsed between two adjacent frames to a glide duration:
 * proportional to the real gap, clamped so even tiny gaps animate visibly and
 * large ones stay sub-real-time. Returns `null` when the transition should be a
 * CUT instead — i.e. the shot clock RESET (delta <= 0), the gap exceeds
 * CUT_GAP_S, or either clock is missing/non-finite.
 */
export function glideDurationMs(
  shotClockFrom: number,
  shotClockTo: number,
): number | null {
  if (!Number.isFinite(shotClockFrom) || !Number.isFinite(shotClockTo)) {
    return null;
  }
  const realGap = shotClockFrom - shotClockTo;
  // delta <= 0 -> the clock went up or stalled: a reset / discontinuity -> CUT.
  if (realGap <= 0) return null;
  if (realGap > CUT_GAP_S) return null;
  return clamp(realGap * GLIDE_MS_PER_S, GLIDE_MIN_MS, GLIDE_MAX_MS);
}

/**
 * Build the dwell/glide/cut timeline for a possession from its per-frame shot
 * clocks (frames 0..decisionFrame inclusive — playback stops at the decision).
 *
 * Layout: dwell(0), [glide|cut](0->1), dwell(1), [glide|cut](1->2), ... ,
 * dwell(decisionFrame). A single-frame possession is just one dwell.
 *
 * Pure + deterministic — unit-tested in src/test/explorer.test.ts.
 */
export function buildTimeline(shotClocks: readonly number[]): Timeline {
  const n = shotClocks.length;
  const segments: TimelineSegment[] = [];
  let t = 0;
  const push = (s: Omit<TimelineSegment, "start">) => {
    segments.push({ ...s, start: t });
    t += s.duration;
  };

  if (n <= 1) {
    push({ kind: "dwell", from: 0, to: 0, duration: DWELL_MS });
    return { segments, total: t, nFrames: Math.max(1, n) };
  }

  for (let i = 0; i < n; i++) {
    push({ kind: "dwell", from: i, to: i, duration: DWELL_MS });
    if (i < n - 1) {
      const glide = glideDurationMs(shotClocks[i], shotClocks[i + 1]);
      if (glide == null) {
        push({ kind: "cut", from: i, to: i + 1, duration: CUT_MS });
      } else {
        push({ kind: "glide", from: i, to: i + 1, duration: glide });
      }
    }
  }
  return { segments, total: t, nFrames: n };
}

/** What the playback head resolves to at a given timeline time. */
export interface TimelineCursor {
  /** continuous frame position (integer during dwell/cut, fractional in glide). */
  playT: number;
  /** active segment kind. */
  kind: TimelineSegmentKind;
  /** the recorded frame to render position from (the held frame during a cut). */
  frame: number;
  /**
   * cut dissolve progress in [0,1] (0 at the held frame, 1 fully on the next
   * frame). 0 outside of a cut. The UI cross-fades position from `from` to `to`
   * WITHOUT interpolating through intermediate court positions.
   */
  cutProgress: number;
}

/**
 * Resolve a timeline time (ms at 1x) to a playback cursor. Times before 0 clamp
 * to the first frame; times at/after `total` clamp to the final dwell.
 *
 * During a GLIDE, `playT` animates `from`..`to` so positional interpolation is
 * driven exactly as before. During a CUT, `playT` is held at `from` (no
 * positional glide) while `cutProgress` advances 0..1 for the dissolve, then
 * snaps to `to` at the end. During a DWELL, `playT` sits on the integer frame.
 */
export function timelineCursor(tl: Timeline, time: number): TimelineCursor {
  const segs = tl.segments;
  if (segs.length === 0) {
    return { playT: 0, kind: "dwell", frame: 0, cutProgress: 0 };
  }
  if (time <= 0) {
    return { playT: 0, kind: "dwell", frame: 0, cutProgress: 0 };
  }
  if (time >= tl.total) {
    const last = segs[segs.length - 1];
    return { playT: last.to, kind: "dwell", frame: last.to, cutProgress: 0 };
  }
  // Linear scan (timelines are short — a few dozen segments at most).
  let seg = segs[0];
  for (const s of segs) {
    if (time >= s.start && time < s.start + s.duration) {
      seg = s;
      break;
    }
    seg = s;
  }
  const local = seg.duration > 0 ? (time - seg.start) / seg.duration : 0;
  if (seg.kind === "glide") {
    return {
      playT: seg.from + clamp(local, 0, 1),
      kind: "glide",
      frame: seg.from,
      cutProgress: 0,
    };
  }
  if (seg.kind === "cut") {
    return {
      playT: seg.from, // hold the source frame; no positional interpolation
      kind: "cut",
      frame: seg.from,
      cutProgress: clamp(local, 0, 1),
    };
  }
  return { playT: seg.from, kind: "dwell", frame: seg.from, cutProgress: 0 };
}

/**
 * Timeline time (ms at 1x) at which a recorded frame is first reached — the
 * start of that frame's dwell. Used to snap the scrubber / markers to a frame.
 */
export function timeForFrame(tl: Timeline, frame: number): number {
  for (const s of tl.segments) {
    if (s.kind === "dwell" && s.from === frame) return s.start;
  }
  // Fall back to the closest dwell at or before `frame`.
  let best = 0;
  for (const s of tl.segments) {
    if (s.kind === "dwell" && s.from <= frame) best = s.start;
  }
  return best;
}
