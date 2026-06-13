// Integration guard for the app's engine glue (engine/explorer.ts +
// engine/tracking.ts) — the exact code the UI runs. Confirms:
//   - the recorded path reproduces the stored Dueling Q-values,
//   - what-if recompute yields finite Q for every action,
//   - the real-time tracking path (velocity derivation, snapshot, live state)
//     is well-formed AND that AT a decision snap the parity-exact recorded
//     path reproduces the stored agent_q_values for all 40 curated possessions.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { describe, expect, it } from "vitest";

import {
  frameToRawState,
  runState,
  whatIfToRawState,
  type WhatIfState,
} from "../engine/explorer";
import {
  deriveVelocityFeature,
  nearestTrackFrame,
  snapshotAt,
  timeForTrackFrame,
  trackSnapshotToRawState,
  trackSpan,
  trackXToFeature,
  TRACK_BASKET_X,
} from "../engine/tracking";
import { loadNetwork, type LoadedNetwork, type WeightsFile } from "../engine/network";
import { indexTracking, makeZoneFgLookup, type ZoneFgTable } from "../data/load";
import { BASKET_X } from "../engine/features";
import type { PossessionsFile, TrackingFile } from "../data/types";

const HERE = dirname(fileURLToPath(import.meta.url));
const readJson = <T,>(rel: string): T =>
  JSON.parse(readFileSync(resolve(HERE, rel), "utf8")) as T;

const poss = readJson<PossessionsFile>("../../public/data/possessions.json");
const trackingFile = readJson<TrackingFile>(
  "../../public/data/tracking_curated.json",
);
const tracking = indexTracking(trackingFile);
const dueling: LoadedNetwork = loadNetwork(
  readJson<WeightsFile>("../../public/data/model_weights.dueling.json"),
);
const dqn: LoadedNetwork = loadNetwork(
  readJson<WeightsFile>("../../public/data/model_weights.dqn.json"),
);

interface ZoneFgFileShape {
  special_entries: { __league_avg__: Record<string, number> };
  by_compact_id: Record<string, Record<string, number>>;
}
const zfFile = readJson<ZoneFgFileShape>("../../public/data/zone_fg.json");
const zoneFg: ZoneFgTable = {
  byCompactId: zfFile.by_compact_id,
  leagueAvg: zfFile.special_entries.__league_avg__,
};

describe("explorer engine glue", () => {
  it("live Dueling path reproduces the stored decision-frame Q-values (|dQ| <= 1e-4)", () => {
    let worst = 0;
    for (const p of poss.possessions) {
      const frame = p.frames[p.decision_frame];
      const state = frameToRawState(frame);
      const { q } = runState(dueling, state, p.entity_ids_network_order);
      for (let a = 0; a < 5; a++) {
        worst = Math.max(worst, Math.abs(q[a] - p.agent_q_values[a]));
      }
    }
    expect(worst).toBeLessThanOrEqual(1e-4);
  });

  it("live argmax matches the stored agent_action for every possession", () => {
    for (const p of poss.possessions) {
      const state = frameToRawState(p.frames[p.decision_frame]);
      const { best } = runState(dueling, state, p.entity_ids_network_order);
      expect(best).toBe(p.agent_action);
    }
  });

  it("what-if recompute yields 5 finite Q-values after a drag (both models)", () => {
    const p = poss.possessions[0];
    const f = p.frames[p.decision_frame];
    const bhFg = makeZoneFgLookup(zoneFg, f.ball_handler.compact_id);
    const tmFg = f.teammates.map((t) => makeZoneFgLookup(zoneFg, t.compact_id));

    const wi: WhatIfState = {
      ballHandler: {
        x: f.ball_handler.x + 4,
        y: f.ball_handler.y - 3,
        vx: 0,
        vy: 0,
      },
      teammates: f.teammates.map((t, i) => ({
        x: t.x,
        y: t.y,
        vx: t.vx,
        vy: t.vy,
        zone_fg_pct: tmFg[i](t.x, t.y),
      })),
      defenders: f.defenders.map((d) => ({
        x: d.x,
        y: d.y,
        vx: d.vx,
        vy: d.vy,
      })),
    };
    const state = whatIfToRawState(wi, bhFg);
    state.shot_clock = f.shot_clock;

    for (const net of [dueling, dqn]) {
      const { q, best } = runState(net, state, p.entity_ids_network_order);
      expect(q.length).toBe(5);
      expect(q.every(Number.isFinite)).toBe(true);
      expect(best).toBeGreaterThanOrEqual(0);
      expect(best).toBeLessThan(5);
    }
  });
});

// ---------------------------------------------------------------------------
// Real-time tracking playback
// ---------------------------------------------------------------------------

describe("tracking coordinate + velocity convention", () => {
  it("trackXToFeature reflects x about the basket axis (distance-preserving)", () => {
    // The basket maps basket->basket: tracking basket x -> feature basket x.
    expect(trackXToFeature(TRACK_BASKET_X)).toBeCloseTo(BASKET_X, 10);
    // The reflection is an involution: applying it twice is identity.
    for (const x of [0, 5.25, 10, 23.5, 47]) {
      expect(trackXToFeature(trackXToFeature(x))).toBeCloseTo(x, 10);
    }
  });

  it("velocity derivation matches the env convention v = dpos/dt (mapped to feature frame)", () => {
    // Golden synthetic vectors: dt, a tracking prev/cur, and the EXACT expected
    // feature-frame velocity (vx negated under the x-reflection, vy preserved).
    const dt = 1 / 12.5; // 0.08 s
    const cases: {
      prev: [number, number];
      cur: [number, number];
      vx: number;
      vy: number;
    }[] = [
      // pure +x track motion -> -x feature velocity
      { prev: [10, 25], cur: [10 + dt * 5, 25], vx: -5, vy: 0 },
      // pure +y track motion -> +y feature velocity (y preserved)
      { prev: [10, 25], cur: [10, 25 + dt * 8], vx: 0, vy: 8 },
      // diagonal
      { prev: [20, 20], cur: [20 - dt * 3, 20 + dt * 4], vx: 3, vy: 4 },
      // no motion
      { prev: [30, 15], cur: [30, 15], vx: 0, vy: 0 },
    ];
    for (const c of cases) {
      const v = deriveVelocityFeature(c.prev, c.cur, dt);
      expect(v.vx).toBeCloseTo(c.vx, 9);
      expect(v.vy).toBeCloseTo(c.vy, 9);
    }
  });

  it("the curated tracking has all 40 possessions", () => {
    expect(tracking.size).toBe(40);
  });
});

describe("real-time snapshot + live state", () => {
  const sample = poss.possessions.find((p) => tracking.has(p.id))!;
  const trk = tracking.get(sample.id)!;

  it("snapshotAt returns 10 players each frame, clamped to the court", () => {
    const span = trackSpan(trk);
    for (const f of [0, 0.25, 0.5, 0.75, 1]) {
      const t = span.start + (span.end - span.start) * f;
      const snap = snapshotAt(trk, t);
      expect(snap.players.length).toBe(10);
      for (const pl of snap.players) {
        expect(pl.x).toBeGreaterThanOrEqual(0);
        expect(pl.x).toBeLessThanOrEqual(47);
        expect(pl.y).toBeGreaterThanOrEqual(0);
        expect(pl.y).toBeLessThanOrEqual(50);
        expect(Number.isFinite(pl.vx)).toBe(true);
        expect(Number.isFinite(pl.vy)).toBe(true);
      }
    }
  });

  it("snapshotAt clamps times outside the span to the endpoints", () => {
    const span = trackSpan(trk);
    const before = snapshotAt(trk, span.start - 100);
    const after = snapshotAt(trk, span.end + 100);
    expect(before.framePos).toBe(0);
    expect(after.framePos).toBeCloseTo(trk.frames.length - 1, 6);
  });

  it("live tracking state yields 5 finite Q-values across the span (both models)", () => {
    const span = trackSpan(trk);
    const bhFg = makeZoneFgLookup(
      zoneFg,
      sample.frames[sample.decision_frame].ball_handler.compact_id,
    );
    const tmFg = sample.frames[sample.decision_frame].teammates.map((t) =>
      makeZoneFgLookup(zoneFg, t.compact_id),
    );
    for (const f of [0, 0.33, 0.66, 1]) {
      const t = span.start + (span.end - span.start) * f;
      const snap = snapshotAt(trk, t);
      const state = trackSnapshotToRawState(
        snap,
        snap.shotClock ?? 14,
        bhFg,
        tmFg,
      );
      for (const net of [dueling, dqn]) {
        const { q, best } = runState(net, state, sample.entity_ids_network_order);
        expect(q.length).toBe(5);
        expect(q.every(Number.isFinite)).toBe(true);
        expect(best).toBeGreaterThanOrEqual(0);
        expect(best).toBeLessThan(5);
      }
    }
  });
});

describe("snap-index parity (live Q at a snap === recorded agent_q_values)", () => {
  it("the playhead at every decision snap lands ON that tracking frame", () => {
    let checked = 0;
    for (const p of poss.possessions) {
      const trk = tracking.get(p.id);
      if (!trk) continue;
      const snaps = trk.decision_snap_indices;
      // The decision-frame snap must round-trip: its frame time maps back to it.
      for (let k = 0; k < snaps.length; k++) {
        const t = timeForTrackFrame(trk, snaps[k]);
        expect(nearestTrackFrame(trk, t)).toBe(snaps[k]);
      }
      checked++;
    }
    expect(checked).toBeGreaterThanOrEqual(40);
  });

  it("recorded path at the decision snap reproduces agent_q_values for all 40 curated", () => {
    let checked = 0;
    let worst = 0;
    for (const p of poss.possessions) {
      const trk = tracking.get(p.id);
      if (!trk) continue;
      // At the decision snap the UI uses frameToRawState(p.frames[decision]).
      const state = frameToRawState(p.frames[p.decision_frame]);
      const { q, best } = runState(dueling, state, p.entity_ids_network_order);
      for (let a = 0; a < 5; a++) {
        worst = Math.max(worst, Math.abs(q[a] - p.agent_q_values[a]));
      }
      expect(best).toBe(p.agent_action);
      checked++;
    }
    expect(checked).toBeGreaterThanOrEqual(40);
    expect(worst).toBeLessThanOrEqual(1e-4);
  });
});
