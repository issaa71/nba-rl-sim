// Integration guard for the app's engine glue (engine/explorer.ts) and the
// what-if recompute path — the exact code the UI runs. Confirms the live path
// reproduces the stored Dueling Q-values and that what-if recompute yields
// finite Q-values for every action.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { describe, expect, it } from "vitest";

import {
  frameToRawState,
  interpolateFrame,
  liveFrameToRawState,
  runState,
  whatIfToRawState,
  type WhatIfState,
} from "../engine/explorer";
import { loadNetwork, type LoadedNetwork, type WeightsFile } from "../engine/network";
import { makeZoneFgLookup, type ZoneFgTable } from "../data/load";
import type { PossessionsFile } from "../data/types";

const HERE = dirname(fileURLToPath(import.meta.url));
const readJson = <T,>(rel: string): T =>
  JSON.parse(readFileSync(resolve(HERE, rel), "utf8")) as T;

const poss = readJson<PossessionsFile>("../../public/data/possessions.json");
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

describe("live interpolation glue (continuous playback)", () => {
  // A possession that has at least two recorded frames to interpolate across.
  const p = poss.possessions.find((q) => q.frames.length >= 2)!;

  const lookupsFor = (frameIdx: number) => {
    const f = p.frames[frameIdx];
    return {
      bhFg: makeZoneFgLookup(zoneFg, f.ball_handler.compact_id),
      tmFg: f.teammates.map((t) => makeZoneFgLookup(zoneFg, t.compact_id)),
    };
  };

  it("at f=0 the live path tracks the recorded frame (positions + finite Q)", () => {
    const lo = 0;
    const a = p.frames[lo];
    const b = p.frames[lo + 1];
    const live = interpolateFrame(a, b, 0);
    // f=0 must reproduce frame `a`'s ball-handler position exactly.
    expect(live.ballHandler.x).toBeCloseTo(a.ball_handler.x, 10);
    expect(live.ballHandler.y).toBeCloseTo(a.ball_handler.y, 10);
    expect(live.shotClock).toBeCloseTo(a.shot_clock, 10);

    const { bhFg, tmFg } = lookupsFor(lo);
    const state = liveFrameToRawState(live, bhFg, tmFg);
    const { q, best } = runState(dueling, state, p.entity_ids_network_order);
    expect(q.every(Number.isFinite)).toBe(true);
    expect(best).toBeGreaterThanOrEqual(0);
    expect(best).toBeLessThan(5);
  });

  it("midpoint interpolation sits between the two endpoints", () => {
    const a = p.frames[0];
    const b = p.frames[1];
    const mid = interpolateFrame(a, b, 0.5);
    const lo = Math.min(a.ball_handler.x, b.ball_handler.x);
    const hi = Math.max(a.ball_handler.x, b.ball_handler.x);
    expect(mid.ballHandler.x).toBeGreaterThanOrEqual(lo - 1e-9);
    expect(mid.ballHandler.x).toBeLessThanOrEqual(hi + 1e-9);
    expect(mid.ballHandler.x).toBeCloseTo(
      (a.ball_handler.x + b.ball_handler.x) / 2,
      10,
    );
  });

  it("yields 5 finite Q-values across the whole interpolated span (both models)", () => {
    const { bhFg, tmFg } = lookupsFor(0);
    for (const f of [0, 0.25, 0.5, 0.75, 1]) {
      const live = interpolateFrame(p.frames[0], p.frames[1], f);
      const state = liveFrameToRawState(live, bhFg, tmFg);
      for (const net of [dueling, dqn]) {
        const { q, best } = runState(net, state, p.entity_ids_network_order);
        expect(q.length).toBe(5);
        expect(q.every(Number.isFinite)).toBe(true);
        expect(best).toBeGreaterThanOrEqual(0);
        expect(best).toBeLessThan(5);
      }
    }
  });
});
