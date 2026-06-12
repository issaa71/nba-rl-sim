// Integration guard for the app's engine glue (engine/explorer.ts) and the
// what-if recompute path — the exact code the UI runs. Confirms the live path
// reproduces the stored Dueling Q-values and that what-if recompute yields
// finite Q-values for every action.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { describe, expect, it } from "vitest";

import {
  buildTimeline,
  CUT_GAP_S,
  CUT_MS,
  DWELL_MS,
  frameToRawState,
  glideDurationMs,
  GLIDE_MAX_MS,
  GLIDE_MIN_MS,
  GLIDE_MS_PER_S,
  interpolateFrame,
  liveFrameToRawState,
  runState,
  timeForFrame,
  timelineCursor,
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

describe("dwell/glide/cut timeline builder", () => {
  it("maps a real gap to a glide duration proportional to the gap (clamped)", () => {
    // A 2 s real gap -> 2 * GLIDE_MS_PER_S, comfortably inside the clamp.
    expect(glideDurationMs(20, 18)).toBe(2 * GLIDE_MS_PER_S);
    // Tiny gap clamps up to the floor; ordering is monotonic in the gap.
    expect(glideDurationMs(20, 19.9)).toBe(GLIDE_MIN_MS);
    expect(glideDurationMs(20, 18)).toBeGreaterThan(glideDurationMs(20, 19)!);
    // Just under the cut threshold still glides and clamps to the ceiling.
    expect(glideDurationMs(20, 20 - CUT_GAP_S + 0.01)).toBe(GLIDE_MAX_MS);
  });

  it("treats a shot-clock RESET (delta <= 0) as a CUT, not a glide", () => {
    // Offensive rebound style reset 10.3 -> 24.0.
    expect(glideDurationMs(10.3, 24.0)).toBeNull();
    // A stall (no change) is also a discontinuity.
    expect(glideDurationMs(12, 12)).toBeNull();
  });

  it("treats an over-long real gap (> CUT_GAP_S) as a CUT", () => {
    expect(glideDurationMs(20, 20 - CUT_GAP_S - 0.01)).toBeNull();
    // The exact possession from the bug report: 19.9 -> 10.3 is a 9.6 s gap.
    expect(glideDurationMs(19.9, 10.3)).toBeNull();
  });

  it("treats a missing / non-finite clock as a CUT", () => {
    expect(glideDurationMs(NaN, 10)).toBeNull();
    expect(glideDurationMs(10, NaN)).toBeNull();
    expect(glideDurationMs(Infinity, 10)).toBeNull();
  });

  it("builds dwell+glide+dwell for a clean proportional possession", () => {
    // 0021500485 prefix style: 2.4 s gaps glide.
    const tl = buildTimeline([15.5, 13.1, 10.7]);
    expect(tl.segments.map((s) => s.kind)).toEqual([
      "dwell",
      "glide",
      "dwell",
      "glide",
      "dwell",
    ]);
    // Each glide duration tracks its real gap.
    const glides = tl.segments.filter((s) => s.kind === "glide");
    expect(glides[0].duration).toBe(glideDurationMs(15.5, 13.1));
    expect(glides[1].duration).toBe(glideDurationMs(13.1, 10.7));
    // Segments are start-contiguous and total is their sum.
    let acc = 0;
    for (const s of tl.segments) {
      expect(s.start).toBe(acc);
      acc += s.duration;
    }
    expect(tl.total).toBe(acc);
  });

  it("inserts a CUT at a shot-clock reset (the 0021500207 bug case)", () => {
    // [19.9, 10.3, 24.0, 18.5]: 9.6 s gap -> CUT, 10.3->24.0 reset -> CUT,
    // 24.0->18.5 (5.5 s) -> glide.
    const tl = buildTimeline([19.9, 10.3, 24.0, 18.5]);
    expect(tl.segments.map((s) => s.kind)).toEqual([
      "dwell",
      "cut",
      "dwell",
      "cut",
      "dwell",
      "glide",
      "dwell",
    ]);
    const cuts = tl.segments.filter((s) => s.kind === "cut");
    expect(cuts.every((s) => s.duration === CUT_MS)).toBe(true);
    // The cut holds its source frame and lands on the next.
    expect(cuts[0].from).toBe(0);
    expect(cuts[0].to).toBe(1);
  });

  it("yields a single dwell for a 1-frame possession", () => {
    const tl = buildTimeline([15.0]);
    expect(tl.segments).toHaveLength(1);
    expect(tl.segments[0].kind).toBe("dwell");
    expect(tl.total).toBe(DWELL_MS);
    expect(tl.nFrames).toBe(1);
    // Empty input degrades to one dwell too (defensive).
    const empty = buildTimeline([]);
    expect(empty.segments).toHaveLength(1);
    expect(empty.total).toBe(DWELL_MS);
  });

  it("total duration is strictly monotonic as frames are appended", () => {
    const clocks = [20, 18, 17.5, 24, 22, 14];
    let prev = -1;
    for (let n = 1; n <= clocks.length; n++) {
      const tl = buildTimeline(clocks.slice(0, n));
      expect(tl.total).toBeGreaterThan(prev);
      prev = tl.total;
    }
  });

  it("the cursor holds integer frames on dwell/cut and animates on glide", () => {
    const tl = buildTimeline([20, 18]); // dwell, glide, dwell
    const [d0, glide, d1] = tl.segments;
    // mid-dwell: playT sits exactly on the integer frame.
    expect(timelineCursor(tl, d0.start + d0.duration / 2).playT).toBe(0);
    // mid-glide: playT is strictly between the two frames.
    const cur = timelineCursor(tl, glide.start + glide.duration / 2);
    expect(cur.kind).toBe("glide");
    expect(cur.playT).toBeGreaterThan(0);
    expect(cur.playT).toBeLessThan(1);
    // final dwell: playT lands on the last frame.
    expect(timelineCursor(tl, d1.start + 1).playT).toBe(1);
    // past the end clamps to the final frame.
    expect(timelineCursor(tl, tl.total + 999).playT).toBe(1);
  });

  it("cut cursor holds the source frame and advances dissolve progress", () => {
    const tl = buildTimeline([19.9, 10.3]); // dwell, cut, dwell (9.6 s gap)
    const cut = tl.segments.find((s) => s.kind === "cut")!;
    const cur = timelineCursor(tl, cut.start + cut.duration / 2);
    expect(cur.kind).toBe("cut");
    expect(cur.playT).toBe(0); // no positional interpolation through the cut
    expect(cur.cutProgress).toBeGreaterThan(0);
    expect(cur.cutProgress).toBeLessThan(1);
  });

  it("timeForFrame maps a frame back to the start of its dwell", () => {
    const tl = buildTimeline([20, 18, 16]);
    for (let f = 0; f < 3; f++) {
      const t = timeForFrame(tl, f);
      expect(timelineCursor(tl, t).frame).toBe(f);
      expect(timelineCursor(tl, t).playT).toBe(f);
    }
  });
});
