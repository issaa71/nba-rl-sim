// Golden-vector parity gate.
//
// THE BUILD IS WRONG UNTIL THESE PASS. Do NOT loosen the tolerances:
//   - features.ts output vs python_features:  |delta| <= 1e-6 per feature
//   - network.ts Q-values vs q_dueling/q_dqn: |dQ|    <= 1e-4 per action
//
// We read the 857 KB fixture from disk (Node env) so it never enters the app
// bundle or tsc's literal type-checking path.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { buildFeatures, type RawState, type ContextInput } from "../engine/features";
import {
  forward,
  loadNetwork,
  type LoadedNetwork,
  type WeightsFile,
} from "../engine/network";

// ---------------------------------------------------------------------------
// Fixture shapes
// ---------------------------------------------------------------------------

interface RawBallHandler {
  compact_id: number;
  nba_id: number;
  team: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  distance_to_basket: number;
  zone_fg_pct: number;
}
interface RawTeammate {
  slot: number;
  compact_id: number;
  nba_id: number;
  team: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
}
interface RawDefender {
  slot: number;
  team: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
}
interface RawContext {
  grid_zone: number;
  is_three_point_zone: number;
  closest_defender_dist: number;
  help_defender_dist: number;
  num_defenders_within_6ft: number;
  best_teammate_openness: number;
  num_open_teammates: number;
  teammate_openness: number[];
  teammate_zone_fg: number[];
  teammate_dist_to_basket: number[];
  pass_lane_features: [number, number, number][];
}
interface GoldenVector {
  row: number;
  id: string;
  grid_zone: number;
  is_three_point_zone: number;
  raw_inputs: {
    ball_handler: RawBallHandler;
    teammates: RawTeammate[];
    defenders: RawDefender[];
    shot_clock: number;
    game_clock: number | null;
    quarter: number | null;
    context: RawContext;
    player_actual_action: number;
  };
  python_features: number[];
  player_ids_compact: number[];
  q_dueling: number[];
  q_dqn: number[];
}
interface GoldenFile {
  note: string;
  feature_layout: string;
  n_vectors: number;
  vectors: GoldenVector[];
}

// ---------------------------------------------------------------------------
// Load fixtures from disk
// ---------------------------------------------------------------------------

const HERE = dirname(fileURLToPath(import.meta.url));
const readJson = <T,>(rel: string): T =>
  JSON.parse(readFileSync(resolve(HERE, rel), "utf8")) as T;

const golden = readJson<GoldenFile>("./fixtures/golden_vectors.json");
const dueling: LoadedNetwork = loadNetwork(
  readJson<WeightsFile>("../../public/data/model_weights.dueling.json"),
);
const dqn: LoadedNetwork = loadNetwork(
  readJson<WeightsFile>("../../public/data/model_weights.dqn.json"),
);

const FEATURE_TOL = 1e-6;
const Q_TOL = 1e-4;

function toRawState(gv: GoldenVector): RawState {
  const ri = gv.raw_inputs;
  const context: ContextInput = {
    grid_zone: ri.context.grid_zone,
    is_three_point_zone: ri.context.is_three_point_zone,
    closest_defender_dist: ri.context.closest_defender_dist,
    help_defender_dist: ri.context.help_defender_dist,
    num_defenders_within_6ft: ri.context.num_defenders_within_6ft,
    best_teammate_openness: ri.context.best_teammate_openness,
    num_open_teammates: ri.context.num_open_teammates,
    teammate_openness: ri.context.teammate_openness,
    teammate_zone_fg: ri.context.teammate_zone_fg,
    teammate_dist_to_basket: ri.context.teammate_dist_to_basket,
    pass_lane_features: ri.context.pass_lane_features,
  };
  return {
    ball_handler: {
      compact_id: ri.ball_handler.compact_id,
      x: ri.ball_handler.x,
      y: ri.ball_handler.y,
      vx: ri.ball_handler.vx,
      vy: ri.ball_handler.vy,
      distance_to_basket: ri.ball_handler.distance_to_basket,
      zone_fg_pct: ri.ball_handler.zone_fg_pct,
    },
    teammates: ri.teammates.map((t) => ({
      slot: t.slot,
      compact_id: t.compact_id,
      x: t.x,
      y: t.y,
      vx: t.vx,
      vy: t.vy,
    })),
    defenders: ri.defenders.map((d) => ({
      slot: d.slot,
      x: d.x,
      y: d.y,
      vx: d.vx,
      vy: d.vy,
    })),
    shot_clock: ri.shot_clock,
    context,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("golden-vector parity", () => {
  it(`has all ${golden.n_vectors} vectors`, () => {
    expect(golden.vectors.length).toBe(golden.n_vectors);
    expect(golden.vectors.length).toBe(250);
  });

  it("features.ts matches python_features (|delta| <= 1e-6)", () => {
    let worst = 0;
    let worstVec = -1;
    let worstIdx = -1;
    for (let v = 0; v < golden.vectors.length; v++) {
      const gv = golden.vectors[v];
      const feats = buildFeatures(toRawState(gv));
      expect(feats.length).toBe(73);
      expect(gv.python_features.length).toBe(73);
      for (let i = 0; i < 73; i++) {
        const d = Math.abs(feats[i] - gv.python_features[i]);
        if (d > worst) {
          worst = d;
          worstVec = v;
          worstIdx = i;
        }
      }
    }
    console.log(
      `[parity] worst feature delta = ${worst.toExponential(4)} ` +
        `(vector ${worstVec}, feature idx ${worstIdx})`,
    );
    expect(worst).toBeLessThanOrEqual(FEATURE_TOL);
  });

  it("network.ts dueling Q matches q_dueling (|dQ| <= 1e-4)", () => {
    let worst = 0;
    let worstVec = -1;
    let worstAct = -1;
    for (let v = 0; v < golden.vectors.length; v++) {
      const gv = golden.vectors[v];
      const { q } = forward(dueling, gv.python_features, gv.player_ids_compact);
      expect(q.length).toBe(5);
      for (let a = 0; a < 5; a++) {
        const d = Math.abs(q[a] - gv.q_dueling[a]);
        if (d > worst) {
          worst = d;
          worstVec = v;
          worstAct = a;
        }
      }
    }
    console.log(
      `[parity] worst dueling dQ = ${worst.toExponential(4)} ` +
        `(vector ${worstVec}, action ${worstAct})`,
    );
    expect(worst).toBeLessThanOrEqual(Q_TOL);
  });

  it("network.ts dqn Q matches q_dqn (|dQ| <= 1e-4)", () => {
    let worst = 0;
    let worstVec = -1;
    let worstAct = -1;
    for (let v = 0; v < golden.vectors.length; v++) {
      const gv = golden.vectors[v];
      const { q } = forward(dqn, gv.python_features, gv.player_ids_compact);
      expect(q.length).toBe(5);
      for (let a = 0; a < 5; a++) {
        const d = Math.abs(q[a] - gv.q_dqn[a]);
        if (d > worst) {
          worst = d;
          worstVec = v;
          worstAct = a;
        }
      }
    }
    console.log(
      `[parity] worst dqn dQ = ${worst.toExponential(4)} ` +
        `(vector ${worstVec}, action ${worstAct})`,
    );
    expect(worst).toBeLessThanOrEqual(Q_TOL);
  });
});
