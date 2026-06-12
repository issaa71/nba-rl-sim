// Client-side inference engine for the NBA RL agents.
//
// This is a pure-TypeScript, Float64 port of the per-entity Deep-Sets backbone
// shared by `q_network.py` (plain DQN) and `dueling_q_network.py` (dueling).
// It reproduces the Python Q-values to |dQ| <= 1e-4 on the 250 golden vectors.
//
// Authoritative spec: export_demo/EXPORT_README.md ("Network forward pass").
//
// LOAD-BEARING DETAILS (do not "simplify"):
//   - phi_defender / phi_teammate end on a TRAILING ReLU (output is post-ReLU).
//   - value_stream / *_advantage / *_head end on a BARE Linear (no trailing ReLU).
//   - Dueling recombination is TYPE-AWARE: pass advantages are mean-centered
//     WITHIN the 4-pass group only; shoot is a singleton (not normalized).
//   - PyTorch Linear weights are stored [out, in]; y = W @ x + b.

// ---------------------------------------------------------------------------
// Weights file schema (see EXPORT_README.md "model_weights.*.json schema")
// ---------------------------------------------------------------------------

export type NetworkKind = "dueling" | "dqn";

export interface WeightsFile {
  kind: NetworkKind;
  checkpoint: string;
  num_players: number; // 450
  embed_dim: number; // 8
  continuous_dim: number; // 73
  state_dim: number; // 78
  action_dim: number; // 5
  state_dict_keys: string[];
  shapes: Record<string, number[]>;
  /** Nested float arrays keyed by state_dict key. */
  tensors: Record<string, number[] | number[][]>;
}

// A 2-D weight matrix stored row-major as [out][in].
type Matrix = Float64Array[];
// A 1-D bias / embedding row.
type Vector = Float64Array;

/**
 * A loaded, ready-to-run network. The exact set of layers present depends on
 * `kind` (dueling has value/advantage streams; dqn has shoot/pass heads).
 */
export interface LoadedNetwork {
  kind: NetworkKind;
  checkpoint: string;
  numPlayers: number;
  embedDim: number;
  /** [450][8] player embedding table. */
  embedding: Matrix;
  /** Linear layers, keyed by their state_dict key, as { w, b }. */
  layers: Record<string, { w: Matrix; b: Vector }>;
}

/** Output of a single forward pass. Action 0 = shoot; 1-4 = pass to teammate slot 1-4. */
export interface ForwardResult {
  /** [shoot, pass_1, pass_2, pass_3, pass_4] */
  q: Float64Array;
  /** Dueling-only intermediates (undefined for the plain DQN). */
  dueling?: {
    value: number;
    aShoot: number;
    aPass: Float64Array; // raw (un-centered) pass advantages, length 4
  };
}

// ---------------------------------------------------------------------------
// Tensor helpers
// ---------------------------------------------------------------------------

function toMatrix(t: number[] | number[][]): Matrix {
  const rows = t as number[][];
  return rows.map((row) => Float64Array.from(row));
}

function toVector(t: number[] | number[][]): Vector {
  return Float64Array.from(t as number[]);
}

/** y = W @ x + b, where W is [out][in]. */
function linear(x: Vector, w: Matrix, b: Vector): Float64Array {
  const out = new Float64Array(w.length);
  for (let i = 0; i < w.length; i++) {
    const row = w[i];
    let acc = b[i];
    for (let j = 0; j < row.length; j++) acc += row[j] * x[j];
    out[i] = acc;
  }
  return out;
}

/** In-place ReLU. */
function relu(x: Float64Array): Float64Array {
  for (let i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0;
  return x;
}

function concat(parts: ArrayLike<number>[]): Float64Array {
  let n = 0;
  for (const p of parts) n += p.length;
  const out = new Float64Array(n);
  let o = 0;
  for (const p of parts) {
    for (let i = 0; i < p.length; i++) out[o++] = p[i];
  }
  return out;
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/**
 * Build a `LoadedNetwork` from a parsed weights JSON. Linear layers are detected
 * by the `<name>.<n>.weight` / `<name>.<n>.bias` convention, so this works for
 * either model without hard-coding head names.
 */
export function loadNetwork(file: WeightsFile): LoadedNetwork {
  const embedding = toMatrix(file.tensors["player_embedding.weight"]);

  const layers: Record<string, { w: Matrix; b: Vector }> = {};
  for (const key of file.state_dict_keys) {
    if (!key.endsWith(".weight")) continue;
    if (key === "player_embedding.weight") continue;
    const base = key.slice(0, -".weight".length); // e.g. "phi_defender.0"
    const biasKey = `${base}.bias`;
    if (file.tensors[biasKey] === undefined) {
      throw new Error(`Missing bias for layer ${base}`);
    }
    layers[base] = {
      w: toMatrix(file.tensors[key]),
      b: toVector(file.tensors[biasKey]),
    };
  }

  return {
    kind: file.kind,
    checkpoint: file.checkpoint,
    numPlayers: file.num_players,
    embedDim: file.embed_dim,
    embedding,
    layers,
  };
}

// ---------------------------------------------------------------------------
// State slicing (EXPORT_README "Network forward pass")
// ---------------------------------------------------------------------------

const SLICE = {
  context: [0, 11], // 11 game-context features
  tmOpen: [11, 15], // defender openness per teammate
  tmFg: [15, 19], // teammate zone FG%
  tmDist: [19, 23], // teammate dist-to-basket
  defPos: [23, 33], // 5 x (x, y)
  tmPos: [33, 41], // 4 x (x, y)
  bhVel: [41, 43], // ball-handler (vx, vy)
  defVel: [43, 53], // 5 x (vx, vy)
  tmVel: [53, 61], // 4 x (vx, vy)
  tmLaneMin: [61, 65], // pass-lane min defender dist
  tmCorridor: [65, 69], // pass-lane corridor defenders
  tmPassDist: [69, 73], // pass distance
} as const;

function slice(s: ArrayLike<number>, lo: number, hi: number): Float64Array {
  const out = new Float64Array(hi - lo);
  for (let i = lo; i < hi; i++) out[i - lo] = s[i];
  return out;
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

/**
 * Run the network on a state.
 *
 * @param net        loaded weights
 * @param features   73 normalized continuous features (indices 0-72)
 * @param playerIds  5 compact embedding indices [BH, tm1, tm2, tm3, tm4] (indices 73-77)
 */
export function forward(
  net: LoadedNetwork,
  features: ArrayLike<number>,
  playerIds: ArrayLike<number>,
): ForwardResult {
  const context = slice(features, SLICE.context[0], SLICE.context[1]); // 11
  const tmOpen = slice(features, SLICE.tmOpen[0], SLICE.tmOpen[1]); // 4
  const tmFg = slice(features, SLICE.tmFg[0], SLICE.tmFg[1]); // 4
  const tmDist = slice(features, SLICE.tmDist[0], SLICE.tmDist[1]); // 4
  const defPos = slice(features, SLICE.defPos[0], SLICE.defPos[1]); // 10
  const tmPos = slice(features, SLICE.tmPos[0], SLICE.tmPos[1]); // 8
  const bhVel = slice(features, SLICE.bhVel[0], SLICE.bhVel[1]); // 2
  const defVel = slice(features, SLICE.defVel[0], SLICE.defVel[1]); // 10
  const tmVel = slice(features, SLICE.tmVel[0], SLICE.tmVel[1]); // 8
  const tmLaneMin = slice(features, SLICE.tmLaneMin[0], SLICE.tmLaneMin[1]); // 4
  const tmCorridor = slice(features, SLICE.tmCorridor[0], SLICE.tmCorridor[1]); // 4
  const tmPassDist = slice(features, SLICE.tmPassDist[0], SLICE.tmPassDist[1]); // 4

  const L = net.layers;

  // --- defenders: shared phi_defender([x, y, vx, vy] -> 32, TRAILING ReLU), MEAN-pooled
  const defPooled = new Float64Array(32);
  for (let i = 0; i < 5; i++) {
    const di = Float64Array.of(
      defPos[i * 2],
      defPos[i * 2 + 1],
      defVel[i * 2],
      defVel[i * 2 + 1],
    );
    let h = relu(linear(di, L["phi_defender.0"].w, L["phi_defender.0"].b));
    h = relu(linear(h, L["phi_defender.2"].w, L["phi_defender.2"].b)); // trailing ReLU
    for (let k = 0; k < 32; k++) defPooled[k] += h[k];
  }
  for (let k = 0; k < 32; k++) defPooled[k] /= 5;

  // --- context vector = 11 game feats ++ bh_vel(2) ++ bh_embed(8) = 21-D
  const bhEmbed = net.embedding[playerIds[0]]; // (8,)
  const ctx = concat([context, bhVel, bhEmbed]); // 21
  const base = concat([ctx, defPooled]); // 53

  // --- teammate phi: per teammate i, shared phi_teammate(18 -> 32, TRAILING ReLU)
  const passIn: Float64Array[] = [];
  for (let i = 0; i < 4; i++) {
    const ti = concat([
      Float64Array.of(tmPos[i * 2], tmPos[i * 2 + 1]),
      Float64Array.of(tmVel[i * 2], tmVel[i * 2 + 1]),
      Float64Array.of(tmOpen[i]),
      Float64Array.of(tmFg[i]),
      Float64Array.of(tmDist[i]),
      Float64Array.of(tmLaneMin[i]),
      Float64Array.of(tmCorridor[i]),
      Float64Array.of(tmPassDist[i]),
      net.embedding[playerIds[i + 1]],
    ]); // 18
    let h = relu(linear(ti, L["phi_teammate.0"].w, L["phi_teammate.0"].b));
    h = relu(linear(h, L["phi_teammate.2"].w, L["phi_teammate.2"].b)); // trailing ReLU
    passIn.push(concat([h, ctx, defPooled])); // 85
  }

  if (net.kind === "dueling") {
    // V = value_stream(base): 53->64->32->1 (NO trailing ReLU)
    let vh = relu(linear(base, L["value_stream.0"].w, L["value_stream.0"].b));
    vh = relu(linear(vh, L["value_stream.2"].w, L["value_stream.2"].b));
    const value = linear(vh, L["value_stream.4"].w, L["value_stream.4"].b)[0];

    // A_shoot = shoot_advantage(base): 53->32->1 (NO trailing ReLU)
    const sh = relu(
      linear(base, L["shoot_advantage.0"].w, L["shoot_advantage.0"].b),
    );
    const aShoot = linear(
      sh,
      L["shoot_advantage.2"].w,
      L["shoot_advantage.2"].b,
    )[0];

    // A_pass_i = pass_advantage(pass_in_i) SHARED: 85->64->16->1 (NO trailing ReLU)
    const aPass = new Float64Array(4);
    for (let i = 0; i < 4; i++) {
      let ph = relu(
        linear(passIn[i], L["pass_advantage.0"].w, L["pass_advantage.0"].b),
      );
      ph = relu(linear(ph, L["pass_advantage.2"].w, L["pass_advantage.2"].b));
      aPass[i] = linear(
        ph,
        L["pass_advantage.4"].w,
        L["pass_advantage.4"].b,
      )[0];
    }

    // Type-aware recombination: mean-center pass advantages WITHIN the pass group.
    let meanA = 0;
    for (let i = 0; i < 4; i++) meanA += aPass[i];
    meanA /= 4;

    const q = new Float64Array(5);
    q[0] = value + aShoot;
    for (let i = 0; i < 4; i++) q[i + 1] = value + (aPass[i] - meanA);

    return { q, dueling: { value, aShoot, aPass } };
  }

  // --- plain DQN: heads emit Q directly
  // Q(shoot) = shoot_head([ctx, def_pooled]): 53->64->32->1 (NO trailing ReLU)
  let qh = relu(linear(base, L["shoot_head.0"].w, L["shoot_head.0"].b));
  qh = relu(linear(qh, L["shoot_head.2"].w, L["shoot_head.2"].b));
  const qShoot = linear(qh, L["shoot_head.4"].w, L["shoot_head.4"].b)[0];

  const q = new Float64Array(5);
  q[0] = qShoot;
  for (let i = 0; i < 4; i++) {
    // Q(pass_i) = pass_head(pass_in_i) SHARED: 85->64->16->1 (NO trailing ReLU)
    let ph = relu(linear(passIn[i], L["pass_head.0"].w, L["pass_head.0"].b));
    ph = relu(linear(ph, L["pass_head.2"].w, L["pass_head.2"].b));
    q[i + 1] = linear(ph, L["pass_head.4"].w, L["pass_head.4"].b)[0];
  }

  return { q };
}

// ---------------------------------------------------------------------------
// Runtime loading helper (browser: fetch from /data/)
// ---------------------------------------------------------------------------

/** Fetch + parse + load a weights file from a URL (e.g. `/data/model_weights.dueling.json`). */
export async function loadNetworkFromUrl(url: string): Promise<LoadedNetwork> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch weights: ${url} (${res.status})`);
  const file = (await res.json()) as WeightsFile;
  return loadNetwork(file);
}
