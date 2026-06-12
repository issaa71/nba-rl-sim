// 73-feature state builder — the TypeScript port of `environment.py::_build_state`.
//
// Authoritative spec: export_demo/EXPORT_README.md ("Feature-vector layout").
//
// Half-court frame: x in [0, 47] (basket at x = 47), y in [0, 50].
// Defenders are sorted closest-first by distance to the ball-handler; teammates
// are sorted by distance to the ball-handler (teammate slot i == pass action i).
//
// Normalization: norm = (raw - low) / (high - low). **NO CLIPPING** — velocity
// features (raw range +/-20 ft/s) occasionally land slightly outside [0, 1] and
// must be left as-is, or Q-values drift.

// ---------------------------------------------------------------------------
// Normalization constants (features 0-72), verbatim from EXPORT_README.md.
// Player ids (73-77) bypass normalization entirely and are NOT part of this vector.
// ---------------------------------------------------------------------------

// prettier-ignore
export const LOW: readonly number[] = [
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

// prettier-ignore
export const HIGH: readonly number[] = [
  49, 50, 50, 50, 5, 50, 4, 24, 1, 1, 1, 50, 50, 50, 50, 1, 1, 1, 1, 50, 50, 50, 50,
  47, 50, 47, 50, 47, 50, 47, 50, 47, 50,
  47, 50, 47, 50, 47, 50, 47, 50,
  20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
  47, 47, 47, 47, 5, 5, 5, 5, 47, 47, 47, 47,
];

export const NUM_FEATURES = 73;

// Court geometry (matches environment.py).
export const BASKET_X = 47;
export const BASKET_Y = 25;
const COURT_LEN = 47; // x span used for the 10-column grid
const COURT_WID = 50; // y span used for the 5-row grid
const GRID_COLS = 10;
const GRID_ROWS = 5;
const THREE_POINT_DIST = 22; // "beyond ~22 ft" (EXPORT_README idx 9)
const OPEN_THRESHOLD = 6; // defender > 6 ft == "open" / "contested" boundary
const CONTEST_RADIUS = 6; // NBA "contested" radius for num_defenders_within_6ft

// ---------------------------------------------------------------------------
// Raw entity input shapes (the export's `raw_inputs` schema)
// ---------------------------------------------------------------------------

export interface BallHandlerInput {
  compact_id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  distance_to_basket: number;
  zone_fg_pct: number;
}

export interface TeammateInput {
  slot: number;
  compact_id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
}

export interface DefenderInput {
  slot: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
}

/**
 * Precomputed scalar context, as stored alongside the entities in the export.
 * For golden-vector parity these come straight from the pickle; for what-if
 * dragging they can be recomputed via {@link deriveContext}.
 */
export interface ContextInput {
  grid_zone: number;
  is_three_point_zone: number;
  closest_defender_dist: number;
  help_defender_dist: number;
  num_defenders_within_6ft: number;
  best_teammate_openness: number;
  num_open_teammates: number;
  /** 4 entries, teammate-slot order. */
  teammate_openness: number[];
  teammate_zone_fg: number[];
  teammate_dist_to_basket: number[];
  /** 4 entries: [min_def_dist, corridor_defs, pass_distance] per teammate. */
  pass_lane_features: [number, number, number][];
}

/** Everything the feature builder consumes (the export's `raw_inputs`, minus render-only fields). */
export interface RawState {
  ball_handler: BallHandlerInput;
  teammates: TeammateInput[]; // length 4, already dist-to-BH sorted
  defenders: DefenderInput[]; // length 5, already closest-first sorted
  shot_clock: number;
  context: ContextInput;
}

// ---------------------------------------------------------------------------
// Feature builder
// ---------------------------------------------------------------------------

/**
 * Build the 73 normalized continuous features from a raw state.
 *
 * Mirrors `environment.py::_build_state`: assembles the raw 73-vector in the
 * exact documented order, then applies `(raw - low) / (high - low)` with NO clip.
 */
export function buildFeatures(state: RawState): Float64Array {
  const raw = buildRawFeatures(state);
  return normalize(raw);
}

/** The un-normalized 73-vector, in EXPORT_README index order. */
export function buildRawFeatures(state: RawState): Float64Array {
  const { ball_handler: bh, teammates: tm, defenders: df, context: ctx } = state;
  const plf = ctx.pass_lane_features;
  const raw = new Float64Array(NUM_FEATURES);

  // 0-10 game context
  raw[0] = ctx.grid_zone;
  raw[1] = bh.distance_to_basket;
  raw[2] = ctx.closest_defender_dist;
  raw[3] = ctx.help_defender_dist;
  raw[4] = ctx.num_defenders_within_6ft;
  raw[5] = ctx.best_teammate_openness;
  raw[6] = ctx.num_open_teammates;
  raw[7] = state.shot_clock;
  raw[8] = Math.max(0, (7 - state.shot_clock) / 7); // shot_clock_urgency (computed pre-norm)
  raw[9] = ctx.is_three_point_zone;
  raw[10] = bh.zone_fg_pct;

  // 11-14 teammate openness, 15-18 teammate zone FG%, 19-22 teammate dist-to-basket
  for (let i = 0; i < 4; i++) raw[11 + i] = ctx.teammate_openness[i];
  for (let i = 0; i < 4; i++) raw[15 + i] = ctx.teammate_zone_fg[i];
  for (let i = 0; i < 4; i++) raw[19 + i] = ctx.teammate_dist_to_basket[i];

  // 23-32 defender positions (5 x x,y), closest-first
  for (let i = 0; i < 5; i++) {
    raw[23 + i * 2] = df[i].x;
    raw[24 + i * 2] = df[i].y;
  }
  // 33-40 teammate positions (4 x x,y), dist-to-BH sorted
  for (let i = 0; i < 4; i++) {
    raw[33 + i * 2] = tm[i].x;
    raw[34 + i * 2] = tm[i].y;
  }

  // 41-42 ball-handler velocity
  raw[41] = bh.vx;
  raw[42] = bh.vy;
  // 43-52 defender velocities
  for (let i = 0; i < 5; i++) {
    raw[43 + i * 2] = df[i].vx;
    raw[44 + i * 2] = df[i].vy;
  }
  // 53-60 teammate velocities
  for (let i = 0; i < 4; i++) {
    raw[53 + i * 2] = tm[i].vx;
    raw[54 + i * 2] = tm[i].vy;
  }

  // 61-72 pass-lane features per teammate: [min_def_dist, corridor_defs, pass_distance]
  for (let i = 0; i < 4; i++) raw[61 + i] = plf[i][0];
  for (let i = 0; i < 4; i++) raw[65 + i] = plf[i][1];
  for (let i = 0; i < 4; i++) raw[69 + i] = plf[i][2];

  return raw;
}

/** Apply `(raw - low) / (high - low)` with NO clipping. */
export function normalize(raw: ArrayLike<number>): Float64Array {
  const out = new Float64Array(NUM_FEATURES);
  for (let i = 0; i < NUM_FEATURES; i++) {
    out[i] = (raw[i] - LOW[i]) / (HIGH[i] - LOW[i]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Geometry helpers — used by what-if dragging to RECOMPUTE the context scalars
// from dragged (x, y) positions. The golden-vector parity path consumes the
// export's precomputed `context` directly; these mirror environment.py so the
// app can rebuild context after a drag.
// ---------------------------------------------------------------------------

function dist(ax: number, ay: number, bx: number, by: number): number {
  const dx = ax - bx;
  const dy = ay - by;
  return Math.sqrt(dx * dx + dy * dy);
}

/** half-court 10x5 grid index (0-49). Column from x, row from y. */
export function gridZone(x: number, y: number): number {
  let col = Math.floor((x / COURT_LEN) * GRID_COLS);
  let row = Math.floor((y / COURT_WID) * GRID_ROWS);
  if (col < 0) col = 0;
  else if (col >= GRID_COLS) col = GRID_COLS - 1;
  if (row < 0) row = 0;
  else if (row >= GRID_ROWS) row = GRID_ROWS - 1;
  return row * GRID_COLS + col;
}

export function distanceToBasket(x: number, y: number): number {
  return dist(x, y, BASKET_X, BASKET_Y);
}

export function isThreePointZone(x: number, y: number): number {
  return distanceToBasket(x, y) >= THREE_POINT_DIST ? 1 : 0;
}

/** Minimum perpendicular distance from any defender to the BH->teammate pass segment. */
function laneClearance(
  bhX: number,
  bhY: number,
  tmX: number,
  tmY: number,
  defenders: { x: number; y: number }[],
): { minDist: number; corridor: number } {
  const segDx = tmX - bhX;
  const segDy = tmY - bhY;
  const segLen2 = segDx * segDx + segDy * segDy;
  let minDist = Infinity;
  let corridor = 0;
  for (const d of defenders) {
    let pd: number;
    if (segLen2 === 0) {
      pd = dist(d.x, d.y, bhX, bhY);
    } else {
      let t = ((d.x - bhX) * segDx + (d.y - bhY) * segDy) / segLen2;
      if (t < 0) t = 0;
      else if (t > 1) t = 1;
      const px = bhX + t * segDx;
      const py = bhY + t * segDy;
      pd = dist(d.x, d.y, px, py);
    }
    if (pd < minDist) minDist = pd;
    if (pd < OPEN_THRESHOLD) corridor++;
  }
  if (!isFinite(minDist)) minDist = 0;
  return { minDist, corridor };
}

/**
 * Recompute the scalar context from raw entity positions, for what-if dragging.
 *
 * Entities must already be in their canonical sorted order (defenders
 * closest-first to the BH, teammates dist-to-BH sorted) — this matches how the
 * environment consumes them. `teammate_zone_fg` is carried through unchanged
 * (FG% is a per-player table lookup, not a geometric quantity).
 */
export function deriveContext(
  bh: { x: number; y: number },
  teammates: { x: number; y: number; zone_fg_pct: number }[],
  defenders: { x: number; y: number }[],
): ContextInput {
  // Defender distances to the ball-handler (sorted order is the caller's contract).
  const defDistToBh = defenders.map((d) => dist(d.x, d.y, bh.x, bh.y));
  const sortedDefDist = [...defDistToBh].sort((a, b) => a - b);
  const closest = sortedDefDist[0] ?? 50;
  const help = sortedDefDist[1] ?? 50;
  let within6 = 0;
  for (const dd of defDistToBh) if (dd < CONTEST_RADIUS) within6++;

  // Per-teammate openness = distance to that teammate's nearest defender.
  const teammate_openness = teammates.map((t) => {
    let best = Infinity;
    for (const d of defenders) {
      const dd = dist(d.x, d.y, t.x, t.y);
      if (dd < best) best = dd;
    }
    return isFinite(best) ? best : 50;
  });
  let bestOpen = 0;
  let numOpen = 0;
  for (const o of teammate_openness) {
    if (o > bestOpen) bestOpen = o;
    if (o > OPEN_THRESHOLD) numOpen++;
  }

  const teammate_dist_to_basket = teammates.map((t) =>
    distanceToBasket(t.x, t.y),
  );
  const teammate_zone_fg = teammates.map((t) => t.zone_fg_pct);

  const pass_lane_features = teammates.map((t) => {
    const { minDist, corridor } = laneClearance(
      bh.x,
      bh.y,
      t.x,
      t.y,
      defenders,
    );
    const passDist = dist(bh.x, bh.y, t.x, t.y);
    return [minDist, corridor, passDist] as [number, number, number];
  });

  return {
    grid_zone: gridZone(bh.x, bh.y),
    is_three_point_zone: isThreePointZone(bh.x, bh.y),
    closest_defender_dist: closest,
    help_defender_dist: help,
    num_defenders_within_6ft: within6,
    best_teammate_openness: bestOpen,
    num_open_teammates: numOpen,
    teammate_openness,
    teammate_zone_fg,
    teammate_dist_to_basket,
    pass_lane_features,
  };
}
