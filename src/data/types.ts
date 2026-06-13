// Shapes for the exported demo JSONs under /public/data.
// Authoritative spec: export_demo/EXPORT_README.md.

/** Curated story buckets shown on the browser. */
export type Category = "declined_the_shot" | "wanted_the_shot" | "agreement";

/**
 * A possession's category as it appears in the data. Curated possessions carry
 * a story bucket; stream possessions (watch-mode filler from the held-out set)
 * are uncategorized and tagged `"stream"`.
 */
export type PossessionCategory = Category | "stream";

export interface FrameBallHandler {
  compact_id: number;
  nba_id: number;
  team: "offense";
  x: number;
  y: number;
  vx: number;
  vy: number;
  distance_to_basket: number;
  zone_fg_pct: number;
}

export interface FrameTeammate {
  slot: number;
  compact_id: number;
  nba_id: number;
  team: "offense";
  x: number;
  y: number;
  vx: number;
  vy: number;
}

export interface FrameDefender {
  slot: number;
  team: "defense";
  x: number;
  y: number;
  vx: number;
  vy: number;
}

export interface FrameContext {
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

export interface Frame {
  ball_handler: FrameBallHandler;
  teammates: FrameTeammate[]; // length 4, dist-to-BH sorted
  defenders: FrameDefender[]; // length 5, closest-first sorted
  ball: { x: number; y: number; note?: string };
  shot_clock: number;
  game_clock: number | null;
  quarter: number | null;
  context: FrameContext;
}

export type OutcomeType = "shot" | "turnover";

export interface Outcome {
  type: OutcomeType;
  made?: boolean;
  is_three?: boolean;
}

export interface Possession {
  id: string;
  category: PossessionCategory;
  game_id: string;
  n_frames: number;
  /** index into `frames` where the recorded decision happens (the last frame). */
  decision_frame: number;
  ball_handler_compact_id: number;
  ball_handler_name: string;
  /** [BH, tm1, tm2, tm3, tm4] compact ids — the embedding indices for the network. */
  entity_ids_network_order: number[];
  entity_names_network_order: string[];
  /** 0 = shoot; 1-4 = pass to teammate slot 1-4. */
  player_action: number;
  agent_action: number;
  /** Dueling model Q-values at the decision frame, network order [shoot, pass_1..4]. */
  agent_q_values: number[];
  /** |Q-gap| between agent and player action. Present on curated possessions only. */
  score?: number;
  outcome: Outcome;
  summary: string;
  frames: Frame[];
}

export interface PossessionsFile {
  note: string;
  model: string;
  categories: Record<Category, string>;
  counts: Record<Category, number>;
  possessions: Possession[];
}

export interface PlayersFile {
  note: string;
  meta: Record<string, number>;
  players: Record<string, { name: string; nba_id: number }>;
}

// ---------------------------------------------------------------------------
// High-frequency tracking (real SportVU motion) — public/data/tracking_*.json.
//
// Authoritative spec: the export `note`. Positions are in the env half-court
// frame but with the BASKET ON THE LEFT at (5.25, 25); x in [0, 47], y in
// [0, 50]. (The feature/possessions frame puts the basket on the RIGHT at
// x = 47 — engine/tracking.ts holds the exact transform.) `players` are 10
// entities in NETWORK order [ball_handler, tm1..4, def1..5]; `ball` is
// [x, y, z] (z drives the shot-arc render). `decision_snap_indices[k]` is the
// tracking-frame index nearest the k-th recorded decision point — the index at
// which playback defers to the parity-exact recorded path.
// ---------------------------------------------------------------------------

/** One tracking frame: real positions + clocks + real ball with height. */
export interface TrackingFrame {
  /** wall-time offset (s) from the span start. */
  t: number;
  /** shot clock (s); may be null when the source clock is missing. */
  sc: number | null;
  /** game clock (s); may be null. */
  gc: number | null;
  /** quarter; may be null. */
  q: number | null;
  /** 10 entities [BH, tm1..4, def1..5], each [x, y] in the tracking frame. */
  players: [number, number][];
  /** real ball [x, y, z] (z for shot-arc rendering). */
  ball: [number, number, number];
}

export interface TrackingPossession {
  id: string;
  game_id: string;
  fps: number;
  n_frames: number;
  /** original game orientation (positions are already basket-relative). */
  attacking_right: boolean;
  /** tracking-frame index of each recorded decision point, ascending. */
  decision_snap_indices: number[];
  frames: TrackingFrame[];
}

export interface TrackingFile {
  note: string;
  fps: number;
  frame_player_order: string[];
  ball_format: string;
  court_frame: string;
  category: "curated" | "stream";
  n_possessions: number;
  possessions: TrackingPossession[];
}

// ---------------------------------------------------------------------------
// Action labels
// ---------------------------------------------------------------------------

export const ACTION_SHOOT = 0;

/** Network action index -> human label given a possession's teammate names. */
export function actionLabel(action: number, teammateNames: string[]): string {
  if (action === ACTION_SHOOT) return "Shoot";
  const name = teammateNames[action - 1];
  return name ? `Pass to ${name}` : `Pass (slot ${action})`;
}
