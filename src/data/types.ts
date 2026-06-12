// Shapes for the exported demo JSONs under /public/data.
// Authoritative spec: export_demo/EXPORT_README.md.

export type Category = "declined_the_shot" | "wanted_the_shot" | "agreement";

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
  category: Category;
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
  score: number;
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
// Action labels
// ---------------------------------------------------------------------------

export const ACTION_SHOOT = 0;

/** Network action index -> human label given a possession's teammate names. */
export function actionLabel(action: number, teammateNames: string[]): string {
  if (action === ACTION_SHOOT) return "Shoot";
  const name = teammateNames[action - 1];
  return name ? `Pass to ${name}` : `Pass (slot ${action})`;
}
