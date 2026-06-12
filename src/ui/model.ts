// Model-mode constants + outcome text (non-component helpers split out so the
// component files stay refresh-friendly).

import type { Outcome } from "../data/types";

export type ModelMode = "dueling" | "dqn" | "player";

export const MODEL_LABELS: Record<ModelMode, string> = {
  dueling: "Dueling DQN",
  dqn: "DQN",
  player: "What the player did",
};

export function outcomeText(o: Outcome): string {
  if (o.type === "turnover") return "Turnover";
  const kind = o.is_three ? "3PT" : "2PT";
  return o.made ? `Made ${kind}` : `Missed ${kind}`;
}
