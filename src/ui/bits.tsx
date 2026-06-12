// Small shared presentational components.

import type { Outcome } from "../data/types";
import { MODEL_LABELS, outcomeText, type ModelMode } from "./model";

export function OutcomeChip({ outcome }: { outcome: Outcome }) {
  let cls = "chip--turnover";
  if (outcome.type === "shot") cls = outcome.made ? "chip--made" : "chip--miss";
  return (
    <span className={`chip ${cls}`}>
      <i className="chip-dot" />
      {outcomeText(outcome)}
    </span>
  );
}

export function ModelToggle({
  value,
  onChange,
}: {
  value: ModelMode;
  onChange: (m: ModelMode) => void;
}) {
  const modes: ModelMode[] = ["dueling", "dqn", "player"];
  return (
    <div className="segment" role="tablist" aria-label="Model">
      {modes.map((m) => (
        <button
          key={m}
          role="tab"
          aria-pressed={value === m}
          aria-selected={value === m}
          onClick={() => onChange(m)}
        >
          {MODEL_LABELS[m]}
        </button>
      ))}
    </div>
  );
}
