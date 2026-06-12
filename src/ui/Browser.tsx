// Possession browser — the 40 curated possessions grouped by story type.

import type { Category, Possession } from "../data/types";
import { OutcomeChip } from "./bits";

const GROUP_ORDER: Category[] = [
  "declined_the_shot",
  "wanted_the_shot",
  "agreement",
];

const GROUP_META: Record<Category, { title: string; sub: string }> = {
  declined_the_shot: {
    title: "Declined the shot",
    sub: "The player shot; the agent wanted a pass.",
  },
  wanted_the_shot: {
    title: "Wanted the shot",
    sub: "The player passed; the agent wanted to shoot.",
  },
  agreement: {
    title: "Agreement",
    sub: "Agent and player chose the same action (control).",
  },
};

function PossessionCard({
  p,
  onOpen,
}: {
  p: Possession;
  onOpen: (id: string) => void;
}) {
  const players = p.entity_names_network_order;
  const headline = players[0] ?? p.ball_handler_name;
  const others = players.slice(1, 3).filter(Boolean).join(", ");
  return (
    <button className="poss-card" onClick={() => onOpen(p.id)}>
      <div className="poss-card__top">
        <span className="poss-card__players">{headline}</span>
        <OutcomeChip outcome={p.outcome} />
      </div>
      {others && (
        <div className="eyebrow eyebrow--faint" style={{ marginTop: 8 }}>
          with {others}
        </div>
      )}
      <p className="poss-card__summary">{p.summary}</p>
      <div className="poss-card__foot">
        {p.score !== undefined && (
          <span className="chip chip--accent">Q-gap {p.score.toFixed(2)}</span>
        )}
        <span className="poss-card__arrow">Explore →</span>
      </div>
    </button>
  );
}

function WatchCallout({ onWatch }: { onWatch: () => void }) {
  return (
    <button className="watch-callout" onClick={onWatch}>
      <span className="watch-callout__play" aria-hidden="true">
        ▶
      </span>
      <span className="watch-callout__body">
        <span className="watch-callout__title">Watch the agent run</span>
        <span className="watch-callout__sub">
          Sit back — the live agent plays through possession after possession,
          calling each shoot-or-pass decision as it happens. Pause anytime to
          drag a player and test a what-if.
        </span>
      </span>
      <span className="watch-callout__arrow" aria-hidden="true">
        →
      </span>
    </button>
  );
}

export function Browser({
  possessions,
  onOpen,
  onWatch,
}: {
  possessions: Possession[];
  onOpen: (id: string) => void;
  onWatch: () => void;
}) {
  return (
    <div className="section">
      <WatchCallout onWatch={onWatch} />
      {GROUP_ORDER.map((cat) => {
        const group = possessions.filter((p) => p.category === cat);
        if (group.length === 0) return null;
        const meta = GROUP_META[cat];
        return (
          <div className="group" key={cat} id={cat}>
            <div className="group__head">
              <h2>{meta.title}</h2>
              <span className="group__sub">{meta.sub}</span>
              <span className="group__count">{group.length}</span>
            </div>
            <div className="cards">
              {group.map((p) => (
                <PossessionCard key={p.id} p={p} onOpen={onOpen} />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
