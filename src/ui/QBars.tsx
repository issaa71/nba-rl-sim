// Hand-drawn horizontal Q-value bars (no chart library).
//
// Q-values are signed and unbounded, so bars are laid out around a shared zero
// baseline: positive grows right of zero, negative grows left. The best action
// is highlighted in accent; the player's recorded action gets a marker.

export interface QBarsProps {
  q: number[]; // [shoot, pass_1..4]
  best: number; // argmax index to highlight
  labels: string[]; // 5 short labels
  /** index of the player's recorded action, for a small marker (optional). */
  playerAction?: number;
}

export function QBars({ q, best, labels, playerAction }: QBarsProps) {
  const min = Math.min(...q, 0);
  const max = Math.max(...q, 0);
  const span = max - min || 1;
  // position of zero within [0,1]
  const zeroFrac = (0 - min) / span;

  return (
    <div className="qbars" role="list" aria-label="Q-values per action">
      {q.map((val, i) => {
        const frac = (val - min) / span;
        let left: number;
        let width: number;
        if (val >= 0) {
          left = zeroFrac;
          width = frac - zeroFrac;
        } else {
          left = frac;
          width = zeroFrac - frac;
        }
        const isBest = i === best;
        const isPlayer = i === playerAction;
        return (
          <div className="qbar" role="listitem" key={i}>
            <span className="qbar__name" title={labels[i]}>
              {isPlayer ? "● " : ""}
              {labels[i]}
            </span>
            <div className="qbar__track">
              <span className="qbar__zero" style={{ left: `${zeroFrac * 100}%` }} />
              <span
                className={
                  "qbar__fill" +
                  (isBest ? " is-best" : val >= 0 ? " is-positive" : "")
                }
                style={{
                  left: `${left * 100}%`,
                  width: `${Math.max(width * 100, 1.5)}%`,
                }}
              />
            </div>
            <span className={"qbar__val" + (isBest ? " is-best" : "")}>
              {val >= 0 ? "+" : ""}
              {val.toFixed(2)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
