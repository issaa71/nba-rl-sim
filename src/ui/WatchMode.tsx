// Watch mode — a hands-off "let the agent run" autopilot wrapper around the
// Explorer. It plays possessions back-to-back: each possession plays from the
// top to its decision frame, a brief outcome interstitial reveals what happened
// (player did X, agent wanted Y, result), then it auto-advances to the next.
//
// Watch mode IS the Explorer (same parity-tested live playback machinery) in an
// autopilot shell — not a parallel implementation. The shell owns the rotation
// (curated 40 + lazily-fetched stream possessions, seed-shuffled), the
// possession counter, the pause/skip/exit controls, and the interstitial.
// Pausing leaves the Explorer fully interactive (what-if drag still works).

import { useCallback, useEffect, useMemo, useState } from "react";
import type { AppData } from "../data/load";
import { buildWatchRotation, loadStreamPossessions } from "../data/load";
import { actionLabel, type Possession } from "../data/types";
import { Explorer } from "./Explorer";
import { ModelToggle } from "./bits";
import { outcomeText, type ModelMode } from "./model";

// How long the outcome interstitial lingers before auto-advancing (ms).
const INTERSTITIAL_MS = 1500;

interface WatchModeProps {
  data: AppData;
  model: ModelMode;
  onModelChange: (m: ModelMode) => void;
  onExit: () => void;
}

type Phase = "playing" | "interstitial";

export function WatchMode({
  data,
  model,
  onModelChange,
  onExit,
}: WatchModeProps) {
  const [stream, setStream] = useState<Possession[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [index, setIndex] = useState(0);
  const [phase, setPhase] = useState<Phase>("playing");
  const [paused, setPaused] = useState(false);

  // Lazy-load the stream possessions ONLY when watch mode mounts. The landing /
  // browser view never imports this payload.
  useEffect(() => {
    let alive = true;
    loadStreamPossessions()
      .then((s) => {
        if (alive) setStream(s);
      })
      .catch((e: unknown) => {
        if (alive) setLoadError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      alive = false;
    };
  }, []);

  // Curated 40 + stream, seed-shuffled into a stable rotation.
  const rotation = useMemo(() => {
    if (!stream) return null;
    return buildWatchRotation(data.possessions.possessions, stream);
  }, [data.possessions.possessions, stream]);

  const current = rotation ? rotation[index] : null;

  const advance = useCallback(() => {
    setPhase("playing");
    setIndex((i) => {
      const n = rotation?.length ?? 1;
      return (i + 1) % n;
    });
  }, [rotation]);

  // Auto-advance after the interstitial lingers. The timer is held while paused
  // (it restarts from full on resume — the linger is short enough that this
  // reads naturally) and cleared on skip/unmount.
  useEffect(() => {
    if (phase !== "interstitial" || paused) return;
    const t = window.setTimeout(advance, INTERSTITIAL_MS);
    return () => window.clearTimeout(t);
  }, [phase, paused, index, advance]);

  const onReachedDecision = useCallback(() => {
    setPhase("interstitial");
  }, []);

  const skip = useCallback(() => {
    advance();
  }, [advance]);

  // --- loading / error states ---
  if (loadError) {
    return (
      <div className="watch">
        <WatchBar
          index={0}
          total={0}
          paused={paused}
          onTogglePause={() => setPaused((v) => !v)}
          onSkip={skip}
          onExit={onExit}
          model={model}
          onModelChange={onModelChange}
          disabled
        />
        <div className="center-state">
          <p>Could not load the watch rotation.</p>
          <p className="mono" style={{ fontSize: 12 }}>
            {loadError}
          </p>
        </div>
      </div>
    );
  }

  if (!rotation || !current) {
    return (
      <div className="watch">
        <WatchBar
          index={0}
          total={0}
          paused={paused}
          onTogglePause={() => setPaused((v) => !v)}
          onSkip={skip}
          onExit={onExit}
          model={model}
          onModelChange={onModelChange}
          disabled
        />
        <div className="center-state">
          <div className="spinner" />
          <p>Loading the rotation…</p>
        </div>
      </div>
    );
  }

  const teammateNames = current.entity_names_network_order.slice(1);
  const showInterstitial = phase === "interstitial";

  return (
    <div className="watch">
      <Explorer
        key={current.id}
        possession={current}
        data={data}
        model={model}
        onModelChange={onModelChange}
        onBack={onExit}
        autoPlay
        paused={paused || showInterstitial}
        onReachedDecision={onReachedDecision}
        topSlot={
          <WatchBar
            index={index}
            total={rotation.length}
            paused={paused}
            onTogglePause={() => setPaused((v) => !v)}
            onSkip={skip}
            onExit={onExit}
            model={model}
            onModelChange={onModelChange}
          />
        }
      />

      {showInterstitial && (
        <Interstitial
          player={actionLabel(current.player_action, teammateNames)}
          agent={actionLabel(current.agent_action, teammateNames)}
          result={outcomeText(current.outcome)}
          paused={paused}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------

function WatchBar({
  index,
  total,
  paused,
  onTogglePause,
  onSkip,
  onExit,
  model,
  onModelChange,
  disabled = false,
}: {
  index: number;
  total: number;
  paused: boolean;
  onTogglePause: () => void;
  onSkip: () => void;
  onExit: () => void;
  model: ModelMode;
  onModelChange: (m: ModelMode) => void;
  disabled?: boolean;
}) {
  return (
    <div className="watchbar">
      <button className="explorer__back" onClick={onExit}>
        ← Exit watch
      </button>
      <span className="watchbar__counter mono">
        {total > 0 ? (
          <>
            possession {index + 1} · {total} in rotation
          </>
        ) : (
          "loading rotation…"
        )}
      </span>
      <div className="watchbar__controls">
        <button
          className="btn btn--sm"
          onClick={onTogglePause}
          disabled={disabled}
          aria-pressed={paused}
        >
          {paused ? "► Resume" : "❚❚ Pause"}
        </button>
        <button
          className="btn btn--sm"
          onClick={onSkip}
          disabled={disabled}
          aria-label="Skip to next possession"
        >
          Skip ▸▸
        </button>
        <ModelToggle value={model} onChange={onModelChange} />
      </div>
    </div>
  );
}

function Interstitial({
  player,
  agent,
  result,
  paused,
}: {
  player: string;
  agent: string;
  result: string;
  paused: boolean;
}) {
  const agree = player === agent;
  return (
    <div className="interstitial" role="status" aria-live="polite">
      <div className="interstitial__card">
        <span className="panel__label">What happened</span>
        <div className="interstitial__rows">
          <div className="interstitial__row">
            <span className="interstitial__k">Player did</span>
            <span className="interstitial__v">{player}</span>
          </div>
          <div className="interstitial__row">
            <span className="interstitial__k">Agent wanted</span>
            <span className="interstitial__v interstitial__v--agent">
              {agent}
            </span>
          </div>
          <div className="interstitial__row">
            <span className="interstitial__k">Result</span>
            <span className="interstitial__v">{result}</span>
          </div>
        </div>
        <p className="interstitial__note">
          {agree
            ? "Agreement — the agent made the same call."
            : "The agent would have chosen differently."}
        </p>
        <span className="interstitial__next mono">
          {paused ? "paused" : "next play…"}
        </span>
      </div>
    </div>
  );
}
