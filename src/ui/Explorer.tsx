// Explorer view — the heart. Court replay + scrubber + live Q-values +
// outcome reveal + what-if dragging. Everything is computed in-browser via the
// (parity-tested) engine.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { LoadedNetwork } from "../engine/network";
import {
  frameToRawState,
  playerChoice,
  runState,
  whatIfToRawState,
  type WhatIfState,
} from "../engine/explorer";
import { BASKET_X, BASKET_Y } from "../engine/features";
import type { AppData } from "../data/load";
import { makeZoneFgLookup } from "../data/load";
import { actionLabel, type Possession } from "../data/types";
import { Court, type CourtArrow, type CourtEntity } from "./Court";
import { QBars } from "./QBars";
import { ModelToggle } from "./bits";
import { MODEL_LABELS, outcomeText, type ModelMode } from "./model";

const SPEEDS = [0.5, 1, 2] as const;

interface ExplorerProps {
  possession: Possession;
  data: AppData;
  model: ModelMode;
  onModelChange: (m: ModelMode) => void;
  onBack: () => void;
}

/** Short labels for the 5 actions from the possession's teammate names. */
function shortLabels(teammateNames: string[]): string[] {
  return [
    "Shoot",
    ...teammateNames.map((n) => firstName(n) ?? "Pass"),
  ];
}
function firstName(full: string): string | undefined {
  if (!full) return undefined;
  const parts = full.split(" ");
  return parts.length > 1 ? parts[parts.length - 1] : full;
}

export function Explorer({
  possession: p,
  data,
  model,
  onModelChange,
  onBack,
}: ExplorerProps) {
  const nFrames = p.frames.length;
  const [frameIdx, setFrameIdx] = useState(p.decision_frame);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>(1);
  const [activeId, setActiveId] = useState<string | null>(null);

  const [whatIf, setWhatIf] = useState(false);
  // What-if snapshot of the current frame's positions (null = follow recorded).
  // Drag events are coalesced to one commit per animation frame via `rafRef` +
  // `pendingRef` (in the drag handler — not an effect), so the engine recompute
  // runs at most once per frame even under a fast drag.
  const [wiState, setWiState] = useState<WhatIfState | null>(null);
  const rafRef = useRef<number | null>(null);
  const pendingRef = useRef<WhatIfState | null>(null);

  const net: LoadedNetwork = model === "dqn" ? data.dqn : data.dueling;
  const teammateNames = p.entity_names_network_order.slice(1);
  const labels = useMemo(() => shortLabels(teammateNames), [teammateNames]);

  const isDecisionFrame = frameIdx === p.decision_frame;
  const frame = p.frames[frameIdx];

  // Snapshot the draggable positions for a given frame.
  const snapshotFrame = useCallback(
    (idx: number): WhatIfState => {
      const f = p.frames[idx];
      return {
        ballHandler: {
          x: f.ball_handler.x,
          y: f.ball_handler.y,
          vx: f.ball_handler.vx,
          vy: f.ball_handler.vy,
        },
        teammates: f.teammates.map((t) => ({
          x: t.x,
          y: t.y,
          vx: t.vx,
          vy: t.vy,
          zone_fg_pct: 0, // recomputed via the per-teammate FG lookup
        })),
        defenders: f.defenders.map((d) => ({
          x: d.x,
          y: d.y,
          vx: d.vx,
          vy: d.vy,
        })),
      };
    },
    [p.frames],
  );

  // Reset the what-if snapshot during render when the frame changes while in
  // what-if mode (the "adjust state during render" recipe — no effect needed).
  const [wiFrame, setWiFrame] = useState(frameIdx);
  if (whatIf && wiFrame !== frameIdx) {
    setWiFrame(frameIdx);
    setWiState(snapshotFrame(frameIdx));
  }

  // --- playback loop (advance frames; stop at decision frame) ---
  useEffect(() => {
    if (!playing || whatIf) return;
    const stepMs = 700 / speed;
    const id = window.setInterval(() => {
      setFrameIdx((i) => {
        if (i >= p.decision_frame) {
          setPlaying(false);
          return p.decision_frame;
        }
        return i + 1;
      });
    }, stepMs);
    return () => window.clearInterval(id);
  }, [playing, speed, whatIf, p.decision_frame]);

  const enterWhatIf = useCallback(() => {
    setWiState(snapshotFrame(frameIdx));
    setWiFrame(frameIdx);
    setWhatIf(true);
    setPlaying(false);
  }, [frameIdx, snapshotFrame]);

  const toggleWhatIf = useCallback(() => {
    if (whatIf) {
      setWhatIf(false);
      setWiState(null);
    } else {
      enterWhatIf();
    }
  }, [whatIf, enterWhatIf]);

  const resetWhatIf = useCallback(() => {
    setWiState(snapshotFrame(frameIdx));
  }, [frameIdx, snapshotFrame]);

  // --- zone-FG lookups for what-if (BH + per teammate) ---
  const bhFgLookup = useMemo(
    () => makeZoneFgLookup(data.zoneFg, frame.ball_handler.compact_id),
    [data.zoneFg, frame.ball_handler.compact_id],
  );
  const teammateFgLookups = useMemo(
    () =>
      frame.teammates.map((t) =>
        makeZoneFgLookup(data.zoneFg, t.compact_id),
      ),
    [data.zoneFg, frame.teammates],
  );

  // --- compute Q for the displayed state ---
  const playerIds = p.entity_ids_network_order;
  const engineWi = wiState;
  const result = useMemo(() => {
    if (whatIf && engineWi) {
      // teammate FG% recomputed from each teammate's table at its (dragged) spot
      const wi: WhatIfState = {
        ...engineWi,
        teammates: engineWi.teammates.map((t, i) => ({
          ...t,
          zone_fg_pct: teammateFgLookups[i](t.x, t.y),
        })),
      };
      const state = whatIfToRawState(wi, bhFgLookup);
      state.shot_clock = frame.shot_clock;
      const live = runState(net, state, playerIds);
      if (model === "player") return playerChoice(live.q, p.player_action);
      return live;
    }
    // recorded path
    const state = frameToRawState(frame);
    const live = runState(net, state, playerIds);
    if (model === "player") return playerChoice(live.q, p.player_action);
    return live;
  }, [
    whatIf,
    engineWi,
    frame,
    net,
    model,
    playerIds,
    p.player_action,
    bhFgLookup,
    teammateFgLookups,
  ]);

  // --- court entities (recorded or what-if positions) ---
  const entities: CourtEntity[] = useMemo(() => {
    const out: CourtEntity[] = [];
    const bh = whatIf && wiState ? wiState.ballHandler : frame.ball_handler;
    out.push({
      id: "bh",
      kind: "ball-handler",
      x: bh.x,
      y: bh.y,
      label: p.entity_names_network_order[0] ?? p.ball_handler_name,
    });
    const tms = whatIf && wiState ? wiState.teammates : frame.teammates;
    tms.forEach((t, i) => {
      out.push({
        id: `tm${i}`,
        kind: "teammate",
        x: t.x,
        y: t.y,
        slot: i + 1,
        label: teammateNames[i] ?? `Teammate ${i + 1}`,
      });
    });
    const dfs =
      whatIf && wiState ? wiState.defenders : frame.defenders;
    dfs.forEach((d, i) => {
      out.push({
        id: `df${i}`,
        kind: "defender",
        x: d.x,
        y: d.y,
        label: `Defender ${i + 1}`,
      });
    });
    return out;
  }, [whatIf, wiState, frame, p, teammateNames]);

  const ball = useMemo(() => {
    const bh = whatIf && wiState ? wiState.ballHandler : frame.ball_handler;
    return { x: bh.x, y: bh.y };
  }, [whatIf, wiState, frame]);

  // --- agent recommendation arrow ---
  const arrow: CourtArrow | null = useMemo(() => {
    const rec = result.best;
    const bh = entities.find((e) => e.id === "bh");
    if (!bh) return null;
    if (rec === 0) {
      // shoot — arrow toward basket
      return {
        fromX: bh.x,
        fromY: bh.y,
        toX: BASKET_X,
        toY: BASKET_Y,
        kind: "shoot",
      };
    }
    const target = entities.find((e) => e.id === `tm${rec - 1}`);
    if (!target) return null;
    return {
      fromX: bh.x,
      fromY: bh.y,
      toX: target.x,
      toY: target.y,
      kind: "pass",
    };
  }, [result.best, entities]);

  // --- drag handler: coalesce rapid pointer-moves to one commit per frame ---
  const applyDrag = useCallback(
    (base: WhatIfState, id: string, x: number, y: number): WhatIfState => {
      if (id === "bh") {
        return { ...base, ballHandler: { ...base.ballHandler, x, y } };
      }
      if (id.startsWith("tm")) {
        const i = Number(id.slice(2));
        return {
          ...base,
          teammates: base.teammates.map((t, j) =>
            j === i ? { ...t, x, y } : t,
          ),
        };
      }
      if (id.startsWith("df")) {
        const i = Number(id.slice(2));
        return {
          ...base,
          defenders: base.defenders.map((d, j) =>
            j === i ? { ...d, x, y } : d,
          ),
        };
      }
      return base;
    },
    [],
  );

  const onDrag = useCallback(
    (id: string, x: number, y: number) => {
      const base = pendingRef.current ?? wiState;
      if (!base) return;
      const next = applyDrag(base, id, x, y);
      pendingRef.current = next;
      if (rafRef.current != null) return; // already scheduled this frame
      rafRef.current = requestAnimationFrame(() => {
        rafRef.current = null;
        if (pendingRef.current) setWiState(pendingRef.current);
        pendingRef.current = null;
      });
    },
    [wiState, applyDrag],
  );

  // Clear any pending rAF on unmount.
  useEffect(() => {
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  // recommendation + player action labels
  const agentLabel = actionLabel(result.best, teammateNames);
  const playerLabel = actionLabel(p.player_action, teammateNames);
  const agree = p.agent_action === p.player_action;

  const showOutcome = isDecisionFrame && !whatIf;
  const modeNote =
    model === "player"
      ? "Bars show the Dueling agent's Q-values; the highlighted action is what the player actually did."
      : `Q-values from the ${MODEL_LABELS[model]} model, computed live in your browser.`;

  return (
    <div className="explorer">
      <button className="explorer__back" onClick={onBack}>
        ← All possessions
      </button>

      <div className="explorer__head">
        <div>
          <p className="eyebrow">{groupLabel(p.category)}</p>
          <h1 className="explorer__title">{p.ball_handler_name}</h1>
          <p className="explorer__meta">{p.summary}</p>
        </div>
        <div className="explorer__model">
          <span className="panel__label" style={{ marginBottom: 8, display: "block" }}>
            Model
          </span>
          <ModelToggle value={model} onChange={onModelChange} />
        </div>
      </div>

      <div className="explorer__grid">
        {/* LEFT: court + transport */}
        <div>
          <div className={"court-wrap" + (whatIf ? " court-wrap--whatif" : "")}>
            {whatIf && <span className="whatif-tag">Hypothetical</span>}
            <div className="court-clock">
              <span className="court-clock__val">
                {frame.shot_clock.toFixed(1)}
              </span>
              <span className="court-clock__label">shot clock</span>
            </div>
            <Court
              entities={entities}
              ball={ball}
              arrow={arrow}
              draggable={whatIf}
              activeId={activeId}
              onActiveChange={setActiveId}
              onDrag={onDrag}
            />
          </div>

          <div className="legend">
            <span>
              <i style={{ background: "#047857" }} /> Offense
            </span>
            <span>
              <i
                style={{
                  background: "#047857",
                  boxShadow: "0 0 0 2px #022c22",
                }}
              />{" "}
              Ball-handler
            </span>
            <span>
              <i style={{ background: "#a39b8d" }} /> Defender
            </span>
            <span>
              <i style={{ background: "#d97706" }} /> Ball
            </span>
          </div>

          {/* transport */}
          <div className="transport">
            <button
              className="iconbtn"
              aria-label={playing ? "Pause" : "Play"}
              disabled={whatIf || nFrames <= 1}
              onClick={() => {
                if (frameIdx >= p.decision_frame) setFrameIdx(0);
                setPlaying((v) => !v);
              }}
            >
              {playing ? "❚❚" : "►"}
            </button>
            <div className="scrub">
              <input
                type="range"
                min={0}
                max={nFrames - 1}
                value={frameIdx}
                disabled={nFrames <= 1}
                onChange={(e) => {
                  setPlaying(false);
                  setFrameIdx(Number(e.target.value));
                }}
                aria-label="Scrub possession frames"
              />
              <span className="scrub__pos">
                {frameIdx + 1}/{nFrames}
              </span>
            </div>
            <div className="segment" aria-label="Playback speed">
              {SPEEDS.map((s) => (
                <button
                  key={s}
                  aria-pressed={speed === s}
                  onClick={() => setSpeed(s)}
                >
                  {s}×
                </button>
              ))}
            </div>
          </div>

          {/* what-if controls */}
          <div className="whatif-bar">
            <label className="switch">
              <input
                type="checkbox"
                checked={whatIf}
                onChange={toggleWhatIf}
              />
              <span className="switch__track">
                <span className="switch__knob" />
              </span>
              What-if mode
            </label>
            {whatIf ? (
              <>
                <button className="btn btn--sm" onClick={resetWhatIf}>
                  Reset to real play
                </button>
                <span className="whatif-hint">
                  Drag any player — Q-values recompute live.
                </span>
              </>
            ) : (
              <span className="whatif-hint">
                Pause on a frame, then drag players to test alternatives.
              </span>
            )}
          </div>
        </div>

        {/* RIGHT: decision + Q-values + outcome */}
        <div className="panel">
          <div className="panel__block">
            <span className="panel__label">
              {isDecisionFrame
                ? "Decision point"
                : `Frame ${frameIdx + 1} — pre-decision`}
            </span>
            <div className="decision-row">
              <div>
                <div className="decision-col__cap">Agent recommends</div>
                <div className="decision-col__val is-agent">{agentLabel}</div>
              </div>
              <div>
                <div className="decision-col__cap">Player chose</div>
                <div className="decision-col__val">{playerLabel}</div>
              </div>
            </div>
            {!whatIf && agree && isDecisionFrame && (
              <p className="agree-note">
                Agreement — the agent and {p.ball_handler_name} made the same
                call here.
              </p>
            )}
          </div>

          <div className="panel__block">
            <span className="panel__label">Action values (Q)</span>
            <QBars
              q={result.q}
              best={result.best}
              labels={labels}
              playerAction={isDecisionFrame ? p.player_action : undefined}
            />
            <p className="agree-note">{modeNote}</p>
          </div>

          {showOutcome && (
            <div className="outcome">
              <span className="panel__label">What happened</span>
              <div className="outcome__rows">
                <div className="outcome__row">
                  <span className="outcome__k">Player did</span>
                  <span className="outcome__v">{playerLabel}</span>
                </div>
                <div className="outcome__row">
                  <span className="outcome__k">Agent wanted</span>
                  <span className="outcome__v" style={{ color: "#047857" }}>
                    {actionLabel(p.agent_action, teammateNames)}
                  </span>
                </div>
                <div className="outcome__row">
                  <span className="outcome__k">Result</span>
                  <span className="outcome__v">{outcomeText(p.outcome)}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function groupLabel(cat: string): string {
  if (cat === "declined_the_shot") return "Declined the shot";
  if (cat === "wanted_the_shot") return "Wanted the shot";
  return "Agreement";
}
