// Explorer view — the heart. Court replay + scrubber + live Q-values +
// outcome reveal + what-if dragging. Everything is computed in-browser via the
// (parity-tested) engine.
//
// Two layers, additive:
//   1. Decision-point experience (unchanged): clickable/steppable decision
//      frames with EXACT recorded values, outcome reveal, what-if drag.
//   2. Honest live playback (added on top): recorded frames are DECISION POINTS
//      at irregular real intervals — NOT uniform 2 Hz samples. Pressing PLAY
//      walks an explicit DWELL / GLIDE / CUT timeline (engine/explorer.ts):
//      decision moments are HELD (dwell — the parity-exact recorded path),
//      movement between them GLIDES with a duration proportional to the real
//      time elapsed, and shot-clock RESETS / over-long gaps CUT (a brief
//      dissolve, no dishonest positional interpolation). The agent is
//      re-evaluated every animation frame during glides (interpolate -> re-sort
//      -> re-lookup zone FG% -> deriveContext -> buildFeatures -> forward) and
//      uses the parity-exact recorded path during dwells.

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import type { LoadedNetwork } from "../engine/network";
import {
  buildTimeline,
  frameToRawState,
  interpolateFrame,
  liveFrameToRawState,
  playerChoice,
  runState,
  timeForFrame,
  timelineCursor,
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

// A playback time within this many frame-units of an integer is treated as
// landing ON that recorded frame -> use the parity-exact recorded path.
const SNAP_EPS = 1e-3;

// Clicking within this fraction of the scrubber range to a recorded-frame
// marker snaps the playback head onto that frame's dwell (exact-parity values).
const SCRUB_SNAP_FRAC = 0.02;

// Chip hysteresis: while interpolating between 2 Hz samples the live argmax can
// flicker between near-tied actions. Only switch the displayed recommendation
// when the new argmax beats the currently-shown action by at least this much Q.
const HYSTERESIS_EPS = 0.01;

interface ExplorerProps {
  possession: Possession;
  data: AppData;
  model: ModelMode;
  onModelChange: (m: ModelMode) => void;
  onBack: () => void;
  /**
   * Autopilot (watch mode): start playback from the top of the possession on
   * mount and replay continuously. Defaults to false (manual decision-point
   * experience). Pausing is still fully interactive (what-if drag etc.).
   */
  autoPlay?: boolean;
  /**
   * Fired once when continuous playback reaches the decision frame. Watch mode
   * uses this to show the outcome interstitial and advance to the next play.
   */
  onReachedDecision?: () => void;
  /**
   * External pause (watch mode). When true the continuous playback loop is held
   * even if internally "playing" — lets an autopilot wrapper freeze the action
   * while still allowing the user to drag (what-if) on the held frame.
   */
  paused?: boolean;
  /**
   * Replaces the default "← All possessions" back button row with custom chrome
   * (watch-mode control bar + counter). When omitted, the back button renders.
   */
  topSlot?: ReactNode;
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
  autoPlay = false,
  onReachedDecision,
  paused = false,
  topSlot,
}: ExplorerProps) {
  const nFrames = p.frames.length;

  // Playback walks an explicit DWELL/GLIDE/CUT timeline over frames 0..decision
  // (playback stops at the decision frame, which is not always the last frame).
  // The timeline is keyed on the per-frame shot clocks; it is the honest map of
  // real possession time -> playback time.
  const timeline = useMemo(
    () =>
      buildTimeline(
        p.frames.slice(0, p.decision_frame + 1).map((f) => f.shot_clock),
      ),
    [p.frames, p.decision_frame],
  );

  // Source of truth for playback: elapsed wall-clock against the timeline, in
  // timeline-ms at 1x. The speed control scales how fast this advances. Manual
  // mode opens parked on the decision frame; autopilot starts at the top.
  const [clock, setClock] = useState(() =>
    autoPlay ? 0 : timeForFrame(timeline, p.decision_frame),
  );
  const [playing, setPlaying] = useState(autoPlay);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>(1);
  const [activeId, setActiveId] = useState<string | null>(null);
  // Player-name labels on the court (user asked for the numbers back). On by
  // default; defenders are unnamed in the data so only offense is labeled.
  const [names, setNames] = useState(true);

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

  // Resolve the playback head against the timeline.
  //   - playT: continuous frame position (integer on a dwell/cut, fractional in
  //     a glide) — drives positional interpolation exactly as before.
  //   - cutKind/cutProgress: during a CUT we hold the source frame and dissolve
  //     to the next one WITHOUT interpolating through court positions.
  const cursor = useMemo(() => timelineCursor(timeline, clock), [timeline, clock]);
  const playT = cursor.playT;
  const inCut = cursor.kind === "cut";

  // Nearest recorded frame to the continuous playback time. Drives the existing
  // stepped UI (decision panel, what-if snapshot, outcome reveal).
  const snapIdx = Math.min(nFrames - 1, Math.max(0, Math.round(playT)));
  // True when playback is sitting (within EPS) ON a recorded frame -> the live
  // path defers to the parity-exact recorded path so values never drift. A CUT
  // holds its source frame, so it counts as ON that frame (no interpolation).
  const onExactFrame = inCut || Math.abs(playT - snapIdx) < SNAP_EPS;
  const isDecisionFrame = snapIdx === p.decision_frame && onExactFrame && !inCut;
  const frame = p.frames[snapIdx];

  // CUT dissolve: dip the court toward the midpoint of the cut so the jump
  // reads as a deliberate dissolve, not a continuous glide (1 = fully visible).
  const cutOpacity = inCut
    ? 1 - Math.sin(cursor.cutProgress * Math.PI) * 0.7
    : 1;

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
  const [wiFrame, setWiFrame] = useState(snapIdx);
  if (whatIf && wiFrame !== snapIdx) {
    setWiFrame(snapIdx);
    setWiState(snapshotFrame(snapIdx));
  }

  // --- timeline playback loop (~per-frame via rAF; advances elapsed wall-clock
  // against the dwell/glide/cut timeline, scaled by `speed`; stops at the end of
  // the timeline = the decision frame's dwell). What-if mode is pause-based, so
  // the loop is inert there. ---
  const playRafRef = useRef<number | null>(null);
  const lastTsRef = useRef<number | null>(null);
  const totalMs = timeline.total;
  useEffect(() => {
    if (!playing || whatIf || paused) return;
    lastTsRef.current = null;
    const tick = (ts: number) => {
      const prev = lastTsRef.current;
      lastTsRef.current = ts;
      if (prev != null) {
        const dt = (ts - prev) * speed;
        setClock((c) => {
          const next = c + dt;
          if (next >= totalMs) {
            setPlaying(false);
            return totalMs;
          }
          return next;
        });
      }
      playRafRef.current = requestAnimationFrame(tick);
    };
    playRafRef.current = requestAnimationFrame(tick);
    return () => {
      if (playRafRef.current != null) cancelAnimationFrame(playRafRef.current);
      playRafRef.current = null;
    };
  }, [playing, speed, whatIf, paused, totalMs]);

  // Autopilot (watch mode): notify the wrapper once playback has settled at the
  // decision frame so it can show the outcome interstitial + advance. Guarded by
  // `!playing` so it fires exactly once per arrival (the rAF loop sets playing
  // false on reaching the end of the timeline, and the Explorer is remounted per
  // possession, so there is no stale "fired" state to track).
  const settledAtDecision = !playing && clock >= totalMs - SNAP_EPS;
  useEffect(() => {
    if (autoPlay && settledAtDecision && !whatIf) {
      onReachedDecision?.();
    }
  }, [autoPlay, settledAtDecision, whatIf, onReachedDecision]);

  const enterWhatIf = useCallback(() => {
    setWiState(snapshotFrame(snapIdx));
    setWiFrame(snapIdx);
    setWhatIf(true);
    setPlaying(false);
    // Lock onto the exact recorded frame when entering what-if (drag is
    // pause-based and operates on a clean recorded snapshot).
    setClock(timeForFrame(timeline, snapIdx));
  }, [snapIdx, snapshotFrame, timeline]);

  const toggleWhatIf = useCallback(() => {
    if (whatIf) {
      setWhatIf(false);
      setWiState(null);
    } else {
      enterWhatIf();
    }
  }, [whatIf, enterWhatIf]);

  const resetWhatIf = useCallback(() => {
    setWiState(snapshotFrame(snapIdx));
  }, [snapIdx, snapshotFrame]);

  // Jump to a recorded decision frame's dwell (stepping/marker click snaps to
  // exact recorded values). Clamped to the timeline span (0..decisionFrame).
  const stepTo = useCallback(
    (idx: number) => {
      setPlaying(false);
      const clamped = Math.min(p.decision_frame, Math.max(0, idx));
      setClock(timeForFrame(timeline, clamped));
    },
    [p.decision_frame, timeline],
  );

  // --- zone-FG lookups (BH + per teammate), keyed by recorded identity ---
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

  // Interpolated live snapshot for GLIDE segments only. `null` on a dwell, in a
  // cut (we hold the source frame — no positional interpolation across a clock
  // reset), and in what-if mode (those use other paths).
  const live = useMemo(() => {
    if (whatIf || onExactFrame || inCut) return null;
    const lo = Math.floor(playT);
    const hi = Math.min(nFrames - 1, lo + 1);
    const f = playT - lo;
    return interpolateFrame(p.frames[lo], p.frames[hi], f);
  }, [whatIf, onExactFrame, inCut, playT, nFrames, p.frames]);

  // --- compute Q for the displayed state ---
  const playerIds = p.entity_ids_network_order;
  const engineWi = wiState;
  const result = useMemo(() => {
    // What-if (pause-based drag) — unchanged.
    if (whatIf && engineWi) {
      const wi: WhatIfState = {
        ...engineWi,
        teammates: engineWi.teammates.map((t, i) => ({
          ...t,
          zone_fg_pct: teammateFgLookups[i](t.x, t.y),
        })),
      };
      const state = whatIfToRawState(wi, bhFgLookup);
      state.shot_clock = frame.shot_clock;
      const liveQ = runState(net, state, playerIds);
      if (model === "player") return playerChoice(liveQ.q, p.player_action);
      return liveQ;
    }
    // Live interpolated path (between 2 Hz samples) — full live recompute.
    if (live) {
      const state = liveFrameToRawState(live, bhFgLookup, teammateFgLookups);
      const liveQ = runState(net, state, playerIds);
      if (model === "player") return playerChoice(liveQ.q, p.player_action);
      return liveQ;
    }
    // Parity-exact recorded path (on a sample point): stored context verbatim.
    const state = frameToRawState(frame);
    const liveQ = runState(net, state, playerIds);
    if (model === "player") return playerChoice(liveQ.q, p.player_action);
    return liveQ;
  }, [
    whatIf,
    engineWi,
    live,
    frame,
    net,
    model,
    playerIds,
    p.player_action,
    bhFgLookup,
    teammateFgLookups,
  ]);

  // --- chip hysteresis: only switch the DISPLAYED recommendation when the new
  // argmax beats the currently-shown action by >= HYSTERESIS_EPS Q. Prevents
  // the chip flickering between near-tied actions during interpolation. The
  // Q-bars themselves always show the true live Q (no hysteresis there). On an
  // exact recorded frame / what-if we trust the argmax outright. Implemented
  // with the React "adjust state during render" recipe (compare against the
  // displayed-best STATE — no refs read during render). ---
  const [shownBest, setShownBest] = useState(result.best);
  if (shownBest !== result.best) {
    // Snap immediately (no hysteresis) in player-choice mode (best is forced to
    // the player's action), when paused on an exact recorded frame, or in
    // what-if — so stepped/recorded values are never softened.
    const forceSnap = model === "player" || (onExactFrame && !playing) || whatIf;
    const beatsByEps =
      result.q[result.best] - (result.q[shownBest] ?? -Infinity) >=
      HYSTERESIS_EPS;
    if (forceSnap || beatsByEps) setShownBest(result.best);
  }

  // --- court entities (recorded / interpolated / what-if positions) ---
  const entities: CourtEntity[] = useMemo(() => {
    const out: CourtEntity[] = [];
    const bh =
      whatIf && wiState
        ? wiState.ballHandler
        : live
          ? live.ballHandler
          : frame.ball_handler;
    out.push({
      id: "bh",
      kind: "ball-handler",
      x: bh.x,
      y: bh.y,
      label: p.entity_names_network_order[0] ?? p.ball_handler_name,
    });
    // Teammates/defenders keep recorded array order for stable identity (labels
    // + arrow targeting); interpolation matches by index, so srcIndex == array
    // index here.
    const tms =
      whatIf && wiState
        ? wiState.teammates
        : live
          ? sortBySrc(live.teammates)
          : frame.teammates;
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
      whatIf && wiState
        ? wiState.defenders
        : live
          ? sortBySrc(live.defenders)
          : frame.defenders;
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
  }, [whatIf, wiState, live, frame, p, teammateNames]);

  const ball = useMemo(() => {
    const bh =
      whatIf && wiState
        ? wiState.ballHandler
        : live
          ? live.ballHandler
          : frame.ball_handler;
    return { x: bh.x, y: bh.y };
  }, [whatIf, wiState, live, frame]);

  // Shot clock shown on the court: interpolated during live playback.
  const shotClock = live ? live.shotClock : frame.shot_clock;

  // --- agent recommendation arrow (tracks the HYSTERESIS-stabilized chip) ---
  const arrow: CourtArrow | null = useMemo(() => {
    const rec = shownBest;
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
  }, [shownBest, entities]);

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

  // recommendation + player action labels. The agent recommendation tracks the
  // hysteresis-stabilized `shownBest` so the chip and arrow stay in lockstep.
  const agentLabel = actionLabel(shownBest, teammateNames);
  const playerLabel = actionLabel(p.player_action, teammateNames);
  const agree = p.agent_action === p.player_action;

  const showOutcome = isDecisionFrame && !whatIf;
  const modeNote =
    model === "player"
      ? "Bars show the Dueling agent's Q-values; the highlighted action is what the player actually did."
      : `Q-values from the ${MODEL_LABELS[model]} model, computed live in your browser.`;

  return (
    <div className="explorer">
      {topSlot ?? (
        <button className="explorer__back" onClick={onBack}>
          ← All possessions
        </button>
      )}

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
            {inCut && (
              <span className="court-cut mono">·· later in the possession</span>
            )}
            <div className="court-clock">
              <span className="court-clock__val">{shotClock.toFixed(1)}</span>
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
              showNames={names}
              fade={cutOpacity}
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
              aria-label="Previous frame"
              disabled={whatIf || nFrames <= 1}
              onClick={() => stepTo(snapIdx - 1)}
              title="Step to previous recorded frame"
            >
              ⏮
            </button>
            <button
              className="iconbtn"
              aria-label={playing ? "Pause" : "Play"}
              disabled={whatIf || nFrames <= 1}
              onClick={() => {
                // Rewind to the top if parked at the end of the timeline.
                if (clock >= totalMs - SNAP_EPS) setClock(0);
                setPlaying((v) => !v);
              }}
            >
              {playing ? "❚❚" : "►"}
            </button>
            <button
              className="iconbtn"
              aria-label="Next frame"
              disabled={whatIf || nFrames <= 1}
              onClick={() => stepTo(snapIdx + 1)}
              title="Step to next recorded frame"
            >
              ⏭
            </button>
            <div className="scrub">
              <div className="scrub__track-wrap">
                <input
                  type="range"
                  min={0}
                  max={totalMs}
                  step={1}
                  value={clock}
                  disabled={whatIf || nFrames <= 1}
                  onChange={(e) => {
                    setPlaying(false);
                    const raw = Number(e.target.value);
                    // Snap affordance: releasing near a recorded-frame marker
                    // locks onto that frame's dwell (exact-parity values).
                    const snapWindow = totalMs * SCRUB_SNAP_FRAC;
                    let nextClock = raw;
                    for (let i = 0; i <= p.decision_frame; i++) {
                      const ft = timeForFrame(timeline, i);
                      if (Math.abs(raw - ft) < snapWindow) {
                        nextClock = ft;
                        break;
                      }
                    }
                    setClock(nextClock);
                  }}
                  aria-label="Scrub possession (decision frames are held; movement between them is compressed)"
                />
                {/* decision-frame markers — each recorded frame in the timeline
                    is a clickable marker placed at the START of its dwell. The
                    decision frame is accented; a beat dot precedes each CUT. */}
                <div className="scrub__markers" aria-hidden="true">
                  {Array.from({ length: p.decision_frame + 1 }, (_, i) => {
                    const pct =
                      totalMs > 0
                        ? (timeForFrame(timeline, i) / totalMs) * 100
                        : 0;
                    return (
                      <button
                        key={i}
                        type="button"
                        tabIndex={-1}
                        className={
                          "scrub__marker" +
                          (i === p.decision_frame ? " is-decision" : "") +
                          (i === snapIdx && onExactFrame ? " is-current" : "")
                        }
                        style={{ left: `${pct}%` }}
                        onClick={() => stepTo(i)}
                        title={
                          i === p.decision_frame
                            ? "Decision point — click to snap"
                            : `Frame ${i + 1} — click to snap`
                        }
                      />
                    );
                  })}
                  {/* CUT beat markers — a faint mono dot where the play jumps
                      (shot-clock reset / long gap) so the discontinuity reads. */}
                  {timeline.segments
                    .filter((s) => s.kind === "cut")
                    .map((s) => {
                      const mid = s.start + s.duration / 2;
                      const pct = totalMs > 0 ? (mid / totalMs) * 100 : 0;
                      return (
                        <span
                          key={`cut-${s.from}`}
                          className="scrub__beat"
                          style={{ left: `${pct}%` }}
                          title="·· later in the possession"
                        />
                      );
                    })}
                </div>
              </div>
              <span className="scrub__pos">
                {snapIdx + 1}/{nFrames}
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
            <label className="switch switch--names">
              <input
                type="checkbox"
                checked={names}
                onChange={() => setNames((v) => !v)}
              />
              <span className="switch__track">
                <span className="switch__knob" />
              </span>
              Names
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
                : inCut
                  ? "·· later in the possession"
                  : onExactFrame
                    ? `Frame ${snapIdx + 1} — held`
                    : "Live — moving to the next moment"}
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
            <p className="live-caption mono">
              live agent — re-evaluated as the play moves; decision moments are
              held, movement between them is compressed, and jumps in the play
              (·· later in the possession) cut rather than glide
            </p>
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
              best={shownBest}
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

/** Order interpolated entities back into recorded array order for stable ids. */
function sortBySrc<T extends { srcIndex: number }>(xs: T[]): T[] {
  return [...xs].sort((a, b) => a.srcIndex - b.srcIndex);
}

function groupLabel(cat: string): string {
  if (cat === "declined_the_shot") return "Declined the shot";
  if (cat === "wanted_the_shot") return "Wanted the shot";
  if (cat === "stream") return "Test-set possession";
  return "Agreement";
}
