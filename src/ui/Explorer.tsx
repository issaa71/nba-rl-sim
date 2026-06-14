// Explorer view — the heart. Court replay + scrubber + live Q-values +
// outcome reveal + what-if dragging. Everything is computed in-browser via the
// (parity-tested) engine.
//
// PLAYBACK IS TRUE REAL-TIME. When a possession has high-frequency tracking
// (engine/tracking.ts), pressing PLAY runs a wall-clock head over the real
// SportVU frames: ~12.5 fps frames 80 ms apart, the head linearly interpolates
// between the two ADJACENT frames it sits between, so the motion is genuine.
// The speed control scales the clock. The real ball renders with a subtle
// shot-arc when its height rises.
//
// LIVE AGENT. Between the recorded decision points the agent is re-evaluated on
// the tracking frames at ~4 Hz (and immediately on pause / scrub): interpolate
// -> re-sort to canonical order -> re-lookup zone FG% -> deriveContext ->
// buildFeatures -> forward. AT a recorded decision snap (tracking index ==
// decision_snap_indices[k]) the UI hands off to the PARITY-EXACT recorded path
// (frameToRawState on possessions.frames[k]) so the displayed Q never drifts
// from the stored agent_q_values. Decision markers sit on the scrubber at those
// moments; clicking one snaps to it.
//
// FALLBACK. A possession WITHOUT tracking (the export dropped a few) falls back
// to the old stepped, low-rate decision-frame view — clearly labelled.

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
  frameToRawState,
  playerChoice,
  runState,
  whatIfToRawState,
  type QResult,
  type WhatIfState,
} from "../engine/explorer";
import {
  foldRawToFeature,
  nearestTrackFrame,
  rawSnapshotAt,
  snapshotAt,
  timeForTrackFrame,
  trackSnapshotToRawState,
  trackSpan,
  type RawTrackSnapshot,
  type TrackSnapshot,
} from "../engine/tracking";
import { BASKET_X, BASKET_Y } from "../engine/features";
import type { AppData } from "../data/load";
import { makeZoneFgLookup } from "../data/load";
import { actionLabel, type Possession, type TrackingPossession } from "../data/types";
import { Court, type CourtArrow, type CourtEntity } from "./Court";
import { QBars } from "./QBars";
import { ModelToggle } from "./bits";
import { MODEL_LABELS, outcomeText, type ModelMode } from "./model";

const SPEEDS = [0.5, 1, 2] as const;

// Half-court line in the RAW full-court frame (ft). The frontcourt gate: the
// live agent only evaluates once the ball-handler crosses into the offense's
// attacking half (mirrors the original app.py:422-426 build_state gate).
const HALF_COURT_X = 47;
// Raw full-court basket positions (ft) for the "shoot" cue target.
const RAW_BASKET_RIGHT = { x: 88.75, y: 25 };
const RAW_BASKET_LEFT = { x: 5.25, y: 25 };
// Neutral Q result shown before the agent's first frontcourt evaluation.
const NEUTRAL_RESULT = { q: [0, 0, 0, 0, 0], best: -1 };

// Live-agent evaluation cadence between decision snaps (Hz). The live state is
// recomputed when the playback clock crosses one of these buckets — and always
// immediately on pause / scrub / snap.
const LIVE_HZ = 4;

// Clicking within this fraction of the scrubber range to a decision marker
// snaps the playback head onto that decision (exact-parity recorded values).
const SCRUB_SNAP_FRAC = 0.015;

// A wall-time within this many seconds of a decision snap's frame time is
// treated as sitting ON that snap -> use the parity-exact recorded path.
const SNAP_EPS_S = 0.02;

// Chip hysteresis: while interpolating between decision points the live argmax
// can flicker between near-tied actions. Only switch the displayed
// recommendation when the new argmax beats the shown action by this much Q.
const HYSTERESIS_EPS = 0.01;

interface ExplorerProps {
  possession: Possession;
  /** Real high-frequency tracking for this possession, if available. */
  tracking?: TrackingPossession | null;
  /**
   * The SAME tracking in RAW full-court coordinates (drives the full-court
   * render). Null/absent -> the explorer renders the half-court view.
   */
  trackingFc?: TrackingPossession | null;
  data: AppData;
  model: ModelMode;
  onModelChange: (m: ModelMode) => void;
  onBack: () => void;
  /**
   * Autopilot (watch mode): start playback from the top of the possession on
   * mount and replay continuously. Defaults to false (manual experience).
   */
  autoPlay?: boolean;
  /**
   * Fired once when continuous playback reaches the FINAL recorded decision
   * moment. Watch mode uses this to show the outcome interstitial; playback
   * keeps running through the post-decision outcome flight underneath it.
   */
  onReachedDecision?: () => void;
  /**
   * External pause (watch mode). When true the playback loop is held even if
   * internally "playing" — lets the autopilot freeze the action while still
   * allowing what-if dragging on the held frame.
   */
  paused?: boolean;
  /**
   * Replaces the default back button row with custom chrome (watch-mode bar).
   */
  topSlot?: ReactNode;
}

/** Short labels for the 5 actions from the possession's teammate names. */
function shortLabels(teammateNames: string[]): string[] {
  return ["Shoot", ...teammateNames.map((n) => firstName(n) ?? "Pass")];
}
function firstName(full: string): string | undefined {
  if (!full) return undefined;
  const parts = full.split(" ");
  return parts.length > 1 ? parts[parts.length - 1] : full;
}

export function Explorer({
  possession: p,
  tracking,
  trackingFc,
  data,
  model,
  onModelChange,
  onBack,
  autoPlay = false,
  onReachedDecision,
  paused = false,
  topSlot,
}: ExplorerProps) {
  // A possession with at least 2 tracking frames gets the real-time path; a
  // missing / degenerate one falls back to the stepped low-rate view.
  const hasTracking = !!tracking && tracking.frames.length >= 2;

  if (hasTracking && tracking) {
    return (
      <TrackedExplorer
        p={p}
        tracking={tracking}
        trackingFc={trackingFc ?? null}
        data={data}
        model={model}
        onModelChange={onModelChange}
        onBack={onBack}
        autoPlay={autoPlay}
        onReachedDecision={onReachedDecision}
        paused={paused}
        topSlot={topSlot}
      />
    );
  }
  return (
    <SteppedExplorer
      p={p}
      data={data}
      model={model}
      onModelChange={onModelChange}
      onBack={onBack}
      onReachedDecision={onReachedDecision}
      autoPlay={autoPlay}
      topSlot={topSlot}
    />
  );
}

// ===========================================================================
// Real-time tracked explorer
// ===========================================================================

interface TrackedProps {
  p: Possession;
  tracking: TrackingPossession;
  trackingFc: TrackingPossession | null;
  data: AppData;
  model: ModelMode;
  onModelChange: (m: ModelMode) => void;
  onBack: () => void;
  autoPlay: boolean;
  onReachedDecision?: () => void;
  paused: boolean;
  topSlot?: ReactNode;
}

function TrackedExplorer({
  p,
  tracking,
  trackingFc,
  data,
  model,
  onModelChange,
  onBack,
  autoPlay,
  onReachedDecision,
  paused,
  topSlot,
}: TrackedProps) {
  const span = useMemo(() => trackSpan(tracking), [tracking]);
  const snaps = tracking.decision_snap_indices;
  // The recorded DECISION point (where the outcome reveal fires). It is the
  // possession's decision_frame-th recorded decision; map it to its tracking
  // frame. (decision_frame is NOT always the last recorded frame — outcome
  // frames can follow it.)
  const decisionSnapIdx = snaps[p.decision_frame] ?? snaps[snaps.length - 1] ?? 0;
  const decisionTime = useMemo(
    () => timeForTrackFrame(tracking, decisionSnapIdx),
    [tracking, decisionSnapIdx],
  );

  // Source of truth: wall-clock time (s) along the tracking span. Manual mode
  // opens parked on the decision; autopilot starts at the top of the span.
  const [clock, setClock] = useState(() => (autoPlay ? span.start : decisionTime));
  const [playing, setPlaying] = useState(autoPlay);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>(1);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [names, setNames] = useState(true);

  const [whatIf, setWhatIf] = useState(false);
  const [wiState, setWiState] = useState<WhatIfState | null>(null);
  const rafRef = useRef<number | null>(null);
  const pendingRef = useRef<WhatIfState | null>(null);

  const net: LoadedNetwork = model === "dqn" ? data.dqn : data.dueling;
  const teammateNames = p.entity_names_network_order.slice(1);
  const labels = useMemo(() => shortLabels(teammateNames), [teammateNames]);

  // Nearest tracking frame to the head, and whether it sits ON a decision snap.
  const nearIdx = useMemo(
    () => nearestTrackFrame(tracking, clock),
    [tracking, clock],
  );
  // Which recorded decision (0..snaps.length-1) is at the nearest frame, if the
  // head is within SNAP_EPS_S of it.
  const recordedDecision = useMemo(() => {
    for (let k = 0; k < snaps.length; k++) {
      if (
        nearIdx === snaps[k] &&
        Math.abs(timeForTrackFrame(tracking, snaps[k]) - clock) <= SNAP_EPS_S
      ) {
        return k;
      }
    }
    return -1;
  }, [snaps, nearIdx, tracking, clock]);
  const onSnap = recordedDecision >= 0;
  const isDecisionFrame = recordedDecision === p.decision_frame && !whatIf;

  // Live-eval key. During PLAYBACK the agent re-evaluates at ~LIVE_HZ (the clock
  // is bucketed so the network forward pass runs a few times a second, not every
  // animation frame). When PAUSED or SCRUBBING the exact clock is used so the Q
  // updates immediately as the user moves the head. On a snap the recorded path
  // is used regardless (handled in `result`).
  const liveBucket = useMemo(
    () => (playing ? Math.round(clock * LIVE_HZ) : clock),
    [playing, clock],
  );

  // The rendered snapshot (feature frame). Recomputed every clock change for
  // smooth motion; the agent eval below is throttled separately via liveBucket.
  const snapshot: TrackSnapshot = useMemo(
    () => snapshotAt(tracking, clock),
    [tracking, clock],
  );

  // --- full-court render twin (raw coords). The agent NEVER reads this; it
  // folds render positions back to the feature frame via `toFeature`. Null ->
  // render the half-court view from the feature `snapshot` (fallback). ---
  const fullCourt = !!trackingFc && trackingFc.frames.length >= 2;
  const attackingRight = trackingFc?.attacking_right ?? true;
  const rawSnap: RawTrackSnapshot | null = useMemo(
    () => (fullCourt && trackingFc ? rawSnapshotAt(trackingFc, clock) : null),
    [fullCourt, trackingFc, clock],
  );
  // Positions handed to the renderer: raw full-court when available, else the
  // half-court feature positions.
  const renderPlayers: { x: number; y: number }[] = useMemo(
    () => (rawSnap ? rawSnap.players : snapshot.players.map((q) => ({ x: q.x, y: q.y }))),
    [rawSnap, snapshot],
  );
  // Map a RENDER position back to the agent's feature frame (identity off the
  // full-court path; the verified fold on it).
  const toFeature = useCallback(
    (x: number, y: number) =>
      fullCourt ? foldRawToFeature(x, y, attackingRight) : { x, y },
    [fullCourt, attackingRight],
  );
  // Frontcourt gate: the live agent only evaluates once the ball-handler is in
  // the offense's attacking half (raw frame). Always true off the full-court
  // path (the feature frame is already canonical frontcourt).
  const bhInFrontcourt = useMemo(() => {
    if (!fullCourt) return true;
    const bx = renderPlayers[0]?.x ?? HALF_COURT_X;
    return attackingRight ? bx >= HALF_COURT_X : bx <= HALF_COURT_X;
  }, [fullCourt, renderPlayers, attackingRight]);

  // Snapshot the draggable positions for the current frame: RENDER positions
  // (raw when full-court) carry the displayed coords; velocities come from the
  // feature snapshot (frame-consistent; unchanged by a static drag).
  const snapshotForWhatIf = useCallback(
    (): WhatIfState => ({
      ballHandler: {
        x: renderPlayers[0]?.x ?? 0,
        y: renderPlayers[0]?.y ?? 0,
        vx: snapshot.players[0]?.vx ?? 0,
        vy: snapshot.players[0]?.vy ?? 0,
      },
      teammates: renderPlayers.slice(1, 5).map((t, i) => ({
        x: t.x,
        y: t.y,
        vx: snapshot.players[1 + i]?.vx ?? 0,
        vy: snapshot.players[1 + i]?.vy ?? 0,
        zone_fg_pct: 0, // recomputed via the per-teammate FG lookup
      })),
      defenders: renderPlayers.slice(5, 10).map((d, i) => ({
        x: d.x,
        y: d.y,
        vx: snapshot.players[5 + i]?.vx ?? 0,
        vy: snapshot.players[5 + i]?.vy ?? 0,
      })),
    }),
    [renderPlayers, snapshot],
  );

  // --- real-time playback loop. Advances the wall-clock by elapsed * speed.
  // Stops at the END of the tracking span (outcome flight included). What-if is
  // pause-based so the loop is inert there. ---
  const playRafRef = useRef<number | null>(null);
  const lastTsRef = useRef<number | null>(null);
  useEffect(() => {
    if (!playing || whatIf || paused) return;
    lastTsRef.current = null;
    const tick = (ts: number) => {
      const prev = lastTsRef.current;
      lastTsRef.current = ts;
      if (prev != null) {
        const dt = ((ts - prev) / 1000) * speed; // ms -> s, scaled
        setClock((c) => {
          const next = c + dt;
          if (next >= span.end) {
            setPlaying(false);
            return span.end;
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
  }, [playing, speed, whatIf, paused, span.end]);

  // Autopilot (watch mode): notify the wrapper ONCE when playback first crosses
  // the final decision moment. Playback keeps running underneath through the
  // outcome flight (the interstitial overlays it). Guarded by a ref so it fires
  // exactly once per possession mount.
  const firedRef = useRef(false);
  useEffect(() => {
    if (!autoPlay || whatIf) return;
    if (!firedRef.current && clock >= decisionTime - SNAP_EPS_S) {
      firedRef.current = true;
      onReachedDecision?.();
    }
  }, [autoPlay, whatIf, clock, decisionTime, onReachedDecision]);

  const enterWhatIf = useCallback(() => {
    setWiState(snapshotForWhatIf());
    setWhatIf(true);
    setPlaying(false);
  }, [snapshotForWhatIf]);

  const toggleWhatIf = useCallback(() => {
    if (whatIf) {
      setWhatIf(false);
      setWiState(null);
    } else {
      enterWhatIf();
    }
  }, [whatIf, enterWhatIf]);

  const resetWhatIf = useCallback(() => {
    setWiState(snapshotForWhatIf());
  }, [snapshotForWhatIf]);

  // Jump the head to a recorded decision (marker click / step). Snaps to that
  // decision's exact frame time so the parity-exact recorded path engages.
  const stepToDecision = useCallback(
    (k: number) => {
      setPlaying(false);
      const clamped = Math.min(snaps.length - 1, Math.max(0, k));
      setClock(timeForTrackFrame(tracking, snaps[clamped]));
    },
    [snaps, tracking],
  );

  // The decision index nearest the head (for the step buttons / panel chrome).
  const nearestDecisionK = useMemo(() => {
    let best = 0;
    let bestD = Infinity;
    for (let k = 0; k < snaps.length; k++) {
      const d = Math.abs(timeForTrackFrame(tracking, snaps[k]) - clock);
      if (d < bestD) {
        bestD = d;
        best = k;
      }
    }
    return best;
  }, [snaps, tracking, clock]);

  // --- zone-FG lookups (BH + per teammate), keyed by recorded identity ---
  // Player identities are stable across a possession, so any recorded frame's
  // compact ids work; use the decision frame for a fixed, well-defined source.
  const idFrame = p.frames[p.decision_frame];
  const bhFgLookup = useMemo(
    () => makeZoneFgLookup(data.zoneFg, idFrame.ball_handler.compact_id),
    [data.zoneFg, idFrame.ball_handler.compact_id],
  );
  const teammateFgLookups = useMemo(
    () => idFrame.teammates.map((t) => makeZoneFgLookup(data.zoneFg, t.compact_id)),
    [data.zoneFg, idFrame.teammates],
  );

  // Fallback shot clock when the tracking clock is missing (network input).
  const fallbackShotClock = p.frames[nearestDecisionK]?.shot_clock ?? 14;

  // --- compute Q for the displayed state ---
  // The live recompute is THROTTLED to ~LIVE_HZ during playback via `liveBucket`
  // (it omits `snapshot` from the deps deliberately — the snapshot is read fresh
  // from the closure when the memo does run). `snapshot.shotClock` is therefore
  // also read inside the memo, never as a per-frame dependency, so the throttle
  // holds. On a snap the recorded path is exact; when paused/scrubbing
  // `liveBucket === clock` so the memo recomputes immediately.
  const playerIds = p.entity_ids_network_order;
  // Compute a FRESH recommendation only when the agent should evaluate this
  // frame: what-if, a recorded snap, or the ball-handler in the frontcourt. In
  // the backcourt this returns null and the display carries the last result
  // (the frontcourt gate, mirroring app.py:422-426 + 725).
  const agentShouldEval = whatIf || onSnap || bhInFrontcourt;
  const freshResult = useMemo<QResult | null>(() => {
    if (!agentShouldEval) return null;
    const shotClock = snapshot.shotClock ?? fallbackShotClock;
    // What-if (pause-based drag). wiState positions are in the RENDER frame
    // (raw when full-court) — fold each back to the feature frame the network
    // reasons in. `toFeature` is identity off the full-court path, so the
    // half-court fallback is unchanged.
    if (whatIf && wiState) {
      const bhF = toFeature(wiState.ballHandler.x, wiState.ballHandler.y);
      const wi: WhatIfState = {
        ballHandler: { x: bhF.x, y: bhF.y, vx: wiState.ballHandler.vx, vy: wiState.ballHandler.vy },
        teammates: wiState.teammates.map((t, i) => {
          const f = toFeature(t.x, t.y);
          return {
            x: f.x,
            y: f.y,
            vx: t.vx,
            vy: t.vy,
            zone_fg_pct: (teammateFgLookups[i] ?? (() => 0))(f.x, f.y),
          };
        }),
        defenders: wiState.defenders.map((d) => {
          const f = toFeature(d.x, d.y);
          return { x: f.x, y: f.y, vx: d.vx, vy: d.vy };
        }),
      };
      const state = whatIfToRawState(wi, bhFgLookup);
      state.shot_clock = shotClock;
      const liveQ = runState(net, state, playerIds);
      return model === "player" ? playerChoice(liveQ.q, p.player_action) : liveQ;
    }
    // Parity-exact recorded path AT a recorded decision snap: stored context
    // verbatim — reproduces the stored agent_q_values exactly. (Decision snaps
    // are shown regardless of the frontcourt gate — they ARE the lesson.)
    if (onSnap) {
      const state = frameToRawState(p.frames[recordedDecision]);
      const liveQ = runState(net, state, playerIds);
      return model === "player" ? playerChoice(liveQ.q, p.player_action) : liveQ;
    }
    // Live tracking path (between decisions) — full live recompute on the
    // interpolated tracking snapshot at ~LIVE_HZ (keyed by liveBucket).
    const state = trackSnapshotToRawState(
      snapshot,
      shotClock,
      bhFgLookup,
      teammateFgLookups,
    );
    const liveQ = runState(net, state, playerIds);
    return model === "player" ? playerChoice(liveQ.q, p.player_action) : liveQ;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    agentShouldEval,
    whatIf,
    wiState,
    onSnap,
    recordedDecision,
    liveBucket,
    toFeature,
    net,
    model,
    playerIds,
    p,
    bhFgLookup,
    teammateFgLookups,
    fallbackShotClock,
  ]);

  // Carry the last fresh recommendation forward across backcourt frames, via the
  // "adjust state during render" recipe (same as the hysteresis below) — no
  // effect, no ref read during render. freshResult is a stable memo, so this
  // converges in one extra render and never loops.
  const [carried, setCarried] = useState<QResult | null>(null);
  if (freshResult && freshResult !== carried) setCarried(freshResult);

  // Displayed result: fresh when evaluating, else the carried prior, else
  // NEUTRAL (best = -1 -> the agent reads as inactive in the UI).
  const result: QResult = freshResult ?? carried ?? NEUTRAL_RESULT;
  const agentActive = result.best >= 0;

  // --- chip hysteresis (adjust-state-during-render recipe) ---
  // Only track REAL actions (best >= 0); a NEUTRAL result (backcourt, no carry)
  // never overwrites the shown action — the UI hides it via `agentActive`.
  const [shownBest, setShownBest] = useState(result.best);
  if (shownBest !== result.best && result.best >= 0) {
    const forceSnap =
      shownBest < 0 || model === "player" || (onSnap && !playing) || whatIf;
    const beatsByEps =
      result.q[result.best] - (result.q[shownBest] ?? -Infinity) >=
      HYSTERESIS_EPS;
    if (forceSnap || beatsByEps) setShownBest(result.best);
  }

  // --- court entities (recorded identity order; network order for stable ids)
  const entities: CourtEntity[] = useMemo(() => {
    const out: CourtEntity[] = [];
    const players =
      whatIf && wiState
        ? [
            wiState.ballHandler,
            ...wiState.teammates,
            ...wiState.defenders,
          ]
        : renderPlayers;
    const bh = players[0];
    if (bh) {
      out.push({
        id: "bh",
        kind: "ball-handler",
        x: bh.x,
        y: bh.y,
        label: p.entity_names_network_order[0] ?? p.ball_handler_name,
      });
    }
    // Teammates: network slots 1..4 -> players[1..4].
    for (let i = 0; i < 4; i++) {
      const t = players[1 + i];
      if (!t) continue;
      out.push({
        id: `tm${i}`,
        kind: "teammate",
        x: t.x,
        y: t.y,
        slot: i + 1,
        label: teammateNames[i] ?? `Teammate ${i + 1}`,
      });
    }
    // Defenders: players[5..9] — ALL FIVE render every frame.
    for (let i = 0; i < 5; i++) {
      const d = players[5 + i];
      if (!d) continue;
      out.push({
        id: `df${i}`,
        kind: "defender",
        x: d.x,
        y: d.y,
        label: `Defender ${i + 1}`,
      });
    }
    return out;
  }, [whatIf, wiState, renderPlayers, p, teammateNames]);

  const ball = useMemo(() => {
    if (whatIf && wiState) {
      return { x: wiState.ballHandler.x, y: wiState.ballHandler.y };
    }
    return rawSnap ? rawSnap.ball : snapshot.ball;
  }, [whatIf, wiState, rawSnap, snapshot]);
  const ballZ = whatIf ? 0 : rawSnap ? rawSnap.ballZ : snapshot.ballZ;

  const shotClock = snapshot.shotClock;

  // --- restrained recommendation cue (tracks the hysteresis-stabilized chip) --
  const arrow: CourtArrow | null = useMemo(() => {
    if (!agentActive) return null; // no cue until the agent is evaluating
    const rec = shownBest;
    const bh = entities.find((e) => e.id === "bh");
    if (!bh) return null;
    if (rec === 0) {
      // "shoot" cue points at the real attacked basket (raw frame) or the
      // canonical right basket in the half-court fallback.
      const basket = fullCourt
        ? attackingRight
          ? RAW_BASKET_RIGHT
          : RAW_BASKET_LEFT
        : { x: BASKET_X, y: BASKET_Y };
      return { fromX: bh.x, fromY: bh.y, toX: basket.x, toY: basket.y, kind: "shoot" };
    }
    const target = entities.find((e) => e.id === `tm${rec - 1}`);
    if (!target) return null;
    return { fromX: bh.x, fromY: bh.y, toX: target.x, toY: target.y, kind: "pass" };
  }, [agentActive, shownBest, entities, fullCourt, attackingRight]);

  // --- drag handler: coalesce rapid pointer-moves to one commit per frame ---
  const applyDrag = useCallback(
    (base: WhatIfState, id: string, x: number, y: number): WhatIfState => {
      if (id === "bh") return { ...base, ballHandler: { ...base.ballHandler, x, y } };
      if (id.startsWith("tm")) {
        const i = Number(id.slice(2));
        return {
          ...base,
          teammates: base.teammates.map((t, j) => (j === i ? { ...t, x, y } : t)),
        };
      }
      if (id.startsWith("df")) {
        const i = Number(id.slice(2));
        return {
          ...base,
          defenders: base.defenders.map((d, j) => (j === i ? { ...d, x, y } : d)),
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
      if (rafRef.current != null) return;
      rafRef.current = requestAnimationFrame(() => {
        rafRef.current = null;
        if (pendingRef.current) setWiState(pendingRef.current);
        pendingRef.current = null;
      });
    },
    [wiState, applyDrag],
  );

  useEffect(() => {
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  const agentLabel = agentActive ? actionLabel(shownBest, teammateNames) : "—";
  const playerLabel = actionLabel(p.player_action, teammateNames);
  const agree = p.agent_action === p.player_action;

  const showOutcome = isDecisionFrame && !whatIf;
  const modeNote =
    model === "player"
      ? "Bars show the Dueling agent's Q-values; the highlighted action is what the player actually did."
      : `Q-values from the ${MODEL_LABELS[model]} model, computed live in your browser.`;

  const atEnd = clock >= span.end - SNAP_EPS_S;

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
            <div className="court-clock">
              <span className="court-clock__val">
                {shotClock != null ? shotClock.toFixed(1) : "—"}
              </span>
              <span className="court-clock__label">shot clock</span>
            </div>
            <Court
              entities={entities}
              ball={ball}
              ballZ={ballZ}
              arrow={arrow}
              draggable={whatIf}
              activeId={activeId}
              onActiveChange={setActiveId}
              onDrag={onDrag}
              showNames={names}
              fullCourt={fullCourt}
            />
          </div>

          <div className="legend">
            <span>
              <i style={{ background: "#047857" }} /> Offense
            </span>
            <span>
              <i style={{ background: "#047857", boxShadow: "0 0 0 2px #022c22" }} />{" "}
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
              aria-label="Previous decision"
              disabled={whatIf || snaps.length <= 1}
              onClick={() => stepToDecision(nearestDecisionK - 1)}
              title="Step to previous decision point"
            >
              ⏮
            </button>
            <button
              className="iconbtn"
              aria-label={playing ? "Pause" : "Play"}
              disabled={whatIf}
              onClick={() => {
                if (atEnd) setClock(span.start);
                setPlaying((v) => !v);
              }}
            >
              {playing ? "❚❚" : "►"}
            </button>
            <button
              className="iconbtn"
              aria-label="Next decision"
              disabled={whatIf || snaps.length <= 1}
              onClick={() => stepToDecision(nearestDecisionK + 1)}
              title="Step to next decision point"
            >
              ⏭
            </button>
            <div className="scrub">
              <div className="scrub__track-wrap">
                <input
                  type="range"
                  min={span.start}
                  max={span.end}
                  step={0.01}
                  value={clock}
                  disabled={whatIf}
                  onChange={(e) => {
                    setPlaying(false);
                    const raw = Number(e.target.value);
                    // Snap affordance: releasing near a decision marker locks
                    // onto that decision's exact frame time (parity values).
                    const dur = span.end - span.start || 1;
                    const snapWindow = dur * SCRUB_SNAP_FRAC;
                    let nextClock = raw;
                    for (let k = 0; k < snaps.length; k++) {
                      const ft = timeForTrackFrame(tracking, snaps[k]);
                      if (Math.abs(raw - ft) < snapWindow) {
                        nextClock = ft;
                        break;
                      }
                    }
                    setClock(nextClock);
                  }}
                  aria-label="Scrub possession in real time; decision points are marked"
                />
                {/* decision markers — one per recorded decision, placed at its
                    real wall-time. The final decision is accented. */}
                <div className="scrub__markers" aria-hidden="true">
                  {snaps.map((s, k) => {
                    const dur = span.end - span.start || 1;
                    const pct =
                      ((timeForTrackFrame(tracking, s) - span.start) / dur) * 100;
                    return (
                      <button
                        key={k}
                        type="button"
                        tabIndex={-1}
                        className={
                          "scrub__marker" +
                          (k === p.decision_frame ? " is-decision" : "") +
                          (k === recordedDecision ? " is-current" : "")
                        }
                        style={{ left: `${pct}%` }}
                        onClick={() => stepToDecision(k)}
                        title={
                          k === p.decision_frame
                            ? "Decision point — click to snap"
                            : `Decision ${k + 1} — click to snap`
                        }
                      />
                    );
                  })}
                </div>
              </div>
              <span className="scrub__pos">
                {(clock - span.start).toFixed(1)}s
              </span>
            </div>
            <div className="segment" aria-label="Playback speed">
              {SPEEDS.map((s) => (
                <button key={s} aria-pressed={speed === s} onClick={() => setSpeed(s)}>
                  {s}×
                </button>
              ))}
            </div>
          </div>

          {/* what-if controls */}
          <div className="whatif-bar">
            <label className="switch">
              <input type="checkbox" checked={whatIf} onChange={toggleWhatIf} />
              <span className="switch__track">
                <span className="switch__knob" />
              </span>
              What-if mode
            </label>
            <label className="switch switch--names">
              <input type="checkbox" checked={names} onChange={() => setNames((v) => !v)} />
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
                Pause anywhere, then drag players to test alternatives.
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
                : onSnap
                  ? `Decision ${recordedDecision + 1} — recorded`
                  : agentActive
                    ? "Live — agent tracking the play"
                    : "Bringing it up — agent waits for the frontcourt"}
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
              real-time playback over the tracking — the live agent re-evaluates
              as the play moves; recorded decision points show the exact stored
              Q-values
            </p>
            {!whatIf && agree && isDecisionFrame && (
              <p className="agree-note">
                Agreement — the agent and {p.ball_handler_name} made the same call
                here.
              </p>
            )}
          </div>

          <div className="panel__block">
            <span className="panel__label">Action values (Q)</span>
            {agentActive ? (
              <>
                <QBars
                  q={result.q}
                  best={shownBest}
                  labels={labels}
                  playerAction={isDecisionFrame ? p.player_action : undefined}
                />
                <p className="agree-note">{modeNote}</p>
              </>
            ) : (
              <p className="live-caption mono">
                the agent starts evaluating once the offense crosses half-court
                into its attacking half
              </p>
            )}
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

// ===========================================================================
// Stepped fallback (no tracking) — the old low-rate decision-frame view.
// ===========================================================================

interface SteppedProps {
  p: Possession;
  data: AppData;
  model: ModelMode;
  onModelChange: (m: ModelMode) => void;
  onBack: () => void;
  onReachedDecision?: () => void;
  autoPlay: boolean;
  topSlot?: ReactNode;
}

function SteppedExplorer({
  p,
  data,
  model,
  onModelChange,
  onBack,
  onReachedDecision,
  autoPlay,
  topSlot,
}: SteppedProps) {
  const nFrames = p.frames.length;
  // No real-time playback here (no tracking) — the fallback opens parked on the
  // decision frame in both manual and autopilot modes.
  const [idx, setIdx] = useState(p.decision_frame);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [names, setNames] = useState(true);

  const net: LoadedNetwork = model === "dqn" ? data.dqn : data.dueling;
  const teammateNames = p.entity_names_network_order.slice(1);
  const labels = useMemo(() => shortLabels(teammateNames), [teammateNames]);
  const frame = p.frames[idx];
  const isDecisionFrame = idx === p.decision_frame;

  // Autopilot (watch mode): report the decision once so the wrapper can show
  // the interstitial + advance. The fallback has no flight, so a short linger
  // before reporting reads as a beat.
  const firedRef = useRef(false);
  useEffect(() => {
    if (autoPlay && !firedRef.current) {
      firedRef.current = true;
      const t = window.setTimeout(() => onReachedDecision?.(), 900);
      return () => window.clearTimeout(t);
    }
  }, [autoPlay, onReachedDecision]);

  const result = useMemo(() => {
    const state = frameToRawState(frame);
    const liveQ = runState(net, state, p.entity_ids_network_order);
    if (model === "player") return playerChoice(liveQ.q, p.player_action);
    return liveQ;
  }, [frame, net, model, p.entity_ids_network_order, p.player_action]);

  const entities: CourtEntity[] = useMemo(() => {
    const out: CourtEntity[] = [];
    out.push({
      id: "bh",
      kind: "ball-handler",
      x: frame.ball_handler.x,
      y: frame.ball_handler.y,
      label: p.entity_names_network_order[0] ?? p.ball_handler_name,
    });
    frame.teammates.forEach((t, i) =>
      out.push({
        id: `tm${i}`,
        kind: "teammate",
        x: t.x,
        y: t.y,
        slot: i + 1,
        label: teammateNames[i] ?? `Teammate ${i + 1}`,
      }),
    );
    frame.defenders.forEach((d, i) =>
      out.push({
        id: `df${i}`,
        kind: "defender",
        x: d.x,
        y: d.y,
        label: `Defender ${i + 1}`,
      }),
    );
    return out;
  }, [frame, p, teammateNames]);

  const arrow: CourtArrow | null = useMemo(() => {
    const bh = entities.find((e) => e.id === "bh");
    if (!bh) return null;
    if (result.best === 0) {
      return { fromX: bh.x, fromY: bh.y, toX: BASKET_X, toY: BASKET_Y, kind: "shoot" };
    }
    const target = entities.find((e) => e.id === `tm${result.best - 1}`);
    if (!target) return null;
    return { fromX: bh.x, fromY: bh.y, toX: target.x, toY: target.y, kind: "pass" };
  }, [result.best, entities]);

  const agentLabel = actionLabel(result.best, teammateNames);
  const playerLabel = actionLabel(p.player_action, teammateNames);

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
        <div>
          <div className="court-wrap">
            <span className="court-cut mono">low-rate data — stepped view</span>
            <div className="court-clock">
              <span className="court-clock__val">
                {frame.shot_clock != null ? frame.shot_clock.toFixed(1) : "—"}
              </span>
              <span className="court-clock__label">shot clock</span>
            </div>
            <Court
              entities={entities}
              ball={{ x: frame.ball_handler.x, y: frame.ball_handler.y }}
              arrow={arrow}
              draggable={false}
              activeId={activeId}
              onActiveChange={setActiveId}
              showNames={names}
            />
          </div>
          <div className="legend">
            <span>
              <i style={{ background: "#047857" }} /> Offense
            </span>
            <span>
              <i style={{ background: "#a39b8d" }} /> Defender
            </span>
          </div>
          <div className="transport">
            <button
              className="iconbtn"
              aria-label="Previous frame"
              disabled={nFrames <= 1}
              onClick={() => setIdx((i) => Math.max(0, i - 1))}
            >
              ⏮
            </button>
            <button
              className="iconbtn"
              aria-label="Next frame"
              disabled={nFrames <= 1}
              onClick={() => setIdx((i) => Math.min(nFrames - 1, i + 1))}
            >
              ⏭
            </button>
            <span className="scrub__pos">
              {idx + 1}/{nFrames}
            </span>
            <label className="switch switch--names" style={{ marginLeft: "auto" }}>
              <input type="checkbox" checked={names} onChange={() => setNames((v) => !v)} />
              <span className="switch__track">
                <span className="switch__knob" />
              </span>
              Names
            </label>
          </div>
        </div>

        <div className="panel">
          <div className="panel__block">
            <span className="panel__label">
              {isDecisionFrame ? "Decision point" : `Frame ${idx + 1}`}
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
              this possession has no high-rate tracking — stepped recorded
              decision frames only
            </p>
          </div>
          <div className="panel__block">
            <span className="panel__label">Action values (Q)</span>
            <QBars
              q={result.q}
              best={result.best}
              labels={labels}
              playerAction={isDecisionFrame ? p.player_action : undefined}
            />
          </div>
          {isDecisionFrame && (
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
  if (cat === "stream") return "Test-set possession";
  return "Agreement";
}
