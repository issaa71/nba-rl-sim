// Raw 2D-canvas court renderer + draggable entities.
//
// TWO MODES, selected by `fullCourt`:
//   - HALF-COURT (default): x in [0, 47], single basket on the RIGHT at x = 47.
//     Used by the stepped fallback view (recorded feature-frame coords).
//   - FULL-COURT: x in [0, 94], BOTH baskets ((5.25,25) and (88.75,25)) + the
//     center line at x = 47. Used by the real-time tracking explorer, which
//     feeds RAW SportVU coordinates so the offense attacks its real basket.
// y is [0, 50] in both. All geometry is parameterized by the active court length
// (`courtX`) so a single set of draw helpers serves both modes.

import { useCallback, useEffect, useMemo, useRef } from "react";

export interface CourtEntity {
  id: string;
  kind: "ball-handler" | "teammate" | "defender";
  x: number; // env feet, 0..courtX
  y: number; // env feet, 0..50
  label: string;
  /** teammate slot 1..4 (for arrow targeting); undefined otherwise. */
  slot?: number;
}

export interface CourtArrow {
  fromX: number;
  fromY: number;
  toX: number;
  toY: number;
  /** "pass" = soft curved cue to a teammate; "shoot" = subtle cue at the rim. */
  kind: "pass" | "shoot";
}

interface CourtProps {
  entities: CourtEntity[];
  ball: { x: number; y: number };
  /**
   * Ball height (ft). When it rises above a threshold the ball renders as a
   * subtle in-flight arc (shot/pass), instead of the resting offset dot.
   */
  ballZ?: number;
  arrow: CourtArrow | null;
  /** drag enabled (what-if mode). */
  draggable: boolean;
  /** highlighted entity id (hover/tap). */
  activeId: string | null;
  onActiveChange: (id: string | null) => void;
  /** called continuously while dragging an entity to new env coords. */
  onDrag?: (id: string, x: number, y: number) => void;
  /** always-on compact name labels under offense dots (defaults on). */
  showNames?: boolean;
  /** court opacity 0..1 (kept for transitional fades). */
  fade?: number;
  /** render the FULL 94ft court (both baskets) instead of a half-court. */
  fullCourt?: boolean;
}

// Court geometry in env feet.
const COURT_X_HALF = 47; // half-court line (x=0) to baseline (x=47)
const COURT_X_FULL = 94; // baseline-to-baseline
const COURT_Y = 50; // width
const PAD = 28; // px padding around the court inside the canvas

// Logical canvas width (CSS px); full court is wider so players stay legible.
const CANVAS_W_HALF = 560;
const CANVAS_W_FULL = 760;

const COL = {
  teammate: "#047857",
  bhRing: "#022c22",
  ball: "#d97706",
  arrow: "#047857",
};

// env (x,y) -> canvas px, for a court `courtX` feet long.
function toPx(x: number, y: number, w: number, h: number, courtX: number) {
  const playW = w - PAD * 2;
  const playH = h - PAD * 2;
  const px = PAD + (x / courtX) * playW;
  const py = PAD + (y / COURT_Y) * playH;
  return { px, py };
}

// canvas px -> env (x,y).
function toEnv(px: number, py: number, w: number, h: number, courtX: number) {
  const playW = w - PAD * 2;
  const playH = h - PAD * 2;
  let x = ((px - PAD) / playW) * courtX;
  let y = ((py - PAD) / playH) * COURT_Y;
  x = Math.max(0, Math.min(courtX, x));
  y = Math.max(0, Math.min(COURT_Y, y));
  return { x, y };
}

/** ft -> px scale on the x axis (used for radii). */
function ftPxX(ft: number, w: number, courtX: number) {
  return Math.abs(toPx(0, 0, w, 0, courtX).px - toPx(ft, 0, w, 0, courtX).px);
}
function ftPxY(ft: number, h: number) {
  return Math.abs(toPx(0, 0, 0, h, 1).py - toPx(0, ft, 0, h, 1).py);
}

/**
 * Draw one scoring end: backboard, rim, painted key, free-throw circle,
 * restricted arc, and 3pt arc. `baselineX` is the end line (0 or courtX);
 * `basketX` is the rim x. The key + arcs open toward center court.
 */
function drawBasketEnd(
  ctx: CanvasRenderingContext2D,
  baselineX: number,
  basketX: number,
  w: number,
  h: number,
  courtX: number,
) {
  const inward = baselineX < basketX ? 1 : -1; // toward center court
  const basket = toPx(basketX, 25, w, h, courtX);

  // backboard (4 ft in from the baseline) + rim
  const bbX = toPx(baselineX + inward * 4, 25, w, h, courtX).px;
  ctx.beginPath();
  ctx.moveTo(bbX, toPx(0, 22, w, h, courtX).py);
  ctx.lineTo(bbX, toPx(0, 28, w, h, courtX).py);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(basket.px + inward * 1, basket.py, 5, 0, Math.PI * 2);
  ctx.stroke();

  // painted key: baseline back ~19 ft, 16 ft wide (y 17..33)
  const keyBaseX = toPx(baselineX, 25, w, h, courtX).px;
  const keyTopX = toPx(baselineX + inward * 19, 25, w, h, courtX).px;
  const keyY1 = toPx(0, 17, w, h, courtX).py;
  const keyY2 = toPx(0, 33, w, h, courtX).py;
  ctx.strokeRect(
    Math.min(keyTopX, keyBaseX),
    Math.min(keyY1, keyY2),
    Math.abs(keyBaseX - keyTopX),
    Math.abs(keyY2 - keyY1),
  );

  // free-throw circle (6 ft radius at the FT line), facing center
  ctx.beginPath();
  ctx.arc(
    keyTopX,
    basket.py,
    ftPxY(6, h),
    inward > 0 ? -Math.PI / 2 : Math.PI / 2,
    inward > 0 ? Math.PI / 2 : (3 * Math.PI) / 2,
  );
  ctx.stroke();

  // restricted-area arc near the rim (4 ft), facing center
  ctx.beginPath();
  ctx.arc(
    basket.px + inward * 1,
    basket.py,
    ftPxX(4, w, courtX),
    inward > 0 ? -Math.PI / 2 : Math.PI / 2,
    inward > 0 ? Math.PI / 2 : (3 * Math.PI) / 2,
  );
  ctx.stroke();

  // three-point arc: ~22 ft radius from the basket, opening toward center
  const r3 = 22;
  const rPxX = ftPxX(r3, w, courtX);
  const rPxY = ftPxY(r3, h);
  ctx.save();
  ctx.translate(basket.px, basket.py);
  ctx.scale(1, rPxY / rPxX);
  ctx.beginPath();
  ctx.arc(
    0,
    0,
    rPxX,
    inward > 0 ? -Math.PI / 2 : Math.PI / 2,
    inward > 0 ? Math.PI / 2 : (3 * Math.PI) / 2,
  );
  ctx.restore();
  ctx.stroke();
}

function drawCourt(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  courtX: number,
  fullCourt: boolean,
) {
  ctx.clearRect(0, 0, w, h);

  const tl = toPx(0, 0, w, h, courtX);
  const br = toPx(courtX, COURT_Y, w, h, courtX);
  const px = tl.px;
  const py = tl.py;
  const pw = br.px - tl.px;
  const ph = br.py - tl.py;

  // wood fill
  ctx.fillStyle = "#f4efe6";
  ctx.fillRect(px, py, pw, ph);

  ctx.lineWidth = 1.5;
  ctx.strokeStyle = "#c8b896";
  ctx.strokeRect(px, py, pw, ph); // outer boundary

  if (fullCourt) {
    // both scoring ends
    drawBasketEnd(ctx, 0, 5.25, w, h, courtX);
    drawBasketEnd(ctx, courtX, courtX - 5.25, w, h, courtX);
    // center line + center circle
    const cl = toPx(courtX / 2, 0, w, h, courtX).px;
    ctx.strokeStyle = "#d3c4a3";
    ctx.beginPath();
    ctx.moveTo(cl, py);
    ctx.lineTo(cl, py + ph);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(toPx(courtX / 2, 25, w, h, courtX).px, toPx(0, 25, w, h, courtX).py, ftPxY(6, h), 0, Math.PI * 2);
    ctx.stroke();
  } else {
    // single right-hand basket + the half-court accent line on the left edge
    drawBasketEnd(ctx, courtX, courtX - 5.25, w, h, courtX);
    // (the half-court line is the left boundary, already stroked)
    ctx.strokeStyle = "#d3c4a3";
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(px, py + ph);
    ctx.stroke();
  }
}

/**
 * Restrained recommendation cue. Two restrained forms:
 *   PASS  — a soft, low-opacity quadratic curve bowing from the ball-handler
 *           toward the target teammate, stopping short of the dot, plus a small
 *           accent ring + connecting tick on the target. No big arrowhead.
 *   SHOOT — no long line; a short tick off the ball-handler toward the basket
 *           and a soft accent ring on the rim, so the call reads subtly.
 */
function drawArrow(
  ctx: CanvasRenderingContext2D,
  a: CourtArrow,
  w: number,
  h: number,
  courtX: number,
) {
  const from = toPx(a.fromX, a.fromY, w, h, courtX);
  const to = toPx(a.toX, a.toY, w, h, courtX);
  const dx = to.px - from.px;
  const dy = to.py - from.py;
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  // point the rim ring/ticks at the basket regardless of which side it's on
  const rimDir = a.toX >= a.fromX ? 1 : -1;

  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  if (a.kind === "shoot") {
    const tick = Math.min(22, len * 0.5);
    const sx = from.px + ux * 15;
    const sy = from.py + uy * 15;
    ctx.strokeStyle = "rgba(4,120,87,0.45)";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(sx + ux * tick, sy + uy * tick);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.strokeStyle = "rgba(4,120,87,0.55)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(to.px - rimDir * 6, to.py, 11, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
    return;
  }

  // PASS cue
  const startGap = 15;
  const endGap = 18;
  const sx = from.px + ux * startGap;
  const sy = from.py + uy * startGap;
  const ex = to.px - ux * endGap;
  const ey = to.py - uy * endGap;
  const bow = Math.min(26, len * 0.16);
  const mx = (sx + ex) / 2 - uy * bow;
  const my = (sy + ey) / 2 + ux * bow;

  ctx.strokeStyle = "rgba(4,120,87,0.32)";
  ctx.lineWidth = 2.25;
  ctx.beginPath();
  ctx.moveTo(sx, sy);
  ctx.quadraticCurveTo(mx, my, ex, ey);
  ctx.stroke();

  ctx.strokeStyle = "rgba(4,120,87,0.6)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(to.px, to.py, 13.5, 0, Math.PI * 2);
  ctx.stroke();

  ctx.restore();
}

/** Compact label: last name for offense (the user's "player numbers"). */
function compactName(label: string): string {
  const parts = label.trim().split(/\s+/);
  return parts.length > 1 ? parts[parts.length - 1] : label;
}

function drawNameLabel(
  ctx: CanvasRenderingContext2D,
  text: string,
  px: number,
  py: number,
  r: number,
  tone: "offense" | "defender",
) {
  ctx.font =
    (tone === "offense" ? "600 " : "500 ") +
    "10px 'Geist', system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const ty = py + r + 3;
  ctx.lineJoin = "round";
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(244,239,230,0.92)";
  ctx.strokeText(text, px, ty);
  ctx.fillStyle =
    tone === "offense" ? "rgba(28,26,23,0.95)" : "rgba(95,90,82,0.78)";
  ctx.fillText(text, px, ty);
}

function drawEntities(
  ctx: CanvasRenderingContext2D,
  entities: CourtEntity[],
  ball: { x: number; y: number },
  ballZ: number,
  activeId: string | null,
  showNames: boolean,
  w: number,
  h: number,
  courtX: number,
) {
  const order = [...entities].sort((a, b) => {
    const rank = (k: CourtEntity["kind"]) =>
      k === "defender" ? 0 : k === "teammate" ? 1 : 2;
    return rank(a.kind) - rank(b.kind);
  });

  for (const e of order) {
    const { px, py } = toPx(e.x, e.y, w, h, courtX);
    const active = e.id === activeId;
    const r = e.kind === "ball-handler" ? 11 : 9.5;

    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    if (e.kind === "defender") {
      ctx.fillStyle = "#a39b8d";
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "#7a7264";
      ctx.stroke();
    } else {
      ctx.fillStyle = COL.teammate;
      ctx.fill();
    }

    if (e.kind === "ball-handler") {
      ctx.beginPath();
      ctx.arc(px, py, r + 4.5, 0, Math.PI * 2);
      ctx.lineWidth = 2;
      ctx.strokeStyle = COL.bhRing;
      ctx.stroke();
    }

    if (showNames && !active && e.kind !== "defender") {
      drawNameLabel(ctx, compactName(e.label), px, py, r, "offense");
    }

    if (active) {
      ctx.beginPath();
      ctx.arc(px, py, r + 8, 0, Math.PI * 2);
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "rgba(4,120,87,0.5)";
      ctx.stroke();

      const text = e.label;
      ctx.font = "600 12px 'Geist', system-ui, sans-serif";
      const tw = ctx.measureText(text).width;
      const bx = px - tw / 2 - 7;
      const by = py - r - 28;
      ctx.fillStyle = "rgba(28,26,23,0.92)";
      roundRect(ctx, bx, by, tw + 14, 20, 5);
      ctx.fill();
      ctx.fillStyle = "#faf9f7";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, px, by + 10);
    }
  }

  // ball — the REAL tracked ball, with a height-driven lift for shots/lobs.
  const bp = toPx(ball.x, ball.y, w, h, courtX);
  const REST_Z = 4;
  const lift = ballZ > REST_Z ? Math.min(1, (ballZ - REST_Z) / 7) : 0;
  if (lift > 0) {
    ctx.beginPath();
    ctx.ellipse(bp.px, bp.py, 5 - lift * 1.5, 2.6 - lift, 0, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(40,32,20,0.18)";
    ctx.fill();
    const ry = bp.py - lift * 26;
    ctx.beginPath();
    ctx.moveTo(bp.px, bp.py);
    ctx.quadraticCurveTo(bp.px - 4, (bp.py + ry) / 2, bp.px, ry);
    ctx.strokeStyle = "rgba(217,119,6,0.28)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(bp.px, ry, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = COL.ball;
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = "#a85d05";
    ctx.stroke();
  } else {
    ctx.beginPath();
    ctx.arc(bp.px + 9, bp.py - 9, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = COL.ball;
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = "#a85d05";
    ctx.stroke();
  }
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

export function Court({
  entities,
  ball,
  ballZ = 0,
  arrow,
  draggable,
  activeId,
  onActiveChange,
  onDrag,
  showNames = true,
  fade = 1,
  fullCourt = false,
}: CourtProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const draggingRef = useRef<string | null>(null);

  // Active court dimensions (depend on mode).
  const courtX = fullCourt ? COURT_X_FULL : COURT_X_HALF;
  const { canvasW, canvasH } = useMemo(() => {
    const cw = fullCourt ? CANVAS_W_FULL : CANVAS_W_HALF;
    const ch = Math.round((cw - PAD * 2) * (COURT_Y / courtX)) + PAD * 2;
    return { canvasW: cw, canvasH: ch };
  }, [fullCourt, courtX]);

  // redraw whenever inputs change
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== canvasW * dpr || canvas.height !== canvasH * dpr) {
      canvas.width = canvasW * dpr;
      canvas.height = canvasH * dpr;
    }
    ctx.save();
    ctx.scale(dpr, dpr);
    drawCourt(ctx, canvasW, canvasH, courtX, fullCourt);
    ctx.globalAlpha = Math.max(0, Math.min(1, fade));
    if (arrow) drawArrow(ctx, arrow, canvasW, canvasH, courtX);
    drawEntities(ctx, entities, ball, ballZ, activeId, showNames, canvasW, canvasH, courtX);
    ctx.restore();
  }, [entities, ball, ballZ, arrow, activeId, showNames, fade, canvasW, canvasH, courtX, fullCourt]);

  const eventToCanvas = useCallback(
    (clientX: number, clientY: number) => {
      const canvas = canvasRef.current!;
      const rect = canvas.getBoundingClientRect();
      const sx = canvasW / rect.width;
      const sy = canvasH / rect.height;
      return { cx: (clientX - rect.left) * sx, cy: (clientY - rect.top) * sy };
    },
    [canvasW, canvasH],
  );

  const hitTest = useCallback(
    (cx: number, cy: number): CourtEntity | null => {
      let best: CourtEntity | null = null;
      let bestD = Infinity;
      for (const e of entities) {
        const { px, py } = toPx(e.x, e.y, canvasW, canvasH, courtX);
        const d = Math.hypot(px - cx, py - cy);
        const r = (e.kind === "ball-handler" ? 11 : 9.5) + 6;
        if (d <= r && d < bestD) {
          bestD = d;
          best = e;
        }
      }
      return best;
    },
    [entities, canvasW, canvasH, courtX],
  );

  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      const { cx, cy } = eventToCanvas(e.clientX, e.clientY);
      const hit = hitTest(cx, cy);
      if (hit) {
        onActiveChange(hit.id);
        if (draggable && onDrag) {
          draggingRef.current = hit.id;
          e.currentTarget.setPointerCapture(e.pointerId);
        }
      } else {
        onActiveChange(null);
      }
    },
    [draggable, eventToCanvas, hitTest, onActiveChange, onDrag],
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (draggingRef.current && draggable && onDrag) {
        const { cx, cy } = eventToCanvas(e.clientX, e.clientY);
        const { x, y } = toEnv(cx, cy, canvasW, canvasH, courtX);
        onDrag(draggingRef.current, x, y);
        return;
      }
      if (e.pointerType === "mouse" && e.buttons === 0) {
        const { cx, cy } = eventToCanvas(e.clientX, e.clientY);
        const hit = hitTest(cx, cy);
        onActiveChange(hit ? hit.id : null);
      }
    },
    [draggable, eventToCanvas, hitTest, onActiveChange, onDrag, canvasW, canvasH, courtX],
  );

  const endDrag = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    if (draggingRef.current) {
      try {
        e.currentTarget.releasePointerCapture(e.pointerId);
      } catch {
        /* capture may already be gone */
      }
      draggingRef.current = null;
    }
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="court-canvas"
      style={{
        aspectRatio: `${canvasW} / ${canvasH}`,
        cursor: draggable ? "grab" : "default",
      }}
      role="img"
      aria-label={
        fullCourt
          ? "Full-court possession diagram. Offense in emerald, defenders in gray, ball-handler ringed; both baskets shown."
          : "Half-court possession diagram. Offense in emerald, defenders in gray, ball-handler ringed."
      }
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onPointerLeave={(e) => {
        if (!draggingRef.current && e.pointerType === "mouse") onActiveChange(null);
      }}
    />
  );
}
