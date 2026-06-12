// Raw 2D-canvas half-court renderer + draggable entities.
//
// Court frame matches the env: x in [0, 47] (basket at x = 47, RIGHT side),
// y in [0, 50]. We render the full half-court with the basket on the right and
// the half-court line on the left, so the offense attacks rightward — the
// natural reading direction for the arrows.

import { useCallback, useEffect, useRef } from "react";

export interface CourtEntity {
  id: string;
  kind: "ball-handler" | "teammate" | "defender";
  x: number; // env feet, 0..47
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
  /** "pass" = emerald solid; "shoot" = emerald toward basket dashed. */
  kind: "pass" | "shoot";
}

interface CourtProps {
  entities: CourtEntity[];
  ball: { x: number; y: number };
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
  /** court opacity 0..1 — dipped during a playback CUT for the dissolve. */
  fade?: number;
}

// Court geometry in env feet.
const COURT_X = 47; // length from half-court line (x=0) to baseline (x=47)
const COURT_Y = 50; // width
const PAD = 28; // px padding around the court inside the canvas
const ASPECT = COURT_Y / COURT_X; // height / width of the play area

// Logical canvas size (CSS pixels); device pixel ratio applied on top.
const CANVAS_W = 560;
const CANVAS_H = Math.round((CANVAS_W - PAD * 2) * ASPECT) + PAD * 2;

const COL = {
  teammate: "#047857",
  bhRing: "#022c22",
  ball: "#d97706",
  arrow: "#047857",
};

// env (x,y) -> canvas px. Basket on the RIGHT (x=47 -> right edge).
function toPx(x: number, y: number, w: number, h: number) {
  const playW = w - PAD * 2;
  const playH = h - PAD * 2;
  const px = PAD + (x / COURT_X) * playW;
  const py = PAD + (y / COURT_Y) * playH;
  return { px, py };
}

// canvas px -> env (x,y).
function toEnv(px: number, py: number, w: number, h: number) {
  const playW = w - PAD * 2;
  const playH = h - PAD * 2;
  let x = ((px - PAD) / playW) * COURT_X;
  let y = ((py - PAD) / playH) * COURT_Y;
  x = Math.max(0, Math.min(COURT_X, x));
  y = Math.max(0, Math.min(COURT_Y, y));
  return { x, y };
}

function drawCourt(ctx: CanvasRenderingContext2D, w: number, h: number) {
  ctx.clearRect(0, 0, w, h);

  // play area
  const tl = toPx(0, 0, w, h);
  const br = toPx(COURT_X, COURT_Y, w, h);
  const px = tl.px;
  const py = tl.py;
  const pw = br.px - tl.px;
  const ph = br.py - tl.py;

  // wood fill
  ctx.fillStyle = "#f4efe6";
  ctx.fillRect(px, py, pw, ph);

  ctx.lineWidth = 1.5;
  ctx.strokeStyle = "#c8b896";

  // outer boundary
  ctx.strokeRect(px, py, pw, ph);

  const basket = toPx(47, 25, w, h);

  // backboard + rim (right side)
  ctx.beginPath();
  const bbX = toPx(43, 25, w, h).px;
  ctx.moveTo(bbX + (basket.px - bbX) * 0.92, toPx(47, 22, w, h).py);
  ctx.lineTo(bbX + (basket.px - bbX) * 0.92, toPx(47, 28, w, h).py);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(basket.px - 6, basket.py, 5, 0, Math.PI * 2);
  ctx.stroke();

  // paint / key (from baseline x=47 back ~19ft, 16ft wide centered on y=25)
  const keyBaseX = toPx(47, 25, w, h).px;
  const keyTopX = toPx(28, 25, w, h).px;
  const keyY1 = toPx(47, 17, w, h).py;
  const keyY2 = toPx(47, 33, w, h).py;
  ctx.strokeRect(
    Math.min(keyTopX, keyBaseX),
    Math.min(keyY1, keyY2),
    Math.abs(keyBaseX - keyTopX),
    Math.abs(keyY2 - keyY1),
  );

  // free-throw circle
  ctx.beginPath();
  ctx.arc(keyTopX, basket.py, Math.abs(toPx(0, 0, w, h).py - toPx(0, 6, w, h).py), -Math.PI / 2, Math.PI / 2);
  ctx.stroke();

  // restricted-area arc near rim
  ctx.beginPath();
  ctx.arc(basket.px - 6, basket.py, Math.abs(toPx(0, 0, w, h).px - toPx(4, 0, w, h).px), Math.PI / 2, (3 * Math.PI) / 2);
  ctx.stroke();

  // three-point arc: 22 ft radius from basket, plus corner straights
  ctx.beginPath();
  const r3 = 22;
  const rPxX = Math.abs(toPx(0, 0, w, h).px - toPx(r3, 0, w, h).px);
  const rPxY = Math.abs(toPx(0, 0, w, h).py - toPx(0, r3, w, h).py);
  ctx.save();
  ctx.translate(basket.px, basket.py);
  ctx.scale(1, rPxY / rPxX);
  ctx.beginPath();
  ctx.arc(0, 0, rPxX, Math.PI / 2, (3 * Math.PI) / 2);
  ctx.restore();
  ctx.stroke();

  // half-court line (left edge x=0) accent
  ctx.strokeStyle = "#d3c4a3";
  ctx.beginPath();
  ctx.moveTo(px, py);
  ctx.lineTo(px, py + ph);
  ctx.stroke();
}

function drawArrow(ctx: CanvasRenderingContext2D, a: CourtArrow, w: number, h: number) {
  const from = toPx(a.fromX, a.fromY, w, h);
  const to = toPx(a.toX, a.toY, w, h);
  const dx = to.px - from.px;
  const dy = to.py - from.py;
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  // shorten so the head sits before the target dot
  const gap = 16;
  const ex = to.px - ux * gap;
  const ey = to.py - uy * gap;
  const sx = from.px + ux * 16;
  const sy = from.py + uy * 16;

  ctx.strokeStyle = COL.arrow;
  ctx.fillStyle = COL.arrow;
  ctx.lineWidth = 2.5;
  ctx.lineCap = "round";
  if (a.kind === "shoot") ctx.setLineDash([7, 6]);
  else ctx.setLineDash([]);

  ctx.beginPath();
  ctx.moveTo(sx, sy);
  ctx.lineTo(ex, ey);
  ctx.stroke();
  ctx.setLineDash([]);

  // arrowhead
  const head = 9;
  const ang = Math.atan2(uy, ux);
  ctx.beginPath();
  ctx.moveTo(ex, ey);
  ctx.lineTo(ex - head * Math.cos(ang - 0.42), ey - head * Math.sin(ang - 0.42));
  ctx.lineTo(ex - head * Math.cos(ang + 0.42), ey - head * Math.sin(ang + 0.42));
  ctx.closePath();
  ctx.fill();
}

/** Compact label: last name for offense (the user's "player numbers"). */
function compactName(label: string): string {
  const parts = label.trim().split(/\s+/);
  return parts.length > 1 ? parts[parts.length - 1] : label;
}

/**
 * Always-on label beneath a dot. Drawn with a paper-colored halo (stroke under
 * fill) so it stays legible over court lines and dots. `tone` controls contrast:
 * offense reads in warm ink, defenders sit fainter.
 */
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
  // paper halo for contrast on the wood + lines
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
  activeId: string | null,
  showNames: boolean,
  w: number,
  h: number,
) {
  // defenders first (under offense)
  const order = [...entities].sort((a, b) => {
    const rank = (k: CourtEntity["kind"]) =>
      k === "defender" ? 0 : k === "teammate" ? 1 : 2;
    return rank(a.kind) - rank(b.kind);
  });

  for (const e of order) {
    const { px, py } = toPx(e.x, e.y, w, h);
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
      // emphasis ring
      ctx.beginPath();
      ctx.arc(px, py, r + 4.5, 0, Math.PI * 2);
      ctx.lineWidth = 2;
      ctx.strokeStyle = COL.bhRing;
      ctx.stroke();
    }

    // Always-on compact name labels. Offense (BH + teammates) is labeled by
    // last name in ink; defenders are unnamed in the data, so they only get a
    // label on hover (the pill below). The active entity shows the full-name
    // pill instead of the compact label to avoid double-drawing.
    if (showNames && !active && e.kind !== "defender") {
      drawNameLabel(ctx, compactName(e.label), px, py, r, "offense");
    }

    if (active) {
      ctx.beginPath();
      ctx.arc(px, py, r + 8, 0, Math.PI * 2);
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "rgba(4,120,87,0.5)";
      ctx.stroke();

      // name label
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

  // ball — co-located with BH; render as a small accented dot offset slightly
  const bp = toPx(ball.x, ball.y, w, h);
  ctx.beginPath();
  ctx.arc(bp.px + 9, bp.py - 9, 4.5, 0, Math.PI * 2);
  ctx.fillStyle = COL.ball;
  ctx.fill();
  ctx.lineWidth = 1;
  ctx.strokeStyle = "#a85d05";
  ctx.stroke();
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
  arrow,
  draggable,
  activeId,
  onActiveChange,
  onDrag,
  showNames = true,
  fade = 1,
}: CourtProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const draggingRef = useRef<string | null>(null);

  // redraw whenever inputs change
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== CANVAS_W * dpr) {
      canvas.width = CANVAS_W * dpr;
      canvas.height = CANVAS_H * dpr;
    }
    ctx.save();
    ctx.scale(dpr, dpr);
    drawCourt(ctx, CANVAS_W, CANVAS_H);
    // CUT dissolve: the court (arrow + entities) fades over the held frame; the
    // wood backdrop stays so the dip reads as a dissolve, not a blackout.
    ctx.globalAlpha = Math.max(0, Math.min(1, fade));
    if (arrow) drawArrow(ctx, arrow, CANVAS_W, CANVAS_H);
    drawEntities(ctx, entities, ball, activeId, showNames, CANVAS_W, CANVAS_H);
    ctx.restore();
  }, [entities, ball, arrow, activeId, showNames, fade]);

  const eventToCanvas = useCallback((clientX: number, clientY: number) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const sx = CANVAS_W / rect.width;
    const sy = CANVAS_H / rect.height;
    return { cx: (clientX - rect.left) * sx, cy: (clientY - rect.top) * sy };
  }, []);

  const hitTest = useCallback(
    (cx: number, cy: number): CourtEntity | null => {
      // topmost (offense over defense) wins; iterate reverse draw order
      let best: CourtEntity | null = null;
      let bestD = Infinity;
      for (const e of entities) {
        const { px, py } = toPx(e.x, e.y, CANVAS_W, CANVAS_H);
        const d = Math.hypot(px - cx, py - cy);
        const r = (e.kind === "ball-handler" ? 11 : 9.5) + 6;
        if (d <= r && d < bestD) {
          bestD = d;
          best = e;
        }
      }
      return best;
    },
    [entities],
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
        const { x, y } = toEnv(cx, cy, CANVAS_W, CANVAS_H);
        onDrag(draggingRef.current, x, y);
        return;
      }
      // hover highlight (desktop only — pointer move without button)
      if (e.pointerType === "mouse" && e.buttons === 0) {
        const { cx, cy } = eventToCanvas(e.clientX, e.clientY);
        const hit = hitTest(cx, cy);
        onActiveChange(hit ? hit.id : null);
      }
    },
    [draggable, eventToCanvas, hitTest, onActiveChange, onDrag],
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
        aspectRatio: `${CANVAS_W} / ${CANVAS_H}`,
        cursor: draggable ? "grab" : "default",
      }}
      role="img"
      aria-label="Half-court possession diagram. Offense in emerald, defenders in gray, ball-handler ringed."
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
