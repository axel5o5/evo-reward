import { useEffect, useRef } from "react";
import { ReplayData, frameView } from "../lib/replayLoader";

interface Props {
  data: ReplayData;
  frameIdx: number;
  // Visual toggles
  showHeading?: boolean;
  energyAlpha?: boolean;
  // Layout: canvas is a square — the parent decides CSS size.
  className?: string;
}

// Species colors. 0 = prey, 1 = predator. Picked to read in both themes.
const COLOR_PREY = "#4ade80";      // green-400
const COLOR_PREDATOR = "#f87171";  // red-400
const COLOR_FOOD = "#60a5fa";      // blue-400
const COLOR_HEADING = "rgba(17, 24, 39, 0.55)"; // gray-900/55

export default function ReplayCanvas({
  data,
  frameIdx,
  showHeading = true,
  energyAlpha = true,
  className,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  // Remember last laid-out CSS size so we only resize the backing store when it actually changes.
  const sizeRef = useRef<{ w: number; h: number; dpr: number }>({ w: 0, h: 0, dpr: 1 });

  // Resize handling — ResizeObserver gives us accurate CSS pixels even under
  // flex/grid parents that don't have an explicit width.
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const applySize = () => {
      const rect = container.getBoundingClientRect();
      const side = Math.max(1, Math.floor(Math.min(rect.width, rect.height)));
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      if (side === sizeRef.current.w && dpr === sizeRef.current.dpr) return;
      sizeRef.current = { w: side, h: side, dpr };
      canvas.width = side * dpr;
      canvas.height = side * dpr;
      canvas.style.width = side + "px";
      canvas.style.height = side + "px";
      draw(); // redraw after resize
    };

    const ro = new ResizeObserver(applySize);
    ro.observe(container);
    applySize();
    return () => ro.disconnect();
    // Intentionally empty deps — applySize closure captures latest draw via ref pattern below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Re-draw on every relevant change.
  useEffect(() => {
    draw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, frameIdx, showHeading, energyAlpha]);

  function draw() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { w, dpr } = sizeRef.current;
    if (w === 0) return;
    const pxSide = w * dpr;
    const world = data.meta.world_size;
    const scale = pxSide / world;

    // Background (pale off-white in light mode; dark panel in dark mode).
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    const isDark = document.documentElement.classList.contains("dark");
    ctx.fillStyle = isDark ? "#0b1220" : "#f8fafc";
    ctx.fillRect(0, 0, pxSide, pxSide);

    // World → pixel: y-flipped so (0,0) renders as bottom-left (matches training intuition).
    ctx.translate(0, pxSide);
    ctx.scale(scale, -scale);

    const fv = frameView(data, frameIdx);

    // --- food ---
    const foodRadius = 3 / scale;
    ctx.fillStyle = COLOR_FOOD;
    for (let i = 0; i < data.meta.food_max; i++) {
      if (!fv.foodActive[i]) continue;
      const x = fv.foodPos[i * 2];
      const y = fv.foodPos[i * 2 + 1];
      ctx.beginPath();
      ctx.arc(x, y, foodRadius, 0, Math.PI * 2);
      ctx.fill();
    }

    // --- agents ---
    // Single loop, color-branched by species. Energy modulates alpha so that
    // near-death agents fade — a cheap visual cue for population health.
    for (let i = 0; i < data.meta.max_agents; i++) {
      if (!fv.alive[i]) continue;
      const x = fv.pos[i * 2];
      const y = fv.pos[i * 2 + 1];
      const r = data.radii[i];
      const species = data.species[i];

      if (energyAlpha) {
        const e = fv.energy[i];
        // 0 at e=0, 1 at e≥200 (capacity is 1000 but most live <200). Clamp.
        const a = Math.max(0.25, Math.min(1, e / 200));
        ctx.globalAlpha = a;
      } else {
        ctx.globalAlpha = 1;
      }

      ctx.fillStyle = species === 1 ? COLOR_PREDATOR : COLOR_PREY;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();

      if (showHeading) {
        const a = fv.angle[i];
        ctx.strokeStyle = COLOR_HEADING;
        ctx.lineWidth = 1.2 / scale;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + Math.cos(a) * r, y + Math.sin(a) * r);
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    ctx.restore();

    // Optional HUD corner: step number + active counts. Drawn in pixel coords.
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    let nPrey = 0, nPred = 0, nFood = 0;
    for (let i = 0; i < data.meta.max_agents; i++) {
      if (!fv.alive[i]) continue;
      if (data.species[i] === 1) nPred++; else nPrey++;
    }
    for (let i = 0; i < data.meta.food_max; i++) if (fv.foodActive[i]) nFood++;
    ctx.fillStyle = isDark ? "rgba(226,232,240,0.85)" : "rgba(15,23,42,0.75)";
    ctx.font = `${12 * dpr}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    const pad = 8 * dpr;
    ctx.fillText(`step ${fv.step}`, pad, pad + 12 * dpr);
    ctx.fillText(`prey ${nPrey}  pred ${nPred}  food ${nFood}`, pad, pad + 28 * dpr);
    ctx.restore();
  }

  return (
    <div ref={containerRef} className={className} style={{ position: "relative" }}>
      <canvas ref={canvasRef} style={{ display: "block" }} />
    </div>
  );
}
