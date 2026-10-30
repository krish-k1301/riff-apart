import { useEffect, useRef } from "react";

const BARS = 84;
const CYCLE = 7.5;     // seconds: merged → split → hold → merge
const SPLIT_AT = 1.0;
const MERGE_AT = 6.2;
const MOVE = 0.9;      // seconds a lane takes to slide out / back
const STAGGER = 0.08;

// Small seeded RNG so every visit draws the same "song"
const seeded = (seed) => () => (seed = (seed * 16807) % 2147483647) / 2147483647;

// Each stem gets a waveform with its own character
function makeWave(id) {
  const r = seeded([...id].reduce((a, c) => a * 31 + c.charCodeAt(0), 7) % 2147483646 + 1);
  return Array.from({ length: BARS }, (_, i) => {
    let v;
    switch (id) {
      case "vocals": v = (Math.sin(i / 7) > -0.35 ? 1 : 0.06) * (0.45 + 0.35 * Math.abs(Math.sin(i * 0.6)) + 0.15 * r()); break;
      case "drums":  v = i % 8 === 0 ? 0.95 : i % 4 === 0 ? 0.7 : i % 2 === 0 ? 0.32 + 0.1 * r() : 0.1 + 0.08 * r(); break;
      case "bass":   v = 0.42 + 0.22 * Math.sin(i / 5) + 0.12 * r(); break;
      case "guitar": v = (Math.floor(i / 10) % 2 ? 0.72 : 0.48) + 0.22 * r(); break;
      case "piano":  v = 0.85 * Math.exp(-(i % 12) / 3.5) + 0.06 * r(); break;
      default:       v = 0.2 + 0.18 * r();
    }
    return Math.min(1, Math.max(0.05, v));
  });
}

const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const clamp01 = (t) => Math.min(1, Math.max(0, t));

// How far lane i has slid out of the mix row at cycle time t (0 = inside the mix, 1 = in its slot)
function laneOpen(t, i) {
  const out = easeInOut(clamp01((t - SPLIT_AT - i * STAGGER) / MOVE));
  const back = easeInOut(clamp01((t - MERGE_AT - (5 - i) * STAGGER) / MOVE));
  return out * (1 - back);
}

const prefersReducedMotion = () =>
  window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

// Animated preview of what the app does: a mixed waveform pulling apart into six stem lanes.
export default function StemSplit({ stems }) {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const hoverRef = useRef(-1);

  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return undefined;

    const waves = stems.map((s) => makeWave(s.id));
    const mix = Array.from({ length: BARS }, (_, i) =>
      Math.min(1, (waves.reduce((sum, w) => sum + w[i], 0) / waves.length) * 1.7));
    const css = getComputedStyle(document.documentElement);
    const accent = css.getPropertyValue("--accent").trim() || "#FF9F1C";
    const muted = css.getPropertyValue("--text-3").trim() || "#85786A";
    const font = css.getPropertyValue("--font-mono").trim() || "monospace";

    let w = 0, h = 0, dpr = 1;
    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = canvas.clientWidth;
      h = canvas.clientHeight;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    };

    const drawRow = (data, y, rowH, color, alpha, scroll, x0, waveW) => {
      const barW = waveW / BARS;
      const frac = scroll % 1;
      const playhead = x0 + waveW * 0.4;
      ctx.fillStyle = color;
      for (let i = 0; i <= BARS; i++) {
        const v = data[(i + Math.floor(scroll)) % BARS];
        const x = x0 + (i - frac) * barW;
        if (x < x0 - 0.5 || x + barW > x0 + waveW + 0.5) continue;
        const bh = Math.max(2, v * rowH * 0.78);
        ctx.globalAlpha = alpha * (x < playhead ? 1 : 0.32);
        ctx.beginPath();
        ctx.roundRect(x + 1, y - bh / 2, Math.max(1, barW - 2.5), bh, 1.5);
        ctx.fill();
      }
    };

    const draw = (time, still) => {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const t = still ? 3 : time % CYCLE;
      const scroll = still ? 0 : time * 5;
      const pad = 18;
      const labelW = Math.min(78, w * 0.2);
      const x0 = pad + labelW;
      const waveW = w - x0 - pad;
      const rowH = (h - pad * 2) / 7;
      const mixY = pad + rowH / 2;
      const hovered = hoverRef.current;

      // the full mix on top
      ctx.font = `600 10px ${font}`;
      ctx.textBaseline = "middle";
      ctx.globalAlpha = 1;
      ctx.fillStyle = accent;
      ctx.fillText("MIX", pad, mixY);
      drawRow(mix, mixY, rowH, accent, 1, scroll, x0, waveW);

      // stems slide out of the mix into their own lanes
      stems.forEach((stem, i) => {
        const open = still ? 1 : laneOpen(t, i);
        if (open <= 0.001) return;
        const slotY = pad + rowH * (i + 1.5);
        const y = mixY + (slotY - mixY) * open;
        const focus = hovered < 0 || hovered === i ? 1 : 0.28;
        ctx.globalAlpha = open * focus;
        ctx.fillStyle = stem.color;
        ctx.fillText(stem.name.toUpperCase(), pad, y);
        drawRow(waves[i], y, rowH, stem.color, open * focus, scroll, x0, waveW);
      });

      // playhead
      ctx.globalAlpha = 0.9;
      ctx.fillStyle = muted;
      ctx.fillRect(x0 + waveW * 0.4, pad * 0.5, 1, h - pad);
      ctx.globalAlpha = 1;
    };

    resize();
    const still = prefersReducedMotion();
    let raf = 0;
    let visible = true;
    const start = performance.now();
    const loop = (now) => {
      draw((now - start) / 1000, false);
      raf = visible ? requestAnimationFrame(loop) : 0;
    };
    const kick = () => { if (!raf && visible && !still) raf = requestAnimationFrame(loop); };

    const ro = new ResizeObserver(() => { resize(); if (still) draw(0, true); });
    ro.observe(canvas);
    const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting && !document.hidden; kick(); });
    io.observe(wrap);
    const onVis = () => { visible = !document.hidden; kick(); };
    document.addEventListener("visibilitychange", onVis);

    const onMove = (e) => {
      const r = canvas.getBoundingClientRect();
      const row = Math.floor(((e.clientY - r.top) - 18) / ((r.height - 36) / 7)) - 1;
      hoverRef.current = row >= 0 && row < stems.length ? row : -1;
      if (still) draw(0, true);
    };
    const onLeave = () => { hoverRef.current = -1; if (still) draw(0, true); };
    wrap.addEventListener("pointermove", onMove);
    wrap.addEventListener("pointerleave", onLeave);

    if (still) draw(0, true); else kick();
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
      document.removeEventListener("visibilitychange", onVis);
      wrap.removeEventListener("pointermove", onMove);
      wrap.removeEventListener("pointerleave", onLeave);
    };
  }, [stems]);

  return (
    <figure ref={wrapRef} className="split">
      <figcaption className="split__bar">
        <span className="split__rec" aria-hidden="true" />
        <span>split preview</span>
        <span className="split__meta">htdemucs_6s · 44.1&nbsp;kHz</span>
      </figcaption>
      <canvas ref={canvasRef} className="split__canvas" aria-hidden="true" />
      <p className="sr-only">
        Animation: a song’s waveform splits into six stem lanes: vocals, drums, bass, guitar, piano and other.
      </p>
    </figure>
  );
}
