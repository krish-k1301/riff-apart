import { useEffect, useRef, useState } from "react";
import { gsap } from "gsap";
import DitherWaves from "./DitherWaves.jsx";

const prefersReducedMotion = () =>
  window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

function formatEta(seconds) {
  if (seconds == null) return "estimating…";
  const s = Math.max(0, Math.round(seconds));
  if (s < 10) return "a few seconds left";
  const m = Math.floor(s / 60);
  return m > 0 ? `${m}:${String(s % 60).padStart(2, "0")} left` : `${s} s left`;
}

// Full-screen loader: glitching % counter with a strike-through bar. When `complete`
// flips on, the counter cuts out and a window opens from a thin line to reveal the
// page underneath, then calls onRevealed.
export default function SeparationLoader({ progress, eta, status, fileName, quip, complete, onRevealed, onCancel }) {
  const rootRef = useRef(null);
  const counterRef = useRef(null);
  const [shown, setShown] = useState(0);          // eased % on screen
  const [glitch, setGlitch] = useState(false);
  const [now, setNow] = useState(() => performance.now());
  const [announced, setAnnounced] = useState(0);  // screen readers hear 10% steps, not every tick
  const onRevealedRef = useRef(onRevealed);       // parent re-renders must not restart the reveal
  onRevealedRef.current = onRevealed;

  // Ease the displayed number toward the real progress; never count backwards
  useEffect(() => {
    let raf;
    const tick = () => {
      setShown((v) => {
        const target = Math.max(v, progress);
        const next = v + (target - v) * 0.08;
        return target - next < 0.05 ? target : next;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [progress]);

  useEffect(() => {
    const step = Math.floor(progress / 10) * 10;
    if (step !== announced) setAnnounced(step);
  }, [progress, announced]);

  // Count the ETA down between polls
  useEffect(() => {
    const t = setInterval(() => setNow(performance.now()), 1000);
    return () => clearInterval(t);
  }, []);

  // Random short glitch bursts
  useEffect(() => {
    if (prefersReducedMotion()) return;
    let t;
    const schedule = () => {
      t = setTimeout(() => {
        setGlitch(true);
        t = setTimeout(() => { setGlitch(false); schedule(); }, 90 + Math.random() * 140);
      }, 700 + Math.random() * 1800);
    };
    schedule();
    return () => clearTimeout(t);
  }, []);

  // The reveal: line → narrow window → letterbox → full screen
  useEffect(() => {
    if (!complete) return;
    const root = rootRef.current;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const hole = { w: 0, h: 0 };
    const paint = () => {
      const l = (vw - hole.w) / 2, r = (vw + hole.w) / 2;
      const t = (vh - hole.h) / 2, b = (vh + hole.h) / 2;
      root.style.clipPath =
        `polygon(evenodd, 0 0, ${vw}px 0, ${vw}px ${vh}px, 0 ${vh}px, 0 0, ` +
        `${l}px ${t}px, ${r}px ${t}px, ${r}px ${b}px, ${l}px ${b}px, ${l}px ${t}px)`;
    };

    const tl = gsap.timeline({ onComplete: () => onRevealedRef.current?.() });
    if (prefersReducedMotion()) {
      tl.to(root, { opacity: 0, duration: 0.3, delay: 0.3 });
      return () => tl.kill();
    }
    // ~1.6 s end to end
    tl.to({}, { duration: 0.15 }) // let 100% land
      .to(counterRef.current, { opacity: 0, duration: 0.12, ease: "steps(3)" })
      .add(() => setGlitch(true), "<")
      .to(hole, { w: 2, h: vh * 0.28, duration: 0.2, ease: "power2.out", onUpdate: paint }, "+=0.05")
      .to(hole, { w: vw * 0.07, duration: 0.18, ease: "power3.out", onUpdate: paint })
      .to(hole, { w: Math.min(vw * 0.46, 760), h: Math.min(vh * 0.32, 380), duration: 0.35, ease: "expo.inOut", onUpdate: paint })
      .to(hole, { w: Math.min(vw * 0.5, 820), h: Math.min(vh * 0.35, 410), duration: 0.15, ease: "none", onUpdate: paint })
      .to(hole, { w: vw + 4, h: vh + 4, duration: 0.4, ease: "expo.inOut", onUpdate: paint });
    return () => tl.kill();
  }, [complete]);

  const pct = Math.min(100, Math.floor(shown));
  const remaining = eta?.seconds == null ? null : eta.seconds - (now - eta.at) / 1000;
  const label = `${String(pct).padStart(2, " ")}%`;

  return (
    <div ref={rootRef} className="sep-loader">
      {/* something to play with while the model works */}
      <div className="sep-bg" aria-hidden="true">
        <DitherWaves paused={complete} />
      </div>
      <div className="sep-scrim" aria-hidden="true" />
      <span className="sep-corner sep-corner--tl" aria-hidden="true" />
      <span className="sep-corner sep-corner--tr" aria-hidden="true" />
      <span className="sep-corner sep-corner--bl" aria-hidden="true" />
      <span className="sep-corner sep-corner--br" aria-hidden="true" />
      <span className="sep-tick" aria-hidden="true">+</span>
      <span className="sep-side sep-side--l" aria-hidden="true">{status}</span>
      <span className="sep-side sep-side--r" aria-hidden="true">06 stems</span>

      <div
        ref={counterRef}
        className={`sep-counter${glitch ? " is-glitching" : ""}`}
        role="progressbar"
        aria-label="Separating stems"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct}
      >
        <span className="sep-bar" style={{ width: `${Math.max(2, shown)}%` }} />
        <span className="sep-num" data-text={label}>{label}</span>
      </div>

      <div className="sep-meta">
        <span className="tabular">{complete ? "done" : formatEta(remaining)}</span>
        <span className="sep-file" translate="no">{fileName}</span>
      </div>

      <div className="sep-quip" aria-hidden="true">{quip}</div>
      <span className="sr-only" role="status" aria-live="polite">
        {complete ? "Stems ready" : `${status}, ${announced}%`}
      </span>

      {!complete && (
        <button type="button" className="sep-cancel" onClick={onCancel}>Cancel</button>
      )}
    </div>
  );
}
