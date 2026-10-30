import { useEffect, useState } from "react";

const GLYPHS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#%&*+=?";

const scramble = (text) =>
  Array.from(text, (ch) => (ch === " " ? ch : GLYPHS[(Math.random() * GLYPHS.length) | 0])).join("");

const prefersReducedMotion = () =>
  window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

// Shows `text` scrambled, then decodes it left→right once `active` turns true.
export default function useScramble(text, active, duration = 900) {
  const [out, setOut] = useState(() => (prefersReducedMotion() ? text : scramble(text)));

  useEffect(() => {
    if (prefersReducedMotion()) {
      setOut(text);
      return;
    }
    if (!active) {
      setOut(scramble(text));
      return;
    }
    let raf;
    const start = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - start) / duration);
      const settled = Math.floor(t * text.length);
      let next = "";
      for (let i = 0; i < text.length; i++) {
        const ch = text[i];
        next += i < settled || ch === " " ? ch : GLYPHS[(Math.random() * GLYPHS.length) | 0];
      }
      setOut(next);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [text, active, duration]);

  return out;
}
