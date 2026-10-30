import { useEffect, useRef, useState } from "react";
import { gsap } from "gsap";
import MaskedHeading from "./MaskedHeading/MaskedHeading.jsx";

const SEEN_KEY = "riffapart:intro-seen";
const VIDEO_SRC = "/intro/guitar.mp4";
const POSTER_SRC = "/intro/guitar-poster.webp";
const WORD = "riffapart";
const ZOOM_LETTER = 1; // the "i" — its stem is the doorway into the site

// Shown inside the letters until real footage lands in public/intro/
const FALLBACK_SRC = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(
  '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#F4D58D"/><stop offset=".55" stop-color="#FF9F1C"/><stop offset="1" stop-color="#E5484D"/></linearGradient></defs><rect width="1600" height="900" fill="url(#g)"/></svg>'
)}`;

export function introAlreadySeen() {
  try { return sessionStorage.getItem(SEEN_KEY) === "1"; } catch { return false; }
}

function markSeen() {
  try { sessionStorage.setItem(SEEN_KEY, "1"); } catch {}
}

const prefersReducedMotion = () =>
  window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

// Viewport point inside the stem of the "i", below its dot.
function findDoorway(stage) {
  const word = stage.querySelector(".masked-heading__word");
  const baseline = stage.querySelector(".masked-heading__baseline");
  const node = word?.firstChild;
  if (!node || node.nodeType !== 3 || !baseline) return null;
  const range = document.createRange();
  range.setStart(node, ZOOM_LETTER);
  range.setEnd(node, ZOOM_LETTER + 1);
  const glyph = range.getBoundingClientRect();
  const fontPx = parseFloat(getComputedStyle(word).fontSize) || 100;
  return {
    x: glyph.left + glyph.width / 2,
    y: baseline.getBoundingClientRect().top - fontPx * 0.25,
  };
}

export default function Intro({ onDone }) {
  const overlayRef = useRef(null);
  const stageRef = useRef(null);
  const [media, setMedia] = useState(null); // { type, src, poster }

  // Use the footage if it exists, otherwise a palette gradient.
  // (Vite's dev server answers missing files with index.html, so check the type.)
  useEffect(() => {
    let alive = true;
    fetch(VIDEO_SRC, { method: "HEAD" })
      .then((r) => r.ok && (r.headers.get("content-type") || "").startsWith("video/"))
      .catch(() => false)
      .then((isVideo) => {
        if (!alive) return;
        setMedia(isVideo
          ? { type: "video", src: VIDEO_SRC, poster: POSTER_SRC }
          : { type: "image", src: FALLBACK_SRC, poster: "" });
      });
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    if (!media) return;
    const overlay = overlayRef.current;
    const stage = stageRef.current;
    let finished = false;

    const finish = () => {
      if (finished) return;
      finished = true;
      markSeen();
      onDone();
    };

    const tl = gsap.timeline({ onComplete: finish });

    if (prefersReducedMotion()) {
      tl.to(overlay, { opacity: 0, duration: 0.3, delay: 0.8 });
    } else {
      // MaskedHeading runs its own ~1.1 s rise; hold briefly, then fly through the "i".
      // Measured lazily (function values run when the tween starts) so layout is final.
      let door = null;
      const getDoor = () => {
        if (door) return door;
        const box = stage.getBoundingClientRect();
        const p = findDoorway(stage) ?? { x: box.left + box.width / 2, y: box.top + box.height / 2 };
        door = { ...p, ox: p.x - box.left, oy: p.y - box.top };
        return door;
      };
      tl.set(stage, { transformOrigin: () => `${getDoor().ox}px ${getDoor().oy}px` }, 1.6);
      tl.to(stage, {
        x:() => window.innerWidth / 2 - getDoor().x,
        y: () => window.innerHeight / 2 - getDoor().y,
        scale: 140,
        duration: 1.25,
        ease: "power3.in",
      }, 1.6);
      tl.to(overlay, { opacity: 0, duration: 0.45, ease: "power1.out" }, 2.55);
    }

    // Any input cancels the animation (Vercel guidelines: let people skip)
    const skip = () => {
      if (finished) return;
      tl.kill();
      gsap.to(overlay, { opacity: 0, duration: 0.25, onComplete: finish });
    };
    const onKey = (e) => { if (!e.repeat) skip(); };
    window.addEventListener("pointerdown", skip, { capture: true });
    window.addEventListener("keydown", onKey, { capture: true });

    return () => {
      tl.kill();
      gsap.killTweensOf([overlay, stage]);
      window.removeEventListener("pointerdown", skip, { capture: true });
      window.removeEventListener("keydown", onKey, { capture: true });
    };
  }, [media, onDone]);

  return (
    <div ref={overlayRef} className="intro" role="presentation">
      <div ref={stageRef} className="intro__stage" aria-hidden="true">
        {media && (
          <MaskedHeading
            className="intro__heading"
            text={WORD}
            tag="div"
            mediaType={media.type}
            src={media.src}
            poster={media.poster}
            reveal="rise"
            trigger="view"
            duration={1.1}
            textScale={0.28}
            tracking={0}
            weight={700}
            tint={media.type === "video" ? "#FF9F1C55" : ""}
            parallax={18}
            drift={14}
            translate="no"
          />
        )}
      </div>
      <button type="button" className="intro__skip">Skip Intro</button>
    </div>
  );
}
