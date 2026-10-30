import { useEffect, useRef } from "react";
import { Renderer, Program, Mesh, Triangle } from "ogl";

const CELL = 3; // CSS px per dither cell; the canvas renders at this low res and scales up crisp

const vertex = /* glsl */ `#version 300 es
in vec2 position;
void main() { gl_Position = vec4(position, 0.0, 1.0); }
`;

// Ordered-dither print of stacked frequency waves (the "ridges") over an
// interference field from drifting point sources. The pointer is one more source.
const fragment = /* glsl */ `#version 300 es
precision highp float;
uniform vec2 uRes;
uniform float uTime;
uniform float uIntro;
uniform vec2 uMouse;
uniform float uMouseAmt;
uniform vec3 uBg;
uniform vec3 uInk;
uniform vec3 uInkDeep;
out vec4 fragColor;

const float BAYER[16] = float[16](0., 8., 2., 10., 12., 4., 14., 6., 3., 11., 1., 9., 15., 7., 13., 5.);

float bayer(vec2 p) {
  ivec2 i = ivec2(mod(p, 4.0));
  return (BAYER[i.x + i.y * 4] + 0.5) / 16.0;
}

void main() {
  vec2 p = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y); // y grows downward
  vec2 uv = p / uRes;
  float aspect = uRes.x / uRes.y;
  vec2 q = vec2(uv.x * aspect, uv.y);
  float t = uTime;

  // interference: three slow-drifting sources plus the pointer
  vec2 s0 = vec2(aspect * 0.18 + 0.06 * sin(t * 0.13), 0.22 + 0.05 * cos(t * 0.11));
  vec2 s1 = vec2(aspect * 0.84 + 0.07 * cos(t * 0.09), 0.30 + 0.04 * sin(t * 0.15));
  vec2 s2 = vec2(aspect * 0.52 + 0.05 * sin(t * 0.05), -0.30);
  float k = 46.0;
  float wave = cos(k * distance(q, s0) - t * 1.5)
             + cos(k * distance(q, s1) - t * 1.2)
             + 0.8 * cos(k * 0.75 * distance(q, s2) - t * 1.0);
  vec2 m = vec2(uMouse.x / uRes.x * aspect, uMouse.y / uRes.y);
  float dm = distance(q, m);
  float near = exp(-dm * 4.0) * uMouseAmt;
  wave += 1.6 * near * cos(k * 1.3 * dm - t * 3.2);
  float fringe = pow(clamp(wave * wave / 7.0, 0.0, 1.0), 1.6); // crisp rings
  float field = fringe * mix(0.2, 0.42, uv.y); // fills the page, thinning toward the top

  float v = field;
  float ink = 0.0; // 0 = deep ink, 1 = bright ink

  // frequency waves, back (higher on the page) to front; nearer ones occlude farther ones
  for (int i = 0; i < 7; i++) {
    float fi = float(i);
    float base = 0.30 + fi * 0.1;
    float amp = 0.018 + fi * 0.005;
    float f = 11.0 + fi * 3.8; // each layer its own frequency, like a spectrum
    float x = q.x;
    float y = base + amp * (sin(x * f - t * (0.9 + fi * 0.18) + fi * 1.7)
                          + 0.3 * sin(x * f * 0.37 + t * 0.4 + fi)); // carrier + slow beat
    y += 0.03 * near * sin(dm * 40.0 - t * 4.0); // the pointer ruffles the lines
    if (uv.y > y) {
      float d = uv.y - y;
      float front = fi / 6.0;
      float crest = exp(-d * uRes.y / 2.6);
      float body = (0.1 + 0.14 * front) * exp(-d * 10.0);
      float mist = 0.07 * (0.5 + 0.5 * sin(x * 2.6 + t * 0.25 + fi * 2.1 + d * 14.0)) * exp(-d * 5.0);
      mist *= 1.0 - smoothstep(0.0, 1.0, near * 2.5); // the pointer parts the mist
      v = max(crest * (0.8 + 0.2 * front), body + mist) + field * 0.6 * exp(-d * 5.0);
      ink = mix(0.35, 1.0, front) * smoothstep(0.0, 0.4, crest + body);
    }
  }

  // develop bottom-up like a print in the tray
  float develop = smoothstep(0.0, 1.0, uIntro * 1.5 - (1.0 - uv.y) * 0.5);
  v *= develop;

  float on = step(bayer(gl_FragCoord.xy), v);
  vec3 color = mix(uInkDeep, uInk, ink);
  fragColor = vec4(mix(uBg, color, on), 1.0);
}
`;

const hex = (h) => {
  const n = parseInt(h.replace("#", ""), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
};

const prefersReducedMotion = () =>
  window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

// muted amber so the waves sit behind the text instead of competing with it
export default function DitherWaves({ paused = false, bg = "#100D0B", ink = "#A3865F", inkDeep = "#45382A" }) {
  const containerRef = useRef(null);
  const pausedRef = useRef(paused);
  const wakeRef = useRef(() => {});
  pausedRef.current = paused;

  useEffect(() => { wakeRef.current(); }, [paused]);

  useEffect(() => {
    const container = containerRef.current;
    const renderer = new Renderer({ dpr: 1, alpha: false, antialias: false, depth: false });
    const gl = renderer.gl;
    const canvas = gl.canvas;
    canvas.style.cssText = "width:100%;height:100%;display:block;image-rendering:pixelated;";
    container.appendChild(canvas);

    const program = new Program(gl, {
      vertex,
      fragment,
      uniforms: {
        uRes: { value: [1, 1] },
        uTime: { value: 0 },
        uIntro: { value: 0 },
        uMouse: { value: [-9999, -9999] },
        uMouseAmt: { value: 0 },
        uBg: { value: hex(bg) },
        uInk: { value: hex(ink) },
        uInkDeep: { value: hex(inkDeep) },
      },
    });
    const mesh = new Mesh(gl, { geometry: new Triangle(gl), program });
    const u = program.uniforms;

    const resize = () => {
      const w = Math.max(1, Math.ceil(container.clientWidth / CELL));
      const h = Math.max(1, Math.ceil(container.clientHeight / CELL));
      renderer.setSize(w, h);
      canvas.style.width = "100%";
      canvas.style.height = "100%";
      u.uRes.value = [w, h];
    };
    resize();

    const still = prefersReducedMotion();
    let raf = 0;
    let last = 0;
    let time = 12; // start mid-motion so the first frame is already interesting
    let intro = still ? 1 : 0;
    const introStart = performance.now(); // wall clock, so slow frames don't stall the reveal
    let mouseTarget = 0;
    let visible = true;

    const render = () => {
      u.uTime.value = time;
      u.uIntro.value = intro;
      renderer.render({ scene: mesh });
    };

    const frame = (now) => {
      raf = 0;
      const dt = last ? Math.min(0.05, (now - last) / 1000) : 0;
      last = now;
      const moving = !pausedRef.current && !still;
      if (moving) time += dt;
      if (!still) intro = Math.min(1, (now - introStart) / 1800);
      u.uMouseAmt.value += (mouseTarget - u.uMouseAmt.value) * (1 - Math.exp(-dt / 0.25));
      render();
      const settling = intro < 1 || Math.abs(mouseTarget - u.uMouseAmt.value) > 0.01;
      if (visible && (moving || settling)) raf = requestAnimationFrame(frame);
    };
    const wake = () => {
      if (!raf && visible) { last = 0; raf = requestAnimationFrame(frame); }
    };
    wakeRef.current = wake;

    const onMove = (e) => {
      const r = container.getBoundingClientRect();
      u.uMouse.value = [(e.clientX - r.left) / CELL, (e.clientY - r.top) / CELL];
      mouseTarget = 1;
      wake();
    };
    const onLeave = () => { mouseTarget = 0; wake(); };
    const onVisibility = () => { visible = !document.hidden; wake(); };

    const ro = new ResizeObserver(() => { resize(); render(); });
    ro.observe(container);
    window.addEventListener("pointermove", onMove, { passive: true });
    document.documentElement.addEventListener("pointerleave", onLeave);
    window.addEventListener("blur", onLeave);
    document.addEventListener("visibilitychange", onVisibility);

    render();
    wake();

    return () => {
      cancelAnimationFrame(raf);
      wakeRef.current = () => {};
      ro.disconnect();
      window.removeEventListener("pointermove", onMove);
      document.documentElement.removeEventListener("pointerleave", onLeave);
      window.removeEventListener("blur", onLeave);
      document.removeEventListener("visibilitychange", onVisibility);
      gl.getExtension("WEBGL_lose_context")?.loseContext();
      canvas.remove();
    };
  }, [bg, ink, inkDeep]);

  return <div ref={containerRef} style={{ position: "absolute", inset: 0 }} />;
}
