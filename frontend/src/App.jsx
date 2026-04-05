import { useState, useEffect, useRef, useCallback } from "react";
import JSZip from "jszip";

// ─── STEM DEFINITIONS ──────────────────────────────────────
const ALL_STEMS = [
  { id: "vocals", name: "Vocals",  instrument: "Voice",         color: "#E8C468", shortcut: "V" },
  { id: "drums",  name: "Drums",   instrument: "Drums",         color: "#E86853", shortcut: "D" },
  { id: "bass",   name: "Bass",    instrument: "Bass Guitar",   color: "#6BCAB4", shortcut: "B" },
  { id: "guitar", name: "Guitar",  instrument: "Guitar",        color: "#E8924A", shortcut: "G" },
  { id: "piano",  name: "Piano",   instrument: "Piano",         color: "#A47EC8", shortcut: "P" },
  { id: "other",  name: "Other",   instrument: "Remaining Mix", color: "#7E9EC8", shortcut: "O" },
];

// Stems that are always shown regardless of classifier output
const ALWAYS_INCLUDED = new Set(["drums", "bass", "other"]);

// Map IRMAS instrument names (as returned by /api/classify) to Demucs stem ids
const CLASSIFY_TO_STEM = {
  "voice":            "vocals",
  "electric guitar":  "guitar",
  "acoustic guitar":  "guitar",
  "piano":            "piano",
};

// ─── LOADING QUIPS ─────────────────────────────────────────
const LOADING_QUIPS = [
  "Convincing the bass to leave the party…",
  "Telling the drummer to take five…",
  "Negotiating with the vocalist's ego…",
  "Finding where the guitarist hid the riff…",
  "Untangling the frequency spaghetti…",
  "Asking each instrument for ID…",
  "The model is doing its thing…",
  "Running the transformer…",
  "Teaching the model what a kick drum sounds like…",
  "Almost there — just one more solo…",
];

const BUFFERING_QUIPS = [
  "Freddie says: \"I want to break free\" (from this buffer)…",
  "Is this the real file? Is this just fantasy?",
  "We will, we will… load you.",
  "Another byte bites the dust…",
  "Don't stop me now — oh wait, buffering.",
];

// ─── HELPERS ───────────────────────────────────────────────
function formatTime(s) {
  if (!s || isNaN(s)) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

// ─── ANGUS YOUNG — DETAILED DUCK WALK ─────────────────────
function GuitaristAnim() {
  return (
    <svg viewBox="0 0 240 200" width="170" height="142" style={{ display: "block", margin: "0 auto" }}>
      <g>
        <animateTransform attributeName="transform" type="translate" values="0,0;4,-10;0,0;-2,-6;0,0" dur="0.55s" repeatCount="indefinite" />
        <g>
          <animateTransform attributeName="transform" type="rotate" values="5,90,40;-10,90,40;5,90,40" dur="0.4s" repeatCount="indefinite" />
          <path d="M73 18 Q68 6 76 2 Q84 -3 92 1 Q100 -2 105 5 Q110 2 108 13 Q112 8 109 16 Q104 10 99 14 Q93 6 88 12 Q82 5 78 13 Q72 8 73 18" fill="var(--accent)" opacity="0.8" />
          <ellipse cx="90" cy="32" rx="15" ry="17" fill="var(--surface-2)" stroke="var(--accent)" strokeWidth="2" />
          <line x1="82" y1="28" x2="87" y2="29" stroke="var(--accent)" strokeWidth="2.2" strokeLinecap="round" />
          <line x1="93" y1="29" x2="98" y2="28" stroke="var(--accent)" strokeWidth="2.2" strokeLinecap="round" />
          <ellipse cx="90" cy="40" rx="5" ry="4" fill="var(--bg)" stroke="var(--accent)" strokeWidth="1.5">
            <animate attributeName="ry" values="3;5;3" dur="0.35s" repeatCount="indefinite" />
          </ellipse>
          <path d="M73 22 L107 22 L104 15 Q90 8 76 15 Z" fill="var(--accent)" />
          <rect x="71" y="21" width="40" height="3.5" rx="1.5" fill="var(--accent)" />
        </g>
        <path d="M78 50 L74 55 L72 90 L80 93 L90 89 L100 93 L108 90 L106 55 L102 50 Z" fill="var(--surface-2)" stroke="var(--accent)" strokeWidth="1.5" />
        <polygon points="90,52 86,64 90,82 94,64" fill="var(--accent)" opacity="0.6" />
        <path d="M72 90 L68 96 M108 90 L112 96" stroke="var(--accent)" strokeWidth="1" opacity="0.4" />
        <path d="M72 90 L69 110 L84 110 L90 93 L96 110 L111 110 L108 90 Z" fill="var(--surface-2)" stroke="var(--accent)" strokeWidth="1.5" />
        <g>
          <animateTransform attributeName="transform" type="rotate" values="6,90,110;-4,90,110;6,90,110" dur="0.55s" repeatCount="indefinite" />
          <path d="M74 110 L64 132 L58 132 L68 110 Z" fill="var(--surface-2)" stroke="var(--accent)" strokeWidth="1.3" />
          <path d="M58 132 L50 158 L46 158 L54 132 Z" fill="var(--accent)" opacity="0.45" />
          <path d="M42 156 L54 155 L54 163 L40 163 Q37 163 37 160 L37 159 Q37 157 40 156 Z" fill="var(--accent)" />
        </g>
        <g>
          <animateTransform attributeName="transform" type="rotate" values="-6,90,110;4,90,110;-6,90,110" dur="0.55s" repeatCount="indefinite" />
          <path d="M102 110 L122 126 L126 122 L106 107 Z" fill="var(--surface-2)" stroke="var(--accent)" strokeWidth="1.3" />
          <path d="M122 126 L148 153 L152 150 L128 123 Z" fill="var(--accent)" opacity="0.45" />
          <path d="M146 148 L166 150 Q170 151 170 154 L168 157 L146 156 Z" fill="var(--accent)" />
        </g>
        <g>
          <path d="M118 58 L124 52 Q128 50 130 52 L132 56 Q134 60 132 64 L134 66 Q136 70 134 74 L132 78 Q130 82 126 82 L122 80 L120 82 Q116 82 114 78 L112 74 Q110 70 112 66 L114 64 Q112 60 114 56 Q116 54 118 54 Z" fill="var(--surface-2)" stroke="var(--accent)" strokeWidth="1.8" />
          <rect x="119" y="22" width="7" height="32" rx="2" fill="var(--surface-2)" stroke="var(--accent)" strokeWidth="1.5" />
          <path d="M118 22 L116 12 Q115 6 118 4 L120 2 L122.5 1 L125 2 L127 4 Q130 6 129 12 L127 22 Z" fill="var(--surface-2)" stroke="var(--accent)" strokeWidth="1.5" />
        </g>
        <g>
          <animateTransform attributeName="transform" type="rotate" values="-14,108,68;14,108,68;-14,108,68" dur="0.22s" repeatCount="indefinite" />
          <path d="M102 58 L108 64 L114 70 L118 66 L110 58 L104 54 Z" fill="var(--surface-2)" stroke="var(--accent)" strokeWidth="1.2" />
          <ellipse cx="123" cy="70" rx="3" ry="2.5" fill="var(--accent)" opacity="0.6" />
        </g>
        <path d="M78 56 L95 50 L112 38 L122 32 L122 36 L110 43 L94 53 L78 60 Z" fill="var(--surface-2)" stroke="var(--accent)" strokeWidth="1" />
      </g>
    </svg>
  );
}

// ─── FREDDIE MERCURY ───────────────────────────────────────
function FreddieAnim() {
  return (
    <svg viewBox="0 0 180 220" width="130" height="160" style={{ display: "block", margin: "0 auto" }}>
      <g>
        <animateTransform attributeName="transform" type="translate" values="-1,0;1,0;-1,0" dur="1.5s" repeatCount="indefinite" />
        <g>
          <animateTransform attributeName="transform" type="rotate" values="-3,90,42;3,90,42;-3,90,42" dur="1s" repeatCount="indefinite" />
          <path d="M74 28 Q72 16 80 12 Q88 8 96 12 Q104 16 102 28" fill="var(--accent)" opacity="0.7" />
          <ellipse cx="88" cy="36" rx="16" ry="18" fill="var(--surface-2)" stroke="var(--accent)" strokeWidth="2" />
          <ellipse cx="82" cy="32" rx="2.5" ry="1.5" fill="var(--accent)" opacity="0.8" />
          <ellipse cx="94" cy="32" rx="2.5" ry="1.5" fill="var(--accent)" opacity="0.8" />
          <path d="M80 43 Q82 46 88 44 Q94 46 96 43" fill="var(--accent)" stroke="var(--accent)" strokeWidth="1.5" strokeLinecap="round" />
          <ellipse cx="88" cy="49" rx="6" ry="5" fill="var(--bg)" stroke="var(--accent)" strokeWidth="1.5">
            <animate attributeName="ry" values="4;6;4" dur="0.8s" repeatCount="indefinite" />
          </ellipse>
        </g>
        <path d="M72 62 L68 68 L66 120 L110 120 L108 68 L104 62 Z" fill="var(--surface-2)" stroke="var(--accent)" strokeWidth="1.5" />
        <path d="M66 121 L62 185 L74 185 L88 125 L102 185 L114 185 L110 121 Z" fill="var(--surface-2)" stroke="var(--accent)" strokeWidth="1.5" />
        <path d="M58 183 L76 183 L76 195 L55 195 Q52 195 52 192 L52 188 Q52 185 55 184 Z" fill="var(--accent)" />
        <path d="M100 183 L118 183 L118 195 L97 195 Q94 195 94 192 L94 188 Q94 185 97 184 Z" fill="var(--accent)" />
        <g>
          <animateTransform attributeName="transform" type="rotate" values="-3,88,80;3,88,80;-3,88,80" dur="1.2s" repeatCount="indefinite" />
          <path d="M72 68 L62 58 L56 48 L52 38 L48 34 L44 38 L48 42 L52 50 L58 62 L68 72 Z" fill="var(--surface-2)" stroke="var(--accent)" strokeWidth="1.3" />
          <rect x="41" y="28" width="12" height="13" rx="4" fill="var(--surface-2)" stroke="var(--accent)" strokeWidth="1.5" />
        </g>
        <g>
          <path d="M104 68 L112 60 L118 52 L114 48 L106 58 L100 66 Z" fill="var(--surface-2)" stroke="var(--accent)" strokeWidth="1.2" />
          <ellipse cx="130" cy="40" rx="4" ry="5" fill="var(--surface-2)" stroke="var(--accent)" strokeWidth="1.3" />
        </g>
        <ellipse cx="88" cy="198" rx="50" ry="4" fill="var(--accent)" opacity="0.06" />
      </g>
    </svg>
  );
}

// ─── WAVEFORM ──────────────────────────────────────────────
function Waveform({ color, isActive, audioUrl, height = 44, currentTime = 0, duration = 0 }) {
  const canvasRef = useRef(null);
  const animRef = useRef(null);
  const samplesRef = useRef(null);
  const [samplesLoaded, setSamplesLoaded] = useState(false);

  useEffect(() => {
    if (!audioUrl) return;
    let cancelled = false;
    fetch(audioUrl)
      .then((r) => r.arrayBuffer())
      .then((buf) => {
        const actx = new (window.AudioContext || window.webkitAudioContext)();
        return actx.decodeAudioData(buf);
      })
      .then((decoded) => {
        if (cancelled) return;
        const channel = decoded.getChannelData(0);
        const barCount = 80;
        const blockSize = Math.floor(channel.length / barCount);
        const raw = Array.from({ length: barCount }, (_, i) => {
          let sum = 0;
          for (let j = 0; j < blockSize; j++) sum += Math.abs(channel[i * blockSize + j]);
          return sum / blockSize;
        });
        const peak = Math.max(...raw, 0.001);
        samplesRef.current = raw.map((v) => v / peak);
        setSamplesLoaded(true);
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [audioUrl]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvas.offsetWidth * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    const W = canvas.offsetWidth;
    const H = height;
    const fallback = Array.from({ length: 80 }, () => Math.random() * 0.35 + 0.1);
    const progress = duration > 0 ? currentTime / duration : 0;

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const bars = samplesRef.current || fallback;
      const barW = W / bars.length;
      bars.forEach((v, i) => {
        const amp = isActive
          ? v * (0.6 + 0.7 * Math.abs(Math.sin(Date.now() * 0.005 + i * 0.5)))
          : v;
        const barH = Math.max(2, amp * H);
        const played = i / bars.length < progress;
        ctx.fillStyle = played ? color : (isActive ? color + "70" : color + "33");
        ctx.beginPath();
        ctx.roundRect(i * barW + 1, (H - barH) / 2, barW - 2, barH, 1);
        ctx.fill();
      });
      animRef.current = requestAnimationFrame(draw);
    }
    draw();
    return () => cancelAnimationFrame(animRef.current);
  }, [color, isActive, height, samplesLoaded, currentTime, duration]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: "100%", height: `${height}px`, display: "block" }}
    />
  );
}

// ─── TRANSPORT BAR ─────────────────────────────────────────
function Transport({ isPlaying, currentTime, duration, onPlayPause, onSeekChange, onSeekCommit, activeStemIds, allStems }) {
  const pct = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div style={{
      background: "var(--surface-1)", border: "1px solid var(--border)",
      borderRadius: "10px", padding: "14px 16px", marginTop: "12px",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
        {/* Play/Pause all */}
        <button
          onClick={onPlayPause}
          disabled={duration === 0}
          style={{
            width: "34px", height: "34px", borderRadius: "50%",
            border: "none",
            background: duration > 0 ? "var(--accent)" : "var(--surface-2)",
            color: "var(--bg)", cursor: duration > 0 ? "pointer" : "default",
            fontSize: "12px", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center",
          }}
        >
          {isPlaying ? "⏸" : "▶"}
        </button>

        {/* Time */}
        <span style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-3)", flexShrink: 0, minWidth: "36px" }}>
          {formatTime(currentTime)}
        </span>

        {/* Scrub slider */}
        <div style={{ flex: 1, position: "relative" }}>
          <div style={{
            height: "3px", background: "var(--surface-2)", borderRadius: "2px",
            position: "relative", overflow: "visible",
          }}>
            <div style={{
              height: "100%", background: "var(--accent)", borderRadius: "2px",
              width: `${pct}%`, pointerEvents: "none",
            }} />
          </div>
          <input
            type="range" min={0} max={duration || 100} step={0.05}
            value={currentTime}
            onChange={(e) => onSeekChange(parseFloat(e.target.value))}
            onMouseUp={(e) => onSeekCommit(parseFloat(e.target.value))}
            onTouchEnd={(e) => onSeekCommit(parseFloat(e.target.value))}
            style={{
              position: "absolute", top: "50%", left: 0, width: "100%",
              transform: "translateY(-50%)", opacity: 0, cursor: "pointer",
              height: "20px", margin: 0,
            }}
          />
        </div>

        {/* Duration */}
        <span style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-3)", flexShrink: 0, minWidth: "36px", textAlign: "right" }}>
          {formatTime(duration)}
        </span>
      </div>

      {/* Active stem indicators */}
      {activeStemIds.size > 0 && (
        <div style={{ marginTop: "10px", display: "flex", gap: "6px", flexWrap: "wrap" }}>
          {allStems.filter(s => activeStemIds.has(s.id)).map(s => (
            <span key={s.id} style={{
              fontFamily: "var(--font-mono)", fontSize: "9px",
              color: s.color, background: s.color + "15",
              padding: "2px 7px", borderRadius: "3px",
              border: `1px solid ${s.color}30`,
            }}>
              {s.name.toLowerCase()}
            </span>
          ))}
          {activeStemIds.size > 1 && (
            <span style={{ fontFamily: "var(--font-mono)", fontSize: "9px", color: "var(--text-3)", padding: "2px 4px" }}>
              playing together
            </span>
          )}
        </div>
      )}
    </div>
  );
}

// ─── STEM CARD ─────────────────────────────────────────────
function StemCard({ stem, isActive, isMuted, audioUrl, onToggle, onMute, onSolo, currentTime, duration }) {
  const [hovered, setHovered] = useState(false);

  const handleDownload = () => {
    if (!audioUrl) return;
    const a = document.createElement("a");
    a.href = audioUrl;
    a.download = `${stem.id}_separated.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: hovered ? "var(--surface-2)" : "var(--surface-1)",
        border: `1px solid ${isActive ? stem.color + "50" : "var(--border)"}`,
        borderRadius: "10px",
        padding: "16px",
        transition: "all 0.2s ease",
        opacity: isMuted ? 0.4 : 1,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <div style={{
            width: "8px", height: "8px", borderRadius: "50%",
            background: isActive ? stem.color : stem.color + "55",
            boxShadow: isActive ? `0 0 8px ${stem.color}80` : "none",
            transition: "all 0.3s ease",
          }} />
          <span style={{ fontFamily: "var(--font-head)", fontSize: "14px", fontWeight: "600", color: "var(--text-1)" }}>
            {stem.name}
          </span>
          <span style={{
            fontFamily: "var(--font-mono)", fontSize: "10px",
            color: stem.color, background: stem.color + "15",
            padding: "2px 6px", borderRadius: "4px",
          }}>
            {stem.instrument}
          </span>
        </div>
        <div style={{ display: "flex", gap: "4px" }}>
          <button onClick={onSolo} title={`Solo ${stem.name}`} style={{
            width: "26px", height: "26px", borderRadius: "5px",
            border: "1px solid var(--border)", background: "transparent",
            color: "var(--text-3)", cursor: "pointer", fontSize: "10px",
            fontFamily: "var(--font-mono)", fontWeight: "600",
          }}>S</button>
          <button onClick={onMute} title={`Mute ${stem.name}`} style={{
            width: "26px", height: "26px", borderRadius: "5px",
            border: "1px solid var(--border)",
            background: isMuted ? "var(--accent)30" : "transparent",
            color: isMuted ? "var(--accent)" : "var(--text-3)",
            cursor: "pointer", fontSize: "10px",
            fontFamily: "var(--font-mono)", fontWeight: "600",
          }}>M</button>
          <button onClick={handleDownload} disabled={!audioUrl} title={`Download ${stem.name}`} style={{
            width: "26px", height: "26px", borderRadius: "5px",
            border: "1px solid var(--border)", background: "transparent",
            color: audioUrl ? "var(--text-3)" : "var(--text-3)44", cursor: audioUrl ? "pointer" : "default", fontSize: "11px",
          }}>&#8595;</button>
          <button onClick={onToggle} disabled={!audioUrl} style={{
            width: "26px", height: "26px", borderRadius: "5px",
            border: `1px solid ${stem.color}40`,
            background: isActive ? stem.color + "20" : "transparent",
            color: audioUrl ? stem.color : "var(--text-3)",
            cursor: audioUrl ? "pointer" : "default", fontSize: "10px",
          }}>
            {isActive ? "⏸" : "▶"}
          </button>
        </div>
      </div>

      <Waveform
        color={stem.color}
        isActive={isActive && !isMuted}
        audioUrl={audioUrl}
        height={38}
        currentTime={currentTime}
        duration={duration}
      />

      {hovered && audioUrl && (
        <div style={{ marginTop: "8px", fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--text-3)" }}>
          press <span style={{ color: stem.color }}>{stem.shortcut}</span> to toggle
        </div>
      )}
    </div>
  );
}

// ─── DROP ZONE ─────────────────────────────────────────────
function DropZone({ onFileSelect, isDragging, setIsDragging }) {
  const fileRef = useRef(null);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && (file.type.startsWith("audio/") || file.name.match(/\.(wav|mp3|flac|ogg|m4a)$/i))) {
      onFileSelect(file);
    }
  }, [onFileSelect, setIsDragging]);

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onClick={() => fileRef.current?.click()}
      style={{
        border: `2px dashed ${isDragging ? "var(--accent)" : "var(--border)"}`,
        borderRadius: "12px",
        padding: "48px 32px",
        textAlign: "center",
        cursor: "pointer",
        transition: "all 0.25s ease",
        background: isDragging ? "var(--accent-dim)" : "transparent",
      }}
    >
      <input
        ref={fileRef} type="file" accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a"
        style={{ display: "none" }}
        onChange={(e) => e.target.files[0] && onFileSelect(e.target.files[0])}
      />
      <div style={{ fontFamily: "var(--font-head)", fontSize: "16px", fontWeight: "600", color: "var(--text-1)", marginBottom: "6px" }}>
        {isDragging ? "Yeah, right there. Drop it." : "Throw a track in here"}
      </div>
      <div style={{ color: "var(--text-3)", fontSize: "13px", fontFamily: "var(--font-body)" }}>
        wav · mp3 · flac · ogg — whatever you've got
      </div>
    </div>
  );
}

// ─── PROCESSING VIEW ───────────────────────────────────────
function ProcessingView({ fileName, progress, modelDone }) {
  const [quipIdx, setQuipIdx] = useState(0);

  useEffect(() => {
    const t = setInterval(() => setQuipIdx((i) => (i + 1) % LOADING_QUIPS.length), 3000);
    return () => clearInterval(t);
  }, []);

  const nearlyDone = progress >= 80;
  // If progress is 0 and we haven't started receiving data yet, show indeterminate bar
  const indeterminate = progress === 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "8px", padding: "20px 0" }}>
      {nearlyDone ? <FreddieAnim /> : <GuitaristAnim />}

      <div style={{ fontFamily: "var(--font-mono)", fontSize: "12px", color: "var(--accent)", textAlign: "center", minHeight: "20px" }}>
        {nearlyDone ? BUFFERING_QUIPS[Math.floor(progress / 10) % BUFFERING_QUIPS.length] : modelDone ? "Almost there — just one more solo…" : LOADING_QUIPS[quipIdx]}
      </div>

      <div style={{
        width: "100%", maxWidth: "340px", height: "3px",
        background: "var(--surface-2)", borderRadius: "2px",
        overflow: "hidden", marginTop: "8px", position: "relative",
      }}>
        {indeterminate ? (
          <div style={{
            position: "absolute", left: 0, top: 0, height: "100%",
            width: "30%", background: "var(--accent)", borderRadius: "2px",
            animation: "indeterminate 1.4s ease-in-out infinite",
          }} />
        ) : (
          <div style={{
            height: "100%", borderRadius: "2px", background: "var(--accent)",
            width: `${progress}%`, transition: "width 0.3s ease",
          }} />
        )}
      </div>

      <span style={{ fontFamily: "var(--font-mono)", fontSize: "22px", fontWeight: "600", color: "var(--text-1)", marginTop: "4px" }}>
        {indeterminate ? "—" : `${progress}%`}
      </span>

      <div style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-3)" }}>
        {fileName} · separating stems
      </div>
    </div>
  );
}

// ─── MAIN APP ──────────────────────────────────────────────
export default function App() {
  const [state, setState] = useState("idle");     // idle | processing | done | error
  const [phase, setPhase] = useState("separating"); // kept for potential future use
  const [file, setFile] = useState(null);
  const [activeStems, setActiveStems] = useState(ALL_STEMS); // stems shown in done state
  const [progress, setProgress] = useState(0);
  const [modelDone, setModelDone] = useState(false);
  const [stemUrls, setStemUrls] = useState({});
  const [activeStemIds, setActiveStemIds] = useState(new Set()); // which stems are playing
  const [mutedIds, setMutedIds] = useState(new Set());
  const [soloId, setSoloId] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [easterEggClicks, setEasterEggClicks] = useState(0);
  const [showEaster, setShowEaster] = useState(false);
  const [error, setError] = useState(null);

  // Web Audio API refs — sample-accurate multi-stem sync
  const audioCtxRef = useRef(null);         // single shared AudioContext
  const audioBuffersRef = useRef({});       // { [stemId]: AudioBuffer } decoded once on load
  const gainNodesRef = useRef({});          // { [stemId]: GainNode } persistent, used for mute
  const sourceNodesRef = useRef({});        // { [stemId]: AudioBufferSourceNode } fresh each play
  const audioOffsetRef = useRef(0);         // song position (s) at last play/pause/seek event
  const ctxTimeAtStartRef = useRef(0);      // audioCtx.currentTime value when last playback began
  const isPlayingRef = useRef(false);       // mirror of isPlaying state for use inside rAF
  const lastPlayedStemIdsRef = useRef(new Set()); // stems that were active before last pause

  const abortRef = useRef(null);
  const rafRef = useRef(null);

  // ── Decode stems into AudioBuffers whenever stemUrls are set ──
  useEffect(() => {
    if (Object.keys(stemUrls).length === 0) return;

    const ctx = new AudioContext();
    audioCtxRef.current = ctx;
    audioBuffersRef.current = {};
    gainNodesRef.current = {};
    sourceNodesRef.current = {};
    audioOffsetRef.current = 0;
    ctxTimeAtStartRef.current = 0;

    // Create one persistent GainNode per stem (for mute/solo)
    activeStems.forEach(stem => {
      const gain = ctx.createGain();
      gain.gain.value = 1;
      gain.connect(ctx.destination);
      gainNodesRef.current[stem.id] = gain;
    });

    // Decode each blob into an AudioBuffer
    activeStems.forEach(stem => {
      const url = stemUrls[stem.id];
      if (!url) return;
      fetch(url)
        .then(r => r.arrayBuffer())
        .then(buf => ctx.decodeAudioData(buf))
        .then(audioBuffer => {
          audioBuffersRef.current[stem.id] = audioBuffer;
          setDuration(prev => prev || audioBuffer.duration);
        })
        .catch(() => {});
    });

    return () => {
      Object.values(sourceNodesRef.current).forEach(node => { try { node?.stop(0); } catch {} });
      sourceNodesRef.current = {};
      ctx.close();
    };
  }, [stemUrls]);

  // ── rAF loop — compute currentTime from audio clock, not from <audio>.currentTime ──
  useEffect(() => {
    const tick = () => {
      if (isPlayingRef.current && audioCtxRef.current) {
        const elapsed = audioCtxRef.current.currentTime - ctxTimeAtStartRef.current;
        setCurrentTime(audioOffsetRef.current + Math.max(0, elapsed));
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, []); // empty — reads only refs

  // ── Keyboard shortcuts ──────────────────────────────────
  useEffect(() => {
    const handler = (e) => {
      if (state !== "done") return;
      const key = e.key.toUpperCase();
      const stem = activeStems.find((s) => s.shortcut === key);
      if (stem && audioBuffersRef.current[stem.id]) toggleStemPlay(stem.id);
      if (e.key === " ") { e.preventDefault(); handlePlayPauseAll(); }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [state, activeStems, activeStemIds, isPlaying]);

  // ── Transport controls ──────────────────────────────────
  const handlePlayPauseAll = useCallback(async () => {
    const ctx = audioCtxRef.current;
    if (!ctx) return;

    if (isPlaying) {
      // Pause: record current song position and which stems were active
      const elapsed = ctx.currentTime - ctxTimeAtStartRef.current;
      audioOffsetRef.current += Math.max(0, elapsed);
      lastPlayedStemIdsRef.current = new Set(activeStemIds); // remember for resume
      Object.values(sourceNodesRef.current).forEach(node => { try { node?.stop(0); } catch {} });
      sourceNodesRef.current = {};
      isPlayingRef.current = false;
      setIsPlaying(false);
      setActiveStemIds(new Set());
    } else {
      await ctx.resume(); // handle browsers that suspend AudioContext until user gesture

      // Resume only the stems that were playing before pause; on first play use all stems
      const stemsToPlay = lastPlayedStemIdsRef.current.size > 0
        ? activeStems.filter(s => lastPlayedStemIdsRef.current.has(s.id))
        : activeStems;

      // Schedule all stems to start at the SAME future audio-clock tick → sample-accurate sync
      const AHEAD = 0.05; // 50 ms gives the browser time to prepare all sources
      const startAt = ctx.currentTime + AHEAD;
      const offset = audioOffsetRef.current;
      ctxTimeAtStartRef.current = startAt;

      const newActiveIds = new Set();
      stemsToPlay.forEach(stem => {
        const buffer = audioBuffersRef.current[stem.id];
        const gain = gainNodesRef.current[stem.id];
        if (!buffer || !gain) return;
        if (offset >= buffer.duration) return; // seek past end — skip

        gain.gain.value = mutedIds.has(stem.id) ? 0 : 1;

        const source = ctx.createBufferSource();
        source.buffer = buffer;
        source.connect(gain);
        source.onended = () => {
          setActiveStemIds(prev => {
            const n = new Set(prev);
            n.delete(stem.id);
            if (n.size === 0) { isPlayingRef.current = false; setIsPlaying(false); }
            return n;
          });
          delete sourceNodesRef.current[stem.id];
        };
        source.start(startAt, offset); // startAt is identical for every stem → perfect sync
        sourceNodesRef.current[stem.id] = source;
        newActiveIds.add(stem.id);
      });

      isPlayingRef.current = true;
      setActiveStemIds(newActiveIds);
      setIsPlaying(true);
    }
  }, [isPlaying, activeStems, activeStemIds, mutedIds]);

  // Called on every drag tick — only update display, don't touch audio sources
  const handleSeekDisplay = useCallback((time) => {
    setCurrentTime(time);
    audioOffsetRef.current = time;
  }, []);

  // Called on pointer-up — restart audio at new position if playing
  const handleSeekCommit = useCallback(async (time) => {
    setCurrentTime(time);
    audioOffsetRef.current = time;

    const ctx = audioCtxRef.current;
    if (!ctx || !isPlaying) return;

    // Stop current sources and restart from the new position, keeping the same stems active
    Object.values(sourceNodesRef.current).forEach(node => { try { node?.stop(0); } catch {} });
    sourceNodesRef.current = {};

    await ctx.resume();
    const AHEAD = 0.05;
    const startAt = ctx.currentTime + AHEAD;
    ctxTimeAtStartRef.current = startAt;

    const newActiveIds = new Set();
    activeStems.forEach(stem => {
      if (!activeStemIds.has(stem.id)) return;
      const buffer = audioBuffersRef.current[stem.id];
      const gain = gainNodesRef.current[stem.id];
      if (!buffer || !gain || time >= buffer.duration) return;

      gain.gain.value = mutedIds.has(stem.id) ? 0 : 1;
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(gain);
      source.onended = () => {
        setActiveStemIds(prev => {
          const n = new Set(prev);
          n.delete(stem.id);
          if (n.size === 0) { isPlayingRef.current = false; setIsPlaying(false); }
          return n;
        });
        delete sourceNodesRef.current[stem.id];
      };
      source.start(startAt, time);
      sourceNodesRef.current[stem.id] = source;
      newActiveIds.add(stem.id);
    });
    setActiveStemIds(newActiveIds);
  }, [isPlaying, activeStems, activeStemIds, mutedIds]);

  // ── Individual stem toggle ──────────────────────────────
  const toggleStemPlay = useCallback(async (id) => {
    const ctx = audioCtxRef.current;
    if (!ctx) return;

    if (activeStemIds.has(id)) {
      // Save current playback position before stopping so resume starts from here
      const elapsed = ctx.currentTime - ctxTimeAtStartRef.current;
      audioOffsetRef.current += Math.max(0, elapsed);
      ctxTimeAtStartRef.current = ctx.currentTime;

      try { sourceNodesRef.current[id]?.stop(0); } catch {}
      delete sourceNodesRef.current[id];
      setActiveStemIds(prev => {
        const n = new Set(prev);
        n.delete(id);
        if (n.size === 0) { isPlayingRef.current = false; setIsPlaying(false); }
        return n;
      });
    } else {
      const buffer = audioBuffersRef.current[id];
      const gain = gainNodesRef.current[id];
      if (!buffer || !gain) return;

      await ctx.resume();
      const AHEAD = 0.05;
      const startAt = ctx.currentTime + AHEAD;

      // Compute the correct buffer offset to stay in sync with any currently playing stems
      let offset;
      if (isPlaying) {
        const elapsed = ctx.currentTime - ctxTimeAtStartRef.current;
        offset = audioOffsetRef.current + Math.max(0, elapsed) + AHEAD;
      } else {
        offset = audioOffsetRef.current;
        // First stem to start — anchor the time reference
        ctxTimeAtStartRef.current = startAt;
        isPlayingRef.current = true;
        setIsPlaying(true);
      }

      if (offset >= buffer.duration) return;

      gain.gain.value = mutedIds.has(id) ? 0 : 1;
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(gain);
      source.onended = () => {
        setActiveStemIds(prev => {
          const n = new Set(prev);
          n.delete(id);
          if (n.size === 0) { isPlayingRef.current = false; setIsPlaying(false); }
          return n;
        });
        delete sourceNodesRef.current[id];
      };
      source.start(startAt, offset);
      sourceNodesRef.current[id] = source;

      setActiveStemIds(prev => { const n = new Set(prev); n.add(id); return n; });
    }
  }, [activeStemIds, isPlaying, mutedIds]);

  // ── Mute/Solo ───────────────────────────────────────────
  const toggleMute = (id) => {
    setMutedIds((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      const gain = gainNodesRef.current[id];
      if (gain) gain.gain.value = next.has(id) ? 0 : 1;
      return next;
    });
    setSoloId(null);
  };

  const toggleSolo = (id) => {
    if (soloId === id) {
      setSoloId(null);
      setMutedIds(new Set());
      activeStems.forEach(s => {
        const gain = gainNodesRef.current[s.id];
        if (gain) gain.gain.value = 1;
      });
    } else {
      setSoloId(id);
      setMutedIds(new Set(activeStems.filter(s => s.id !== id).map(s => s.id)));
      activeStems.forEach(s => {
        const gain = gainNodesRef.current[s.id];
        if (gain) gain.gain.value = s.id === id ? 1 : 0;
      });
    }
  };

  // ── Main file processing ────────────────────────────────
  const handleFileSelect = async (f) => {
    Object.values(stemUrls).forEach(url => URL.revokeObjectURL(url));
    if (audioCtxRef.current) {
      Object.values(sourceNodesRef.current).forEach(node => { try { node?.stop(0); } catch {} });
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    audioBuffersRef.current = {};
    gainNodesRef.current = {};
    sourceNodesRef.current = {};
    audioOffsetRef.current = 0;
    ctxTimeAtStartRef.current = 0;
    isPlayingRef.current = false;

    setFile(f);
    setState("processing");
    setPhase("separating");
    setProgress(0);
    setModelDone(false);
    setStemUrls({});
    setError(null);
    setActiveStemIds(new Set());
    setIsPlaying(false);
    setMutedIds(new Set());
    setSoloId(null);
    setCurrentTime(0);
    setDuration(0);
    setActiveStems(ALL_STEMS);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      // Step 1: separate all stems (one Demucs inference run)
      const sepForm = new FormData();
      sepForm.append("file", f);
      const sepRes = await fetch(`${import.meta.env.VITE_API_URL || ""}/api/separate_all`, {
        method: "POST", body: sepForm, signal: controller.signal,
      });
      if (!sepRes.ok) {
        const err = await sepRes.json().catch(() => ({ detail: sepRes.statusText }));
        throw new Error(err.detail || "Separation failed");
      }
      setModelDone(true);

      // Step 2: stream the response body to show real download progress (0→80%)
      const contentLength = parseInt(sepRes.headers.get("content-length") || "0");
      const reader = sepRes.body.getReader();
      const chunks = [];
      let received = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;
        if (contentLength > 0) {
          setProgress(Math.round((received / contentLength) * 80));
        }
      }

      // Step 3: unzip (80→100%)
      const zipBlob = new Blob(chunks);
      const zip = await JSZip.loadAsync(zipBlob);
      const urls = {};
      let done = 0;
      for (const stem of ALL_STEMS) {
        const entry = zip.file(`${stem.id}.wav`);
        if (entry) {
          const wav = await entry.async("blob");
          urls[stem.id] = URL.createObjectURL(wav);
        }
        done++;
        setProgress(80 + Math.round((done / ALL_STEMS.length) * 20));
      }
      setStemUrls(urls);
      setTimeout(() => setState("done"), 400);

    } catch (err) {
      if (err.name === "AbortError") return;
      setState("error");
      setError(err.message);
    }
  };

  // ── Reset ───────────────────────────────────────────────
  const handleReset = () => {
    abortRef.current?.abort();
    Object.values(stemUrls).forEach(url => URL.revokeObjectURL(url));
    if (audioCtxRef.current) {
      Object.values(sourceNodesRef.current).forEach(node => { try { node?.stop(0); } catch {} });
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    audioBuffersRef.current = {};
    gainNodesRef.current = {};
    sourceNodesRef.current = {};
    audioOffsetRef.current = 0;
    ctxTimeAtStartRef.current = 0;
    isPlayingRef.current = false;
    setState("idle");
    setFile(null);
    setProgress(0);
    setStemUrls({});
    setActiveStemIds(new Set());
    setIsPlaying(false);
    setMutedIds(new Set());
    setSoloId(null);
    setCurrentTime(0);
    setDuration(0);
    setError(null);
    setActiveStems(ALL_STEMS);
  };

  // ── Download all ─────────────────────────────────────────
  const downloadAll = () => {
    activeStems.forEach((stem, i) => {
      const url = stemUrls[stem.id];
      if (!url) return;
      setTimeout(() => {
        const a = document.createElement("a");
        a.href = url;
        a.download = `${stem.id}_separated.wav`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      }, i * 200);
    });
  };

  // ── Easter egg ──────────────────────────────────────────
  const handleLogoClick = () => {
    handleReset();
    const next = easterEggClicks + 1;
    setEasterEggClicks(next);
    if (next >= 7) {
      setShowEaster(true);
      setEasterEggClicks(0);
      setTimeout(() => setShowEaster(false), 4000);
    }
  };

  // ─────────────────────────────────────────────────────────
  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=JetBrains+Mono:wght@400;500&family=Sora:wght@400;500;600;700&display=swap');

        :root {
          --bg: #0A0A0D;
          --surface-1: #121216;
          --surface-2: #1A1A20;
          --border: #262630;
          --text-1: #E4E2ED;
          --text-2: #8F8DA0;
          --text-3: #504E5E;
          --accent: #E8C468;
          --accent-dim: #E8C46810;
          --font-head: 'Sora', sans-serif;
          --font-body: 'DM Sans', sans-serif;
          --font-mono: 'JetBrains Mono', monospace;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: var(--bg); color: var(--text-1); font-family: var(--font-body); }
        ::selection { background: var(--accent); color: var(--bg); }
        button { transition: all 0.15s ease; }
        button:hover { filter: brightness(1.2); }
        button:active { transform: scale(0.97); }

        input[type=range] { -webkit-appearance: none; appearance: none; background: transparent; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 12px; height: 12px; border-radius: 50%; background: var(--accent); cursor: pointer; }
        input[type=range]::-moz-range-thumb { width: 12px; height: 12px; border-radius: 50%; background: var(--accent); cursor: pointer; border: none; }

        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes easterPop {
          0% { transform: scale(0) rotate(-20deg) translate(-50%, -50%); opacity: 0; }
          50% { transform: scale(1.15) rotate(5deg) translate(-50%, -50%); opacity: 1; }
          100% { transform: scale(1) rotate(0deg) translate(-50%, -50%); opacity: 1; }
        }
        @keyframes easterFade {
          0%, 80% { opacity: 1; }
          100% { opacity: 0; }
        }
        @keyframes indeterminate {
          0%   { transform: translateX(-100%); }
          100% { transform: translateX(500%); }
        }
        .stem-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        @media (max-width: 580px) { .stem-grid { grid-template-columns: 1fr; } }
      `}</style>

      <div style={{ minHeight: "100vh", padding: "20px", maxWidth: "700px", margin: "0 auto" }}>

        {showEaster && (
          <div style={{
            position: "fixed", top: "50%", left: "50%",
            transform: "translate(-50%, -50%)", zIndex: 100,
            background: "var(--surface-1)", border: "1px solid var(--accent)",
            borderRadius: "16px", padding: "32px 40px", textAlign: "center",
            animation: "easterPop 0.4s ease, easterFade 4s ease forwards",
            boxShadow: "0 0 60px #E8C46820",
          }}>
            <div style={{ fontSize: "36px", marginBottom: "12px" }}>🎸</div>
            <div style={{ fontFamily: "var(--font-head)", fontSize: "15px", color: "var(--accent)", fontWeight: "600" }}>
              You found the secret chord.
            </div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-3)", marginTop: "6px" }}>
              David played it and it pleased the Lord.
            </div>
          </div>
        )}

        <header style={{
          display: "flex", alignItems: "center", justifyContent: "space-between",
          marginBottom: "40px", animation: "fadeIn 0.4s ease",
        }}>
          <div onClick={handleLogoClick} style={{ display: "flex", alignItems: "baseline", gap: "6px", cursor: "pointer", userSelect: "none" }}>
            <span style={{ fontFamily: "var(--font-head)", fontSize: "28px", fontWeight: "700", color: "var(--text-1)", letterSpacing: "-0.5px" }}>riff</span>
            <span style={{ fontFamily: "var(--font-head)", fontSize: "28px", fontWeight: "700", color: "var(--accent)", letterSpacing: "-0.5px" }}>apart</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            {state !== "idle" && (
              <button onClick={handleReset} style={{
                padding: "7px 14px", borderRadius: "7px", border: "1px solid var(--border)",
                background: "transparent", color: "var(--text-2)", fontSize: "12px",
                cursor: "pointer", fontFamily: "var(--font-mono)",
              }}>new track</button>
            )}
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--text-3)", letterSpacing: "0.5px" }}>v0.2</div>
          </div>
        </header>

        <main style={{ animation: "fadeIn 0.5s ease 0.05s both" }}>

          {state === "idle" && (
            <div>
              <div style={{ marginBottom: "32px" }}>
                <h1 style={{
                  fontFamily: "var(--font-head)", fontSize: "32px", fontWeight: "700",
                  letterSpacing: "-1px", lineHeight: "1.15", color: "var(--text-1)", marginBottom: "10px",
                }}>
                  Pull a song apart,<br />
                  <span style={{ color: "var(--accent)" }}>hear what's inside.</span>
                </h1>
                <p style={{ color: "var(--text-2)", fontSize: "14px", lineHeight: "1.6", maxWidth: "400px" }}>
                  Drop any track. The model identifies and isolates each instrument — vocals,
                  drums, bass, guitar, piano, and more. You get the stems.
                </p>
              </div>
              <DropZone onFileSelect={handleFileSelect} isDragging={isDragging} setIsDragging={setIsDragging} />
              <div style={{ marginTop: "20px", fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--text-3)", display: "flex", gap: "16px", flexWrap: "wrap" }}>
                {ALL_STEMS.map(s => (
                  <span key={s.id} style={{ color: s.color }}>{s.name}</span>
                ))}
              </div>
            </div>
          )}

          {state === "processing" && (
            <ProcessingView fileName={file?.name || ""} progress={progress} modelDone={modelDone} />
          )}

          {state === "error" && (
            <div style={{ textAlign: "center", padding: "48px 0", animation: "fadeIn 0.4s ease" }}>
              <div style={{ fontFamily: "var(--font-head)", fontSize: "18px", fontWeight: "700", color: "var(--text-1)", marginBottom: "8px" }}>
                Something went wrong.
              </div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "12px", color: "#E86853", marginBottom: "24px", maxWidth: "400px", margin: "0 auto 24px" }}>
                {error}
              </div>
              <button onClick={handleReset} style={{
                padding: "8px 20px", borderRadius: "8px", border: "1px solid var(--border)",
                background: "transparent", color: "var(--text-2)", fontSize: "12px",
                cursor: "pointer", fontFamily: "var(--font-mono)",
              }}>try again</button>
            </div>
          )}

          {state === "done" && (
            <div style={{ animation: "slideUp 0.4s ease" }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "16px" }}>
                <div>
                  <div style={{ fontFamily: "var(--font-head)", fontSize: "18px", fontWeight: "700", color: "var(--text-1)", letterSpacing: "-0.3px" }}>
                    Done. Here's what was in there.
                  </div>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-3)", marginTop: "3px" }}>
                    {file?.name} · {activeStems.length} stems · keyboard shortcuts active
                  </div>
                </div>
                <div style={{ display: "flex", gap: "8px" }}>
                  <button onClick={handlePlayPauseAll} style={{
                    padding: "8px 16px", borderRadius: "8px", border: "1px solid var(--accent)",
                    background: isPlaying ? "var(--accent)20" : "transparent",
                    color: "var(--accent)", fontSize: "12px", fontWeight: "600", cursor: "pointer", fontFamily: "var(--font-mono)",
                  }}>{isPlaying ? "⏸ pause all" : "▶ play all"}</button>
                  <button onClick={downloadAll} style={{
                    padding: "8px 16px", borderRadius: "8px", border: "none", background: "var(--accent)",
                    color: "var(--bg)", fontSize: "12px", fontWeight: "600", cursor: "pointer", fontFamily: "var(--font-mono)",
                  }}>&#8595; all stems</button>
                </div>
              </div>

              <div className="stem-grid">
                {activeStems.map((stem, i) => (
                  <div key={stem.id} style={{ animation: `slideUp 0.35s ease ${i * 0.07}s both` }}>
                    <StemCard
                      stem={stem}
                      isActive={activeStemIds.has(stem.id)}
                      isMuted={mutedIds.has(stem.id)}
                      audioUrl={stemUrls[stem.id]}
                      onToggle={() => toggleStemPlay(stem.id)}
                      onMute={() => toggleMute(stem.id)}
                      onSolo={() => toggleSolo(stem.id)}
                      currentTime={currentTime}
                      duration={duration}
                    />
                  </div>
                ))}
              </div>

              <Transport
                isPlaying={isPlaying}
                currentTime={currentTime}
                duration={duration}
                onPlayPause={handlePlayPauseAll}
                onSeekChange={handleSeekDisplay}
                onSeekCommit={handleSeekCommit}
                activeStemIds={activeStemIds}
                allStems={activeStems}
              />

              <div style={{
                marginTop: "10px", padding: "12px 16px", background: "var(--surface-1)",
                borderRadius: "8px", border: "1px solid var(--border)",
                display: "flex", justifyContent: "space-between", alignItems: "center",
                fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--text-3)",
                flexWrap: "wrap", gap: "8px",
              }}>
                <span style={{ display: "flex", flexWrap: "wrap", gap: "0" }}>
                  {activeStems.map((s) => (
                    <span key={s.id} style={{ marginRight: "14px" }}>
                      <span style={{
                        display: "inline-block", width: "18px", height: "18px", lineHeight: "18px",
                        textAlign: "center", border: "1px solid var(--border)", borderRadius: "3px",
                        color: s.color, marginRight: "4px", fontSize: "10px",
                      }}>{s.shortcut}</span>
                      {s.name.toLowerCase()}
                    </span>
                  ))}
                </span>
                <span>
                  <span style={{
                    display: "inline-block", padding: "1px 6px", border: "1px solid var(--border)",
                    borderRadius: "3px", marginRight: "4px", fontSize: "9px",
                  }}>space</span>
                  play / pause all
                </span>
              </div>
            </div>
          )}
        </main>

        <footer style={{
          marginTop: "72px", paddingTop: "20px", borderTop: "1px solid var(--border)",
          fontFamily: "var(--font-mono)", textAlign: "center",
        }}>
          <div style={{ fontSize: "13px", color: "#E8C468" }}>
            built by a guitarist who got tired of bad stems
          </div>
          <div style={{ fontSize: "11px", color: "#888888", marginTop: "6px" }}>
            (yellow font btw)
          </div>
        </footer>
      </div>
    </>
  );
}
