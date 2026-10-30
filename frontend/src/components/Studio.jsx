import { memo, useEffect, useRef, useState } from "react";
import useScramble from "./useScramble.js";

const NUDGE = 5; // seconds for rewind / forward and arrow keys

function clock(s, tenths = false) {
  if (!s || Number.isNaN(s)) s = 0;
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  const base = `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
  return tenths ? { base, tenth: Math.floor((s % 1) * 10) } : base;
}

// Pick a ruler step that leaves roughly 90px between labels
function rulerStep(duration, width) {
  const target = (duration / Math.max(width, 1)) * 90;
  return [1, 2, 5, 10, 15, 30, 60, 120, 300].find((s) => s >= target) ?? 600;
}

const mixHex = (hex, other, t) => {
  const a = parseInt(hex.slice(1), 16), b = parseInt(other.slice(1), 16);
  const ch = (n, sh) => (n >> sh) & 255;
  const c = (sh) => Math.round(ch(a, sh) + (ch(b, sh) - ch(a, sh)) * t);
  return `rgb(${c(16)}, ${c(8)}, ${c(0)})`;
};

// ── icons ────────────────────────────────────────────────
const Icon = ({ children, size = 18 }) => (
  <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth="1.8"
    strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{children}</svg>
);
const STEM_ICONS = {
  vocals: <><rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21M8.5 21h7" /></>,
  drums: <><ellipse cx="12" cy="9" rx="8" ry="3" /><path d="M4 9v6c0 1.7 3.6 3 8 3s8-1.3 8-3V9M8 3l3 5M16 3l-3 5" /></>,
  bass: <><path d="M14.5 9.5 20 4M18 2l4 4" /><path d="M11 9.6a4 4 0 0 0-5.6 1.2L4 13a3.6 3.6 0 0 0 5 5l2.2-1.4a4 4 0 0 0 1.2-5.6Z" /><path d="m9.5 13.5 1.5 1.5" /></>,
  guitar: <><path d="m13 11 7-7M18.5 2.5l3 3" /><path d="M10.3 10.8a3.5 3.5 0 0 0-4.6.3L4.4 12.4a4 4 0 0 0 5.7 5.7l1.3-1.3a3.5 3.5 0 0 0 .3-4.6l1-1-1.4-1.4Z" /><circle cx="8.3" cy="14.3" r="1" /></>,
  piano: <><rect x="3" y="5" width="18" height="14" rx="2" /><path d="M8 5v9M12 5v9M16 5v9M8 14v5M12 14v5M16 14v5" /></>,
  other: <><path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6" /></>,
};

// ── region waveform (static; redraws only when size or data changes) ──
const RegionWave = memo(function RegionWave({ peaks, color }) {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !peaks) return undefined;
    const draw = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = canvas.clientWidth, h = canvas.clientHeight;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const buckets = peaks.length / 2;
      let loudest = 0;
      for (let i = 0; i < peaks.length; i++) loudest = Math.max(loudest, Math.abs(peaks[i]));
      const scale = 1 / Math.max(loudest, 0.25); // quiet stems stay visibly quiet
      const mid = h / 2;
      ctx.fillStyle = color;
      for (let x = 0; x < w; x++) {
        const from = Math.floor((x / w) * buckets);
        const to = Math.max(from + 1, Math.floor(((x + 1) / w) * buckets));
        let min = 0, max = 0;
        for (let b = from; b < to; b++) {
          min = Math.min(min, peaks[b * 2]);
          max = Math.max(max, peaks[b * 2 + 1]);
        }
        const top = mid - Math.pow(max * scale, 0.8) * mid * 0.92;
        const bottom = mid - Math.pow(Math.abs(min) * scale, 0.8) * -mid * 0.92;
        ctx.fillRect(x, top, 1, Math.max(1, bottom - top));
      }
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [peaks, color]);
  return peaks
    ? <canvas ref={ref} className="region__wave" aria-hidden="true" />
    : <span className="region__loading">decoding…</span>;
});

// ── track header ─────────────────────────────────────────
const TrackHead = memo(function TrackHead({ stem, muted, solo, playingAlone, volume, meterElsRef, onPlayStem, onMute, onSolo, onVolume, onDownload }) {
  return (
    <div className={`track-head${muted ? " is-muted" : ""}${solo ? " is-solo" : ""}`} style={{ "--stem": stem.color }}>
      <span className="track-head__icon"><Icon size={20}>{STEM_ICONS[stem.id]}</Icon></span>
      <div className="track-head__main">
        <div className="track-head__top">
          <h3 className="track-head__name">{stem.name}</h3>
          <div className="track-head__btns">
            <button
              type="button" className="ms ms--play" aria-pressed={playingAlone}
              aria-label={playingAlone ? `Pause ${stem.name}` : `Play ${stem.name} only`}
              title={playingAlone ? "Pause" : `Play ${stem.name} only`} onClick={() => onPlayStem(stem.id)}
            >
              {playingAlone
                ? <Icon size={11}><path d="M8 5v14M16 5v14" strokeWidth="3.4" /></Icon>
                : <svg viewBox="0 0 24 24" width="11" height="11" aria-hidden="true"><path d="M7 4.5v15l13-7.5-13-7.5Z" fill="currentColor" /></svg>}
            </button>
            <button type="button" className="ms" aria-pressed={muted} aria-label={`Mute ${stem.name}`} title={`Mute (${stem.shortcut})`} onClick={() => onMute(stem.id)}>M</button>
            <button type="button" className="ms ms--solo" aria-pressed={solo} aria-label={`Solo ${stem.name}`} title={`Solo (Shift+${stem.shortcut})`} onClick={() => onSolo(stem.id)}>S</button>
            <button type="button" className="ms ms--dl" aria-label={`Download ${stem.name}`} title="Download MP3" onClick={() => onDownload(stem)}>
              <Icon size={13}><path d="M12 4v11M7 10l5 5 5-5M5 20h14" /></Icon>
            </button>
          </div>
        </div>
        <label className="track-head__vol">
          <span className="sr-only">{stem.name} volume</span>
          <input
            type="range" min={0} max={1.5} step={0.01} value={volume}
            onChange={(e) => onVolume(stem.id, parseFloat(e.target.value))}
            style={{ "--fill": `${(volume / 1.5) * 100}%` }}
            aria-valuetext={`${Math.round(volume * 100)}%`}
          />
        </label>
      </div>
      <span className="meter" aria-hidden="true" ref={(el) => { meterElsRef.current[stem.id] = el; }}><i /></span>
    </div>
  );
});

// ── lane with its region ─────────────────────────────────
const Lane = memo(function Lane({ stem, peaks, muted }) {
  const waveColor = mixHex(stem.color, "#FFFFFF", 0.62);
  return (
    <div className={`lane${muted ? " is-muted" : ""}`}>
      <div className="region" style={{ "--stem": stem.color }}>
        <span className="region__label">{stem.name}</span>
        <RegionWave peaks={peaks} color={waveColor} />
      </div>
    </div>
  );
});

export default function Studio({
  stems, fileName, revealed, peaks, duration, currentTime, isPlaying, loop, mutedIds, soloId, volumes, masterVolume,
  meterElsRef, onPlayPause, onToStart, onLoop, onSeekDisplay, onSeekCommit, onPlayStem, onMute, onSolo, onVolume, onMaster,
  onDownload, onDownloadAll,
}) {
  const timelineRef = useRef(null);
  const draggingRef = useRef(false);
  const [width, setWidth] = useState(800);
  const title = useScramble(fileName.replace(/\.[^.]+$/, ""), revealed);

  useEffect(() => {
    const el = timelineRef.current;
    if (!el) return undefined;
    const ro = new ResizeObserver(([e]) => setWidth(e.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const timeAt = (clientX) => {
    const r = timelineRef.current.getBoundingClientRect();
    return Math.min(duration, Math.max(0, ((clientX - r.left) / r.width) * duration));
  };
  const onPointerDown = (e) => {
    if (!duration || e.button !== 0) return;
    draggingRef.current = true;
    e.currentTarget.setPointerCapture(e.pointerId);
    onSeekDisplay(timeAt(e.clientX));
  };
  const onPointerMove = (e) => { if (draggingRef.current) onSeekDisplay(timeAt(e.clientX)); };
  const onPointerUp = (e) => {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    onSeekCommit(timeAt(e.clientX));
  };
  const onTimelineKey = (e) => {
    const step = e.shiftKey ? 1 : NUDGE;
    if (e.key === "ArrowRight") { e.preventDefault(); onSeekCommit(currentTime + step); }
    if (e.key === "ArrowLeft") { e.preventDefault(); onSeekCommit(currentTime - step); }
    if (e.key === "Home") { e.preventDefault(); onSeekCommit(0); }
    if (e.key === "End") { e.preventDefault(); onSeekCommit(duration); }
  };

  const pct = duration > 0 ? (currentTime / duration) * 100 : 0;
  const step = rulerStep(duration, width);
  const marks = duration > 0 ? Array.from({ length: Math.floor(duration / step) + 1 }, (_, i) => i * step) : [];
  const now = clock(currentTime, true);
  const sampleRate = 44.1;

  return (
    <section className="studio" aria-label="Stem mixer">
      {/* ── toolbar ── */}
      <div className="studio__toolbar">
        <div className="tb-song">
          <span className="tb-song__label">Project</span>
          <span className="tb-song__name" translate="no" aria-label={fileName}>{title}</span>
        </div>

        <div className="tb-transport" role="group" aria-label="Transport">
          <button type="button" className="tb-btn" aria-label="Go to Start" title="Go to start (↵)" onClick={onToStart}>
            <Icon><path d="M6 5v14M18 6 9 12l9 6V6Z" /></Icon>
          </button>
          <button type="button" className="tb-btn" aria-label={`Back ${NUDGE} Seconds`} onClick={() => onSeekCommit(currentTime - NUDGE)}>
            <Icon><path d="M11 6 4 12l7 6V6ZM20 6l-7 6 7 6V6Z" /></Icon>
          </button>
          <button type="button" className={`tb-btn tb-play${isPlaying ? " is-on" : ""}`} aria-label={isPlaying ? "Pause" : "Play"} title="Play / pause (Space)" onClick={onPlayPause} disabled={!duration}>
            {isPlaying
              ? <Icon size={20}><path d="M8 5v14M16 5v14" strokeWidth="3" /></Icon>
              : <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><path d="M7 4.5v15l13-7.5-13-7.5Z" fill="currentColor" /></svg>}
          </button>
          <button type="button" className="tb-btn" aria-label={`Forward ${NUDGE} Seconds`} onClick={() => onSeekCommit(currentTime + NUDGE)}>
            <Icon><path d="m13 6 7 6-7 6V6ZM4 6l7 6-7 6V6Z" /></Icon>
          </button>
          <button type="button" className={`tb-btn tb-loop${loop ? " is-on" : ""}`} aria-pressed={loop} aria-label="Loop" title="Loop (L)" onClick={onLoop}>
            <Icon><path d="M17 2l3 3-3 3M4 11V9a4 4 0 0 1 4-4h12M7 22l-3-3 3-3M20 13v2a4 4 0 0 1-4 4H4" /></Icon>
          </button>
        </div>

        <div className="lcd" role="group" aria-label="Song position">
          <div className="lcd__cell lcd__cell--time">
            <span className="lcd__big">{now.base}<small>.{now.tenth}</small></span>
            <span className="lcd__label">Position</span>
          </div>
          <div className="lcd__cell">
            <span className="lcd__mid">{clock(duration)}</span>
            <span className="lcd__label">Length</span>
          </div>
          <div className="lcd__cell">
            <span className="lcd__mid">{stems.length}</span>
            <span className="lcd__label">Stems</span>
          </div>
          <div className="lcd__cell">
            <span className="lcd__mid">{sampleRate}</span>
            <span className="lcd__label">kHz</span>
          </div>
        </div>

        <div className="tb-right">
          <label className="tb-master" title="Master volume">
            <Icon><path d="M4 9v6h4l5 4V5L8 9H4ZM16.5 8.5a5 5 0 0 1 0 7M19 6a8.5 8.5 0 0 1 0 12" /></Icon>
            <span className="sr-only">Master volume</span>
            <input
              type="range" min={0} max={1.2} step={0.01} value={masterVolume}
              onChange={(e) => onMaster(parseFloat(e.target.value))}
              style={{ "--fill": `${(masterVolume / 1.2) * 100}%` }}
              aria-valuetext={`${Math.round(masterVolume * 100)}%`}
            />
          </label>
          <button type="button" className="tb-export" onClick={onDownloadAll}>
            <Icon size={15}><path d="M12 4v11M7 10l5 5 5-5M5 20h14" /></Icon>
            Export All
          </button>
        </div>
      </div>

      {/* ── tracks ── */}
      <div className="studio__body" style={{ "--grid-step": `${duration ? (step / duration) * 100 : 10}%` }}>
        <div className="studio__heads">
          <div className="studio__corner">
            <span>Tracks</span>
            <span className="tabular">{stems.length - mutedIds.size}/{stems.length} live</span>
          </div>
          {stems.map((stem) => (
            <TrackHead
              key={stem.id}
              stem={stem}
              muted={mutedIds.has(stem.id)}
              solo={soloId === stem.id}
              playingAlone={isPlaying && soloId === stem.id}
              volume={volumes[stem.id] ?? 1}
              meterElsRef={meterElsRef}
              onPlayStem={onPlayStem}
              onMute={onMute}
              onSolo={onSolo}
              onVolume={onVolume}
              onDownload={onDownload}
            />
          ))}
        </div>

        <div
          ref={timelineRef}
          className="studio__timeline"
          role="slider"
          tabIndex={0}
          aria-label="Playhead"
          aria-valuemin={0}
          aria-valuemax={Math.round(duration)}
          aria-valuenow={Math.round(currentTime)}
          aria-valuetext={`${clock(currentTime)} of ${clock(duration)}`}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onKeyDown={onTimelineKey}
        >
          <div className="ruler" aria-hidden="true">
            {marks.map((t) => (
              <span key={t} className="ruler__mark" style={{ left: `${(t / duration) * 100}%` }}>
                <span className="ruler__label">{clock(t)}</span>
              </span>
            ))}
          </div>
          {stems.map((stem) => (
            <Lane key={stem.id} stem={stem} peaks={peaks[stem.id]} muted={mutedIds.has(stem.id)} />
          ))}
          <div className="playhead" style={{ left: `${pct}%` }} aria-hidden="true"><span className="playhead__cap" /></div>
        </div>
      </div>

      {/* ── key legend ── */}
      <div className="studio__keys" aria-label="Keyboard shortcuts">
        <span><kbd>Space</kbd> play / pause</span>
        <span><kbd>↵</kbd> to start</span>
        <span><kbd>L</kbd> loop</span>
        <span><kbd>←</kbd><kbd>→</kbd> nudge (timeline focused)</span>
        <span>
          {stems.map((s) => <kbd key={s.id} style={{ color: s.color }}>{s.shortcut}</kbd>)} mute · <kbd>Shift</kbd>+key solo
        </span>
      </div>
    </section>
  );
}
