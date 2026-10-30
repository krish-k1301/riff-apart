import { useState, useEffect, useRef, useCallback } from "react";
import JSZip from "jszip";
import DitherWaves from "./components/DitherWaves.jsx";
import ElectricLogo from "./components/ElectricLogo/ElectricLogo.jsx";
import Intro, { introAlreadySeen } from "./components/Intro.jsx";
import useWordmarkSrc from "./components/useWordmarkSrc.js";
import SeparationLoader from "./components/SeparationLoader.jsx";
import Studio from "./components/Studio.jsx";
import StemSplit from "./components/StemSplit.jsx";

// ─── STEM DEFINITIONS ──────────────────────────────────────
const ALL_STEMS = [
  { id: "vocals", name: "Vocals",  color: "#F4D58D", shortcut: "V" },
  { id: "drums",  name: "Drums",   color: "#E5484D", shortcut: "D" },
  { id: "bass",   name: "Bass",    color: "#4D96FF", shortcut: "B" },
  { id: "guitar", name: "Guitar",  color: "#46D39A", shortcut: "G" },
  { id: "piano",  name: "Piano",   color: "#C77DFF", shortcut: "P" },
  { id: "other",  name: "Other",   color: "#8D99AE", shortcut: "O" },
];

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
  "Freddie says: “I want to break free” (from this buffer)…",
  "Is this the real file? Is this just fantasy?",
  "We will, we will… load you.",
  "Another byte bites the dust…",
  "Don’t stop me now — oh wait, buffering.",
];

const AUDIO_EXT = /\.(wav|mp3|flac|ogg|m4a)$/i;
const isAudioFile = (f) => !!f && (f.type.startsWith("audio/") || AUDIO_EXT.test(f.name));


const TICKER = [
  ...ALL_STEMS.map((s) => ({ text: s.name, color: s.color })),
  { text: "Full songs" }, { text: "No login" }, { text: "320 kbps MP3" }, { text: "Mute & solo" },
];

const STEPS = [
  { n: "01", title: "Drop a Track", body: "Any wav, mp3, flac, ogg or m4a. Full length, no time limit." },
  { n: "02", title: "Watch It Split", body: "Meta’s HTDemucs model pulls out six stems. You see the real progress and an ETA." },
  { n: "03", title: "Mix & Download", body: "Solo the bassline, mute the vocals, grab any stem as a 320 kbps MP3." },
];

// ─── HELPERS ───────────────────────────────────────────────
const API = import.meta.env.VITE_API_URL || "";

// Overall progress is split across the steps the user waits on
const STAGE = { upload: [0, 6], loading: [6, 8], separating: [8, 92], download: [92, 98], unzip: [98, 100] };
const stagePct = (stage, frac) => {
  const [a, b] = STAGE[stage];
  return a + (b - a) * Math.min(1, Math.max(0, frac));
};
const STATUS_LABEL = {
  uploading: "uploading", queued: "queued", loading: "loading model",
  separating: "separating", packaging: "packaging", downloading: "downloading", unzipping: "unpacking",
};

// fetch() can't report upload progress, XHR can
function uploadWithProgress(url, form, onProgress, signal) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => resolve({ status: xhr.status, body: xhr.response });
    xhr.onerror = () => reject(new Error("Couldn’t reach the server. Check your connection and try again."));
    xhr.onabort = () => reject(new DOMException("Aborted", "AbortError"));
    signal.addEventListener("abort", () => xhr.abort(), { once: true });
    xhr.send(form);
  });
}

const sleep = (ms, signal) => new Promise((resolve, reject) => {
  const t = setTimeout(resolve, ms);
  signal.addEventListener("abort", () => { clearTimeout(t); reject(new DOMException("Aborted", "AbortError")); }, { once: true });
});

// Stream a response body, reporting 0→1 as it arrives
async function readWithProgress(res, onProgress) {
  const total = parseInt(res.headers.get("content-length") || "0");
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total > 0) onProgress(received / total);
  }
  return new Blob(chunks);
}

// Min/max pairs per bucket, for drawing a region's waveform without touching the audio again
function computePeaks(buffer, buckets = 2400) {
  const channels = Array.from({ length: buffer.numberOfChannels }, (_, c) => buffer.getChannelData(c));
  const len = buffer.length;
  const size = Math.max(1, Math.floor(len / buckets));
  const out = new Float32Array(buckets * 2);
  for (let b = 0; b < buckets; b++) {
    let min = 0, max = 0;
    const start = b * size;
    const end = Math.min(len, start + size);
    for (let i = start; i < end; i += 4) { // every 4th sample is plenty for a picture
      let v = 0;
      for (const ch of channels) v += ch[i];
      v /= channels.length;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    out[b * 2] = min;
    out[b * 2 + 1] = max;
  }
  return out;
}

// ─── DROP ZONE ─────────────────────────────────────────────
function DropZone({ onFileSelect, isDragging, setIsDragging }) {
  const fileRef = useRef(null);
  const dragDepth = useRef(0); // dragenter/leave fire for every child, so count them
  const [rejected, setRejected] = useState(null);

  const accept = useCallback((file) => {
    if (!file) return;
    if (!isAudioFile(file)) {
      setRejected(file.name);
      return;
    }
    setRejected(null);
    onFileSelect(file);
  }, [onFileSelect]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    dragDepth.current = 0;
    setIsDragging(false);
    accept(e.dataTransfer.files[0]);
  }, [accept, setIsDragging]);

  const openPicker = () => fileRef.current?.click();

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        aria-label="Choose a track to separate, or drop an audio file here"
        aria-describedby={rejected ? "drop-error" : "drop-hint"}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openPicker(); }
        }}
        onDragEnter={(e) => { e.preventDefault(); dragDepth.current += 1; setIsDragging(true); }}
        onDragOver={(e) => e.preventDefault()}
        onDragLeave={() => { dragDepth.current = Math.max(0, dragDepth.current - 1); if (!dragDepth.current) setIsDragging(false); }}
        onDrop={handleDrop}
        onClick={openPicker}
        className={`dropzone${isDragging ? " is-dragging" : ""}${rejected ? " is-rejected" : ""}`}
      >
        <svg className="dropzone__outline" aria-hidden="true"><rect x="1" y="1" rx="15" ry="15" /></svg>
        <span className="dropzone__icon" aria-hidden="true">
          {[0.45, 0.8, 1, 0.65, 0.9, 0.5, 0.7].map((v, i) => <i key={i} style={{ height: `${v * 100}%`, animationDelay: `${i * 0.09}s` }} />)}
        </span>
        <input
          ref={fileRef} type="file" accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a"
          tabIndex={-1}
          style={{ display: "none" }}
          onChange={(e) => { accept(e.target.files[0]); e.target.value = ""; }}
        />
        <div className="dropzone__text">
          <div className="dropzone__title">
            {isDragging ? "Yeah, right there. Drop it." : "Throw a track in here"}
          </div>
          <div id="drop-hint" className="dropzone__hint">
            wav · mp3 · flac · ogg · m4a, whatever you’ve got
          </div>
        </div>
        <span className="dropzone__btn" aria-hidden="true">Browse Files</span>
      </div>
      <div
        id="drop-error"
        role="alert"
        style={{ minHeight: "18px", marginTop: "8px", fontFamily: "var(--font-mono)", fontSize: "11px", color: "#E5484D" }}
      >
        {rejected && <>“{rejected}” isn’t an audio file. Try a wav, mp3, flac, ogg or m4a.</>}
      </div>
    </div>
  );
}

// ─── QUALITY TOGGLE ────────────────────────────────────────
const QUALITIES = [
  { id: "high", label: "High", hint: "two passes, cleaner guitar" },
  { id: "fast", label: "Fast", hint: "one pass, about twice as quick" },
];

function readQuality() {
  try { return localStorage.getItem("riffapart:quality") === "high" ? "high" : "fast"; } catch { return "fast"; }
}

function QualityToggle({ value, onChange }) {
  const pick = (id) => {
    onChange(id);
    try { localStorage.setItem("riffapart:quality", id); } catch { /* private mode */ }
  };
  return (
    <div className="quality">
      <div className="quality__seg" role="radiogroup" aria-label="Separation quality">
        {QUALITIES.map((q) => (
          <button
            key={q.id} type="button" role="radio" aria-checked={value === q.id}
            className="quality__opt" onClick={() => pick(q.id)}
          >{q.label}</button>
        ))}
      </div>
      <span className="quality__hint">{QUALITIES.find((q) => q.id === value).hint}</span>
    </div>
  );
}

// ─── MAIN APP ──────────────────────────────────────────────
export default function App() {
  const [state, setState] = useState("idle");     // idle | processing | done | error
  const [phase, setPhase] = useState("uploading");  // job status shown by the loader
  const [eta, setEta] = useState(null);              // { seconds, at } from the server
  const [loaderOpen, setLoaderOpen] = useState(false); // stays up through the reveal
  const [quality, setQuality] = useState(readQuality); // high | fast
  const [quipIdx, setQuipIdx] = useState(0);
  const [introDone, setIntroDone] = useState(introAlreadySeen);
  const [file, setFile] = useState(null);
  const [activeStems, setActiveStems] = useState(ALL_STEMS); // stems shown in done state
  const [progress, setProgress] = useState(0);
  const [stemUrls, setStemUrls] = useState({});
  const [peaks, setPeaks] = useState({});           // { [stemId]: Float32Array } waveform min/max pairs
  const [volumes, setVolumes] = useState(() => Object.fromEntries(ALL_STEMS.map(st => [st.id, 1])));
  const [masterVolume, setMasterVolume] = useState(0.9);
  const [loop, setLoop] = useState(false);
  const [mutedIds, setMutedIds] = useState(new Set());
  const [soloId, setSoloId] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [easterEggClicks, setEasterEggClicks] = useState(0);
  const [showEaster, setShowEaster] = useState(false);
  const [error, setError] = useState(null);

  // Web Audio — every stem plays from one shared clock; mute/solo/volume are gains
  const audioCtxRef = useRef(null);         // single shared AudioContext
  const audioBuffersRef = useRef({});       // { [stemId]: AudioBuffer } decoded once on load
  const gainNodesRef = useRef({});          // { [stemId]: GainNode } mute/solo/volume
  const analysersRef = useRef({});          // { [stemId]: AnalyserNode } for the level meters
  const masterGainRef = useRef(null);
  const sourceNodesRef = useRef({});        // { [stemId]: AudioBufferSourceNode } fresh each play
  const audioOffsetRef = useRef(0);         // song position (s) at last play/pause/seek event
  const ctxTimeAtStartRef = useRef(0);      // audioCtx.currentTime value when last playback began
  const isPlayingRef = useRef(false);       // mirror of isPlaying state for use inside rAF
  const meterElsRef = useRef({});           // { [stemId]: HTMLElement } meters, written to directly
  const volumesRef = useRef({});
  const mutedRef = useRef(new Set());
  const loopRef = useRef(false);

  const abortRef = useRef(null);
  const rafRef = useRef(null);
  const jobIdRef = useRef(null);
  const wordmarkSrc = useWordmarkSrc();

  volumesRef.current = volumes;
  mutedRef.current = mutedIds;
  loopRef.current = loop;

  const gainFor = (id) => (mutedRef.current.has(id) ? 0 : volumesRef.current[id] ?? 1);
  const applyGains = (muted = mutedRef.current, vols = volumesRef.current) => {
    Object.entries(gainNodesRef.current).forEach(([id, g]) => {
      g.gain.value = muted.has(id) ? 0 : vols[id] ?? 1;
    });
  };

  // ── Decode stems into AudioBuffers whenever stemUrls are set ──
  useEffect(() => {
    if (Object.keys(stemUrls).length === 0) return;

    const ctx = new AudioContext();
    audioCtxRef.current = ctx;
    audioBuffersRef.current = {};
    gainNodesRef.current = {};
    analysersRef.current = {};
    sourceNodesRef.current = {};
    audioOffsetRef.current = 0;
    ctxTimeAtStartRef.current = 0;

    const master = ctx.createGain();
    master.gain.value = masterVolume;
    master.connect(ctx.destination);
    masterGainRef.current = master;

    // stem → gain → analyser → master
    activeStems.forEach(stem => {
      const gain = ctx.createGain();
      gain.gain.value = gainFor(stem.id);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      gain.connect(analyser);
      analyser.connect(master);
      gainNodesRef.current[stem.id] = gain;
      analysersRef.current[stem.id] = analyser;
    });

    // Decode each blob, then reduce it to min/max peaks for the region waveforms
    activeStems.forEach(stem => {
      const url = stemUrls[stem.id];
      if (!url) return;
      fetch(url)
        .then(r => r.arrayBuffer())
        .then(buf => ctx.decodeAudioData(buf))
        .then(audioBuffer => {
          audioBuffersRef.current[stem.id] = audioBuffer;
          setDuration(prev => prev || audioBuffer.duration);
          setPeaks(prev => ({ ...prev, [stem.id]: computePeaks(audioBuffer) }));
        })
        .catch(() => {});
    });

    return () => {
      stopAllSources();
      ctx.close();
    };
  }, [stemUrls]);

  // ── rAF loop: playhead from the audio clock + level meters ──
  useEffect(() => {
    const meterBuf = new Float32Array(512);
    const clearMeters = () => Object.values(meterElsRef.current).forEach(el => el?.style.setProperty("--level", "0"));
    if (!isPlaying) { clearMeters(); return; }
    const tick = () => {
      const ctx = audioCtxRef.current;
      if (isPlayingRef.current && ctx) {
        const elapsed = Math.max(0, ctx.currentTime - ctxTimeAtStartRef.current);
        let pos = audioOffsetRef.current + elapsed;
        if (loopRef.current && duration > 0) pos %= duration;
        setCurrentTime(pos);
        Object.entries(analysersRef.current).forEach(([id, analyser]) => {
          const el = meterElsRef.current[id];
          if (!el) return;
          analyser.getFloatTimeDomainData(meterBuf);
          let sum = 0;
          for (let i = 0; i < meterBuf.length; i++) sum += meterBuf[i] * meterBuf[i];
          const rms = Math.sqrt(sum / meterBuf.length);
          el.style.setProperty("--level", Math.min(1, rms * 3.2).toFixed(3));
        });
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => { cancelAnimationFrame(rafRef.current); clearMeters(); };
  }, [isPlaying, duration]);

  // ── Tab title follows the screen ──────────────────────────
  useEffect(() => {
    const titles = {
      idle: "riffapart",
      processing: `Separating ${file?.name ?? "track"}… · riffapart`,
      done: `${activeStems.length} stems ready · riffapart`,
      error: "Couldn’t separate · riffapart",
    };
    document.title = titles[state];
  }, [state, file, activeStems.length]);

  // ── Rotate quips while the loader is up ───────────────────
  useEffect(() => {
    if (state !== "processing") return;
    const t = setInterval(() => setQuipIdx((i) => (i + 1) % LOADING_QUIPS.length), 3000);
    return () => clearInterval(t);
  }, [state]);

  // ── Transport ───────────────────────────────────────────
  // Stopping a source on purpose must not trigger its "ended" cleanup
  const stopAllSources = () => {
    Object.values(sourceNodesRef.current).forEach(node => {
      if (!node) return;
      node.onended = null;
      try { node.stop(0); } catch {}
    });
    sourceNodesRef.current = {};
  };

  const currentPosition = () => {
    const ctx = audioCtxRef.current;
    if (!ctx || !isPlayingRef.current) return audioOffsetRef.current;
    const pos = audioOffsetRef.current + Math.max(0, ctx.currentTime - ctxTimeAtStartRef.current);
    return loopRef.current && duration > 0 ? pos % duration : pos;
  };

  // Schedule every stem to start on the same audio-clock tick → sample-accurate sync
  const startAllSources = (offset) => {
    const ctx = audioCtxRef.current;
    const AHEAD = 0.05; // 50 ms gives the browser time to prepare all sources
    const startAt = ctx.currentTime + AHEAD;
    ctxTimeAtStartRef.current = startAt;
    audioOffsetRef.current = offset;
    let started = 0;
    activeStems.forEach(stem => {
      const buffer = audioBuffersRef.current[stem.id];
      const gain = gainNodesRef.current[stem.id];
      if (!buffer || !gain || offset >= buffer.duration) return;
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.loop = loopRef.current;
      source.connect(gain);
      source.onended = () => {
        if (sourceNodesRef.current[stem.id] !== source) return;
        delete sourceNodesRef.current[stem.id];
        if (Object.keys(sourceNodesRef.current).length === 0) {
          // reached the end: park the playhead back at the start
          isPlayingRef.current = false;
          setIsPlaying(false);
          audioOffsetRef.current = 0;
          setCurrentTime(0);
        }
      };
      source.start(startAt, offset);
      sourceNodesRef.current[stem.id] = source;
      started++;
    });
    return started > 0;
  };

  const handlePlayPauseAll = useCallback(async () => {
    const ctx = audioCtxRef.current;
    if (!ctx) return;
    if (isPlayingRef.current) {
      audioOffsetRef.current = currentPosition();
      stopAllSources();
      isPlayingRef.current = false;
      setIsPlaying(false);
      setCurrentTime(audioOffsetRef.current);
    } else {
      await ctx.resume(); // browsers suspend the AudioContext until a user gesture
      const from = audioOffsetRef.current >= duration - 0.05 ? 0 : audioOffsetRef.current;
      if (startAllSources(from)) {
        isPlayingRef.current = true;
        setIsPlaying(true);
      }
    }
  }, [activeStems, duration]);

  // Called on every drag tick: only move the playhead, don't touch audio
  const handleSeekDisplay = useCallback((time) => {
    setCurrentTime(time);
  }, []);

  // Called on release: jump the audio to the new position
  const handleSeekCommit = useCallback(async (time) => {
    const t = Math.max(0, Math.min(time, Math.max(0, duration - 0.01)));
    setCurrentTime(t);
    audioOffsetRef.current = t;
    const ctx = audioCtxRef.current;
    if (!ctx || !isPlayingRef.current) return;
    stopAllSources();
    await ctx.resume();
    startAllSources(t);
  }, [activeStems, duration]);

  const handleToStart = useCallback(() => handleSeekCommit(0), [handleSeekCommit]);

  const toggleLoop = useCallback(() => {
    setLoop(prev => {
      const next = !prev;
      loopRef.current = next;
      Object.values(sourceNodesRef.current).forEach(node => { if (node) node.loop = next; });
      return next;
    });
  }, []);

  // ── Mixer: mute / solo / volume ─────────────────────────
  const toggleMute = useCallback((id) => {
    setSoloId(null);
    setMutedIds(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      applyGains(next);
      return next;
    });
  }, []);

  const toggleSolo = useCallback((id) => {
    setSoloId(prev => {
      const next = prev === id ? null : id;
      const muted = next ? new Set(activeStems.filter(s => s.id !== next).map(s => s.id)) : new Set();
      setMutedIds(muted);
      applyGains(muted);
      return next;
    });
  }, [activeStems]);

  // Solo a stem and start playback; pressing it again while it plays alone pauses
  const playStem = useCallback((id) => {
    if (isPlayingRef.current && soloId === id) { handlePlayPauseAll(); return; }
    const muted = new Set(activeStems.filter(s => s.id !== id).map(s => s.id));
    setSoloId(id);
    setMutedIds(muted);
    applyGains(muted);
    if (!isPlayingRef.current) handlePlayPauseAll();
  }, [soloId, activeStems, handlePlayPauseAll]);

  const setStemVolume = useCallback((id, value) => {
    setVolumes(prev => {
      const next = { ...prev, [id]: value };
      volumesRef.current = next;
      const g = gainNodesRef.current[id];
      if (g && !mutedRef.current.has(id)) g.gain.value = value;
      return next;
    });
  }, []);

  const changeMasterVolume = useCallback((value) => {
    setMasterVolume(value);
    if (masterGainRef.current) masterGainRef.current.gain.value = value;
  }, []);

  // ── Keyboard shortcuts ──────────────────────────────────
  useEffect(() => {
    const handler = (e) => {
      if (state !== "done" || loaderOpen) return;
      if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return; // leave browser shortcuts alone
      const tag = e.target?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      if ((e.key === " " || e.key === "Enter") && tag === "BUTTON") return; // buttons handle their own keys
      if (e.key === " ") { e.preventDefault(); handlePlayPauseAll(); return; }
      if (e.key === "Enter") { e.preventDefault(); handleToStart(); return; }
      if (e.key.toUpperCase() === "L") { toggleLoop(); return; }
      const stem = activeStems.find((s) => s.shortcut === e.key.toUpperCase());
      if (stem) (e.shiftKey ? toggleSolo : toggleMute)(stem.id);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [state, loaderOpen, activeStems, handlePlayPauseAll, handleToStart, toggleLoop, toggleMute, toggleSolo]);

  // ── Main file processing ────────────────────────────────
  const handleFileSelect = async (f) => {
    Object.values(stemUrls).forEach(url => URL.revokeObjectURL(url));
    if (audioCtxRef.current) {
      stopAllSources();
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
    setLoaderOpen(true);
    setPhase("uploading");
    setEta(null);
    setProgress(0);
    setStemUrls({});
    setError(null);
    setPeaks({});
    setIsPlaying(false);
    setMutedIds(new Set());
    setSoloId(null);
    setCurrentTime(0);
    setDuration(0);
    setActiveStems(ALL_STEMS);

    const controller = new AbortController();
    abortRef.current = controller;
    const { signal } = controller;

    try {
      // 1. Upload and start a separation job
      const form = new FormData();
      form.append("file", f);
      const up = await uploadWithProgress(`${API}/api/jobs?quality=${quality}`, form, (frac) => setProgress(stagePct("upload", frac)), signal);
      const zipBlob = await runJob(up, signal);
      if (!zipBlob) return; // cancelled

      // 4. Unzip
      setPhase("unzipping");
      const zip = await JSZip.loadAsync(zipBlob);
      const urls = {};
      let done = 0;
      for (const stem of ALL_STEMS) {
        const entry = zip.file(`${stem.id}.mp3`);
        if (entry) {
          const mp3 = new Blob([await entry.async("arraybuffer")], { type: "audio/mpeg" });
          urls[stem.id] = URL.createObjectURL(mp3);
        }
        done++;
        setProgress(stagePct("unzip", done / ALL_STEMS.length));
      }
      setEta({ seconds: 0, at: performance.now() });
      setStemUrls(urls);
      setState("done"); // the loader sees this and opens onto the mixer

    } catch (err) {
      if (err.name === "AbortError") return;
      setLoaderOpen(false);
      setState("error");
      setError(err.message);
    }
  };

  // 2–3. Poll a separation job for progress + ETA, then download its stems.
  // Resolves to null if the job was cancelled.
  const runJob = async (up, signal) => {
    if (up.status >= 400) throw new Error(up.body?.detail || "Upload failed. Try the track again.");
    const jobId = up.body.id;
    jobIdRef.current = jobId;

    while (true) {
      const res = await fetch(`${API}/api/jobs/${jobId}`, { signal });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || "Lost track of the separation. Try again.");
      const job = await res.json();
      setPhase(job.status);
      if (job.eta_seconds != null) setEta({ seconds: job.eta_seconds + 2, at: performance.now() }); // +2 s for download
      if (job.status === "loading" || job.status === "queued") setProgress(stagePct("loading", 0.5));
      if (job.status === "separating") setProgress(stagePct("separating", job.progress));
      if (job.status === "packaging") setProgress(stagePct("separating", 1));
      if (job.status === "error") throw new Error(job.error || "Separation failed. Try a different file.");
      if (job.status === "cancelled") return null;
      if (job.status === "done") break;
      await sleep(600, signal);
    }

    setPhase("downloading");
    const res = await fetch(`${API}/api/jobs/${jobId}/result`, { signal });
    if (!res.ok) throw new Error("Couldn’t download the stems. Try again.");
    const zipBlob = await readWithProgress(res, (frac) => setProgress(stagePct("download", frac)));
    jobIdRef.current = null;
    return zipBlob;
  };

  // ── Reset ───────────────────────────────────────────────
  const handleReset = () => {
    abortRef.current?.abort();
    if (jobIdRef.current) {
      fetch(`${API}/api/jobs/${jobIdRef.current}`, { method: "DELETE" }).catch(() => {});
      jobIdRef.current = null;
    }
    setLoaderOpen(false);
    Object.values(stemUrls).forEach(url => URL.revokeObjectURL(url));
    if (audioCtxRef.current) {
      stopAllSources();
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
    setPeaks({});
    setIsPlaying(false);
    setMutedIds(new Set());
    setSoloId(null);
    setCurrentTime(0);
    setDuration(0);
    setError(null);
    setActiveStems(ALL_STEMS);
  };

  // ── Downloads ────────────────────────────────────────────
  const downloadStem = useCallback((stem) => {
    const url = stemUrls[stem.id];
    if (!url) return;
    const a = document.createElement("a");
    a.href = url;
    a.download = `${stem.id}_separated.mp3`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }, [stemUrls]);

  const downloadAll = () => {
    activeStems.forEach((stem, i) => {
      const url = stemUrls[stem.id];
      if (!url) return;
      setTimeout(() => {
        const a = document.createElement("a");
        a.href = url;
        a.download = `${stem.id}_separated.mp3`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      }, i * 200);
    });
  };

  // ── Intro finished ──────────────────────────────────────
  const onIntroDone = useCallback(() => setIntroDone(true), []);

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

      <div className="bg-layer" aria-hidden="true">
        <DitherWaves paused={!introDone || state !== "idle"} />
      </div>
      <div className="bg-scrim" aria-hidden="true" />

      {!introDone && (
        <Intro onDone={onIntroDone} />
      )}

      {loaderOpen && (
        <SeparationLoader
          progress={progress}
          eta={eta}
          status={STATUS_LABEL[phase] ?? phase}
          fileName={file?.name || ""}
          quip={progress >= STAGE.download[0]
            ? BUFFERING_QUIPS[Math.floor(progress) % BUFFERING_QUIPS.length]
            : LOADING_QUIPS[quipIdx]}
          complete={state === "done"}
          onRevealed={() => setLoaderOpen(false)}
          onCancel={handleReset}
        />
      )}

      <div className="content-layer" style={{ minHeight: "100vh", padding: "20px", maxWidth: state === "done" ? "1480px" : "1160px", margin: "0 auto" }}>

        {showEaster && (
          <div style={{
            position: "fixed", top: "50%", left: "50%",
            transform: "translate(-50%, -50%)", zIndex: 100,
            background: "var(--surface-1)", border: "1px solid var(--accent)",
            borderRadius: "16px", padding: "32px 40px", textAlign: "center",
            animation: "easterPop 0.4s ease, easterFade 4s ease forwards",
            boxShadow: "0 0 60px #FF9F1C20",
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
          <button type="button" onClick={handleLogoClick} className="logo-btn" aria-label="riffapart, start over" translate="no">
            <span aria-hidden="true" style={{ fontFamily: "var(--font-head)", fontSize: "34px", fontWeight: "700", color: "var(--text-1)", letterSpacing: "0.01em" }}>riff</span>
            <span aria-hidden="true" style={{ fontFamily: "var(--font-head)", fontSize: "34px", fontWeight: "700", color: "var(--accent)", letterSpacing: "0.01em" }}>apart</span>
            {introDone && wordmarkSrc && (
              <span className="logo-electric" aria-hidden="true">
                <ElectricLogo
                  src={wordmarkSrc}
                  color="#FF9F1C"
                  glowColor="#FF5E1A"
                  fill={0}
                  scale={0.62}
                  intensity={0.45}
                  glow={0.22}
                  arcs={0.35}
                  thickness={1.2}
                  theme="dark"
                />
              </span>
            )}
          </button>
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            {state !== "idle" && (
              <button onClick={handleReset} style={{
                padding: "7px 14px", borderRadius: "7px", border: "1px solid var(--border)",
                background: "transparent", color: "var(--text-2)", fontSize: "12px",
                cursor: "pointer", fontFamily: "var(--font-mono)",
              }}>new track</button>
            )}
          </div>
        </header>

        <main style={{
          animation: introDone ? "fadeIn 0.5s ease 0.05s both" : "none", opacity: introDone ? 1 : 0,
          maxWidth: state === "idle" || state === "done" ? "none" : "700px", margin: "0 auto",
        }}>

          {state === "idle" && (
            <div className="landing">
              <section className="hero">
                <div className="hero__copy">
                  <h1 className="hero__title">
                    Pull a song apart,<br />
                    <span>hear what’s inside.</span>
                  </h1>
                  <p className="hero__lede">
                    Drop any track. The model isolates every instrument (vocals, drums, bass, guitar,
                    piano and the rest) so you can solo, mute and download each one.
                  </p>
                  <DropZone onFileSelect={handleFileSelect} isDragging={isDragging} setIsDragging={setIsDragging} />
                  <QualityToggle value={quality} onChange={setQuality} />
                  <dl className="hero__stats">
                    <div><dt>Stems</dt><dd>6</dd></div>
                    <div><dt>Length Limit</dt><dd>None</dd></div>
                    <div><dt>Output</dt><dd>MP3</dd></div>
                    <div><dt>Logins</dt><dd>0</dd></div>
                  </dl>
                </div>
                <div className="hero__visual">
                  <StemSplit stems={ALL_STEMS} />
                </div>
              </section>

              <div className="ticker" aria-hidden="true">
                <div className="ticker__track">
                  {[0, 1].map((copy) => TICKER.map((item) => (
                    <span key={`${copy}-${item.text}`} className="ticker__item">
                      <span style={item.color ? { color: item.color } : undefined} className={item.color ? "" : "ticker__ghost"}>{item.text}</span>
                      <span className="ticker__star">✦</span>
                    </span>
                  )))}
                </div>
              </div>

              <section className="steps" aria-labelledby="how-it-works">
                <h2 id="how-it-works" className="steps__title">How It Works</h2>
                <ol className="steps__list">
                  {STEPS.map((step) => (
                    <li key={step.n} className="step">
                      <span className="step__n" aria-hidden="true">{step.n}</span>
                      <h3>{step.title}</h3>
                      <p>{step.body}</p>
                    </li>
                  ))}
                </ol>
              </section>
            </div>
          )}

          {state === "error" && (
            <div role="alert" style={{ textAlign: "center", padding: "48px 0", animation: "fadeIn 0.4s ease" }}>
              <h2 style={{ fontFamily: "var(--font-head)", fontSize: "18px", fontWeight: "700", color: "var(--text-1)", marginBottom: "8px" }}>
                That track didn’t separate.
              </h2>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "12px", color: "#E5484D", marginBottom: "24px", maxWidth: "400px", margin: "0 auto 24px" }}>
                {error}
              </div>
              <button onClick={handleReset} style={{
                padding: "8px 20px", borderRadius: "8px", border: "1px solid var(--border)",
                background: "transparent", color: "var(--text-2)", fontSize: "12px",
                cursor: "pointer", fontFamily: "var(--font-mono)",
              }}>Try Another Track</button>
            </div>
          )}

          {state === "done" && (
            <Studio
              stems={activeStems}
              fileName={file?.name || "untitled"}
              revealed={!loaderOpen}
              peaks={peaks}
              duration={duration}
              currentTime={currentTime}
              isPlaying={isPlaying}
              loop={loop}
              mutedIds={mutedIds}
              soloId={soloId}
              volumes={volumes}
              masterVolume={masterVolume}
              meterElsRef={meterElsRef}
              onPlayPause={handlePlayPauseAll}
              onToStart={handleToStart}
              onLoop={toggleLoop}
              onSeekDisplay={handleSeekDisplay}
              onSeekCommit={handleSeekCommit}
              onPlayStem={playStem}
              onMute={toggleMute}
              onSolo={toggleSolo}
              onVolume={setStemVolume}
              onMaster={changeMasterVolume}
              onDownload={downloadStem}
              onDownloadAll={downloadAll}
            />
          )}
        </main>

        <footer style={{
          marginTop: "72px", paddingTop: "20px", borderTop: "1px solid var(--border)",
          fontFamily: "var(--font-mono)", textAlign: "center",
        }}>
          <div style={{ fontSize: "13px", color: "var(--text-1)", textShadow: "0 1px 3px #000, 0 0 10px #000" }}>
            built by a guitarist who got tired of bad stems
          </div>
          <div style={{ fontSize: "11px", color: "var(--text-1)", textShadow: "0 1px 3px #000, 0 0 10px #000", marginTop: "6px" }}>
            (amber font btw)
          </div>
        </footer>
      </div>
    </>
  );
}
