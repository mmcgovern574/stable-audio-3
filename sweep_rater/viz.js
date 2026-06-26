/* viz.js — audio-reactive "Swarm" visualizer for the rater.
 *
 * The original yuruyurau Swarm, unchanged in look: thousands of white particles
 * advected through an XOR bit-field flow, drawn on black with motion trails.
 * The audio MODULATES it (no recoloring, no splitting):
 *   • bass  → motion speed + brightness of the CORE (inner particles)
 *   • highs → brightness of the OUTER halo (edge particles)
 *   • mids  → flow speed + brightness of the mid radius
 *   • beat  → a burst of new particles
 * Idle/paused, it self-animates as the plain white swarm.
 *
 * AUDIO PATH: the <audio> element plays straight to the speakers (no Web Audio
 * in the audible path, so playback is never muted by a suspended AudioContext).
 * For the visuals we decode a copy of the clip once and read the spectrum at the
 * current playback position with a small Goertzel filter bank.
 *
 * createAudioViz(container, opts) returns a WaveSurfer-7-compatible shim:
 *   load(url) play() pause() isPlaying() getDuration() setTime(t)
 *   getMediaElement() on(event, fn)   events: ready timeupdate play pause error
 *   setKey() ensureAudible()  — no-ops kept for API compatibility
 */
(function (global) {
  "use strict";

  const C = Math.cos, S = Math.sin, PI = Math.PI, TAU = PI * 2, HYP = Math.hypot;

  // ── tunables ────────────────────────────────────────────────────────────
  const SPAWN      = 16;     // particles spawned per frame
  const CAP        = 5000;   // max live particles
  const FADE       = 0.06;   // trail fade per frame (lower = longer trails)
  const SLOW       = 0.5;    // global motion slowdown
  const BASE_ALPHA = 0.33;   // resting particle opacity
  const REACH      = 140;    // model radius used to normalize core↔edge
  const WIN        = 2048;   // analysis window (samples)

  const BASS_F = [55, 80, 110, 150, 200];
  const MID_F  = [300, 450, 700, 1100, 1600];
  const HIGH_F = [2600, 3800, 5600, 8000];

  function goertzelAt(buf, start, N, sr, f) {
    const w = 2 * PI * f / sr, coeff = 2 * Math.cos(w);
    let s1 = 0, s2 = 0;
    const end = Math.min(buf.length, start + N);
    for (let i = start; i < end; i++) { const s0 = buf[i] + coeff * s1 - s2; s2 = s1; s1 = s0; }
    const M = end - start || 1;
    return Math.sqrt(Math.max(0, s1 * s1 + s2 * s2 - coeff * s1 * s2)) / M;
  }

  function createAudioViz(container, opts) {
    opts = opts || {};
    const cssH = opts.height || 240;

    container.style.position = "relative";
    container.innerHTML = "";
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "width:100%;height:" + cssH + "px;display:block;border-radius:8px";
    container.appendChild(canvas);
    const g = canvas.getContext("2d");

    const dpr = Math.min(global.devicePixelRatio || 1, 1.5);
    let WD = 0, HD = 0, CX = 0, CY = 0, FS = 1;
    function resize() {
      const cssW = Math.max(120, container.clientWidth || 600);
      WD = Math.min(Math.round(cssW * dpr), 1600); HD = Math.round(cssH * dpr);
      canvas.width = WD; canvas.height = HD;
      CX = WD / 2; CY = HD / 2; FS = (HD / 540);
      g.fillStyle = "#070707"; g.fillRect(0, 0, WD, HD);
    }

    let zoom = 1.05, tt = 0;
    let bass = 0, mid = 0, high = 0, beat = 0, bassAvg = 0, lastBeat = -9;
    let peakB = 1e-4, peakM = 1e-4, peakH = 1e-4;
    let particles = [];
    function rand3D() { const u = Math.random() * 2 - 1, th = Math.random() * TAU, r = Math.sqrt(1 - u * u); return { x: r * C(th), y: r * S(th), z: u }; }

    // ── audio element (plays straight to speakers) ───────────────────────
    const audio = new Audio();
    audio.preload = "auto"; audio.style.display = "none";
    container.appendChild(audio);

    const listeners = {};
    function emit(ev, arg) { (listeners[ev] || []).forEach(fn => { try { fn(arg); } catch (_) {} }); }
    let readyEmitted = false;
    audio.addEventListener("canplay", () => { if (!readyEmitted) { readyEmitted = true; emit("ready"); } });
    audio.addEventListener("timeupdate", () => emit("timeupdate", audio.currentTime));
    audio.addEventListener("play", () => emit("play"));
    audio.addEventListener("pause", () => emit("pause"));
    audio.addEventListener("error", () => emit("error"));

    // ── decoded copy for analysis only (no audio output) ─────────────────
    let decCtx = null;
    function getDec() {
      if (!decCtx) {
        const OC = global.OfflineAudioContext || global.webkitOfflineAudioContext;
        try { decCtx = OC ? new OC(1, 1, 44100) : new (global.AudioContext || global.webkitAudioContext)(); }
        catch (_) { decCtx = null; }
      }
      return decCtx;
    }
    let aBuf = null, aSR = 44100, aToken = 0;
    function decodeFor(url) {
      const my = ++aToken; aBuf = null;
      if (!global.fetch) return;
      const dc = getDec(); if (!dc) return;
      fetch(url).then(r => r.arrayBuffer()).then(ab => dc.decodeAudioData(ab)).then(b => {
        if (my !== aToken) return;            // a newer clip loaded; ignore
        const n = b.length, c0 = b.getChannelData(0);
        if (b.numberOfChannels > 1) { const c1 = b.getChannelData(1); const m = new Float32Array(n); for (let i = 0; i < n; i++) m[i] = (c0[i] + c1[i]) * 0.5; aBuf = m; }
        else aBuf = c0;
        aSR = b.sampleRate;
      }).catch(() => {});
    }
    function bandAt(freqs, start) { let e = 0; for (let i = 0; i < freqs.length; i++) e += goertzelAt(aBuf, start, WIN, aSR, freqs[i]); return e / freqs.length; }

    // ── render ───────────────────────────────────────────────────────────
    let running = true;
    function frame() {
      if (running) {
        const playing = aBuf && !audio.paused;
        if (playing) {
          let start = Math.floor(audio.currentTime * aSR) - (WIN >> 1);
          if (start < 0) start = 0;
          if (start > aBuf.length - WIN) start = Math.max(0, aBuf.length - WIN);
          const be = bandAt(BASS_F, start), me = bandAt(MID_F, start), he = bandAt(HIGH_F, start);
          peakB = Math.max(be, peakB * 0.999); peakM = Math.max(me, peakM * 0.999); peakH = Math.max(he, peakH * 0.999);
          const rb = Math.min(1, be / (peakB + 1e-6)), rm = Math.min(1, me / (peakM + 1e-6)), rh = Math.min(1, he / (peakH + 1e-6));
          bass += (rb > bass ? 0.45 : 0.07) * (rb - bass);
          mid  += (rm > mid  ? 0.35 : 0.07) * (rm - mid);
          high += (rh > high ? 0.5  : 0.09) * (rh - high);
          bassAvg = bassAvg * 0.95 + rb * 0.05;
          const now = performance.now() / 1000;
          if (rb > bassAvg * 1.4 && rb > 0.3 && now - lastBeat > 0.16) { beat = 1; lastBeat = now; }
          beat *= 0.92;
        } else { bass *= 0.94; mid *= 0.94; high *= 0.94; beat *= 0.9; }
        const idle = !playing;

        // trails
        g.fillStyle = "rgba(5,5,7," + FADE + ")"; g.fillRect(0, 0, WD, HD);

        // spawn (steady stream + beat burst)
        const nsp = SPAWN + Math.round(beat * 36);
        for (let n = 0; n < nsp; n++) particles.push(rand3D());
        if (particles.length > CAP) particles = particles.slice(-CAP);

        // update + draw — original XOR flow, white particles, brightness modulated by band×radius
        tt += (1 + mid) * SLOW;
        const sp = (1 + bass * 1.2 + beat) * SLOW;
        const sz = Math.max(1, Math.round(HD / 200 * zoom));
        for (let j = 0; j < particles.length; j++) {
          const v = particles[j];
          const k = v.x + 5 + v.y, r = 9 * (((v.x * k) ^ (((v.y * k) + (tt / 90)) | 0)) & 7) - 0.1;
          v.x += S(r * v.y) / 119 * sp; v.y += C(v.x * r) / 119 * sp;
          const lx = v.x * 119, ly = v.y * 119;
          const px = (CX + lx * FS * zoom) | 0, py = (CY + ly * FS * zoom) | 0;
          if (px < 0 || px >= WD || py < 0 || py >= HD) continue;
          let al;
          if (idle) { al = BASE_ALPHA; }
          else {
            const rc = Math.min(1, HYP(lx, ly) / REACH);   // 0 = core, 1 = edge
            const innerW = 1 - rc, outerW = rc, midW = 1 - Math.abs(rc - 0.5) * 2;
            const react = bass * innerW + high * outerW + 0.7 * mid * midW;
            al = BASE_ALPHA * (0.45 + 1.5 * react);
            if (al > 0.85) al = 0.85; else if (al < 0.04) al = 0.04;
          }
          g.fillStyle = "rgba(250,250,252," + al.toFixed(3) + ")";
          g.fillRect(px, py, sz, sz);
        }
      }
      requestAnimationFrame(frame);
    }

    resize();
    if (global.ResizeObserver) { new ResizeObserver(() => resize()).observe(container); }
    else global.addEventListener("resize", resize);
    document.addEventListener("visibilitychange", () => { running = !document.hidden; });
    requestAnimationFrame(frame);

    // ── WaveSurfer-compatible API ────────────────────────────────────────
    return {
      load(url) { readyEmitted = false; audio.src = url; audio.load(); decodeFor(url); },
      setKey() { /* no-op */ },
      ensureAudible() { return false; /* no-op: audio plays natively */ },
      play() { return audio.play(); },
      pause() { audio.pause(); },
      isPlaying() { return !audio.paused && !audio.ended; },
      getDuration() { return isFinite(audio.duration) ? audio.duration : 0; },
      setTime(t) { if (isFinite(t)) try { audio.currentTime = t; } catch (_) {} },
      getMediaElement() { return audio; },
      getCurrentForm() { return "Swarm"; },
      on(ev, fn) { (listeners[ev] = listeners[ev] || []).push(fn); },
      destroy() { running = false; try { audio.pause(); audio.removeAttribute("src"); } catch (_) {} }
    };
  }

  global.createAudioViz = createAudioViz;
})(window);
