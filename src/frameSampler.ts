// Video frame sampler: <video> → canvas → JPEG at ~1–2 fps for the session
// socket's 0x02 lane (backend_overhaul.md §F3). Downscales to ~512 px on the
// longest edge, guards overlapping captures, and skips frames while the socket
// send buffer is backed up (slow uplink beats stale frames — §1C keep-latest).
//
// Timestamps: ONE monotone session clock for the whole session. The streaming
// VLM embeds each frame's ts as literal "N seconds" text and requires values
// to never decrease across the session's KV stream (live_control_overhaul.md
// §12), so the clock is segmented: ts = base + delta-in-segment, where a
// 'live' segment (camera/screen) advances by wall clock and a 'media' segment
// (video file) by the <video>'s currentTime. setClock() starts a new segment
// with base picked up from wherever the previous segment's clock stands — a
// source swap continues the timeline, never rewinds it — and a backward seek
// inside a media segment re-anchors instead of emitting a smaller ts.

export type SamplerClock = 'live' | 'media';

export interface FrameSamplerOptions {
  fps?: number;
  /** Static cap, or a per-capture getter — the source-aware profile lets a
   * mid-session camera → screen-share swap raise the cap without rebuilding
   * the sampler (screen text needs ~1280px to stay legible; camera 512). */
  maxEdge?: number | (() => number);
  quality?: number | (() => number);
  /** Live-mode effective capture rate (overrides the interval cadence via a
   *  pacing gate — lets a camera → screen-share swap slow to 1 fps without
   *  rebuilding the sampler). Media-clock mode keeps the constructor fps. */
  effFps?: () => number;
  /** Per-capture frame dedup (screen share): a tiny grayscale signature is
   *  compared with the last SENT frame; near-identical captures are dropped
   *  before encoding. Static screen content then costs ~zero vision tokens
   *  (a 1280px stream at 2fps once OOM'd the realtime KV cache within 5 min:
   *  531 frames × ~6× camera tokens). */
  dedup?: () => boolean;
  /** Skip capture when false (e.g. ws.bufferedAmount too high, camera off). */
  shouldSend?: () => boolean;
  /** Timestamp basis of the FIRST segment (default 'live'). A media segment
   *  additionally gates sampling on playback advancing a full interval, so a
   *  paused/ended video stops the frame flow by itself (board parity). */
  clock?: SamplerClock;
}

export interface FrameSampler {
  stop(): void;
  /** Begin a new timestamp segment (source swap): base continues from the
   *  previous segment's clock (never below any ts already emitted) and the
   *  anchors reset to "now". Camera↔screen swaps share one live segment and
   *  never need this — call it only when the clock BASIS changes. */
  setClock(clock: SamplerClock): void;
  /** Current session-clock value in ms. Use for `session_ts_start` on
   *  source-change events and other out-of-band stamps. */
  currentTs(): number;
  /** Send one out-of-band frame (e.g. a still image) stamped with the current
   *  session clock, through the sampler's own send path so the monotone
   *  bookkeeping sees it. */
  sendStill(jpeg: ArrayBuffer): void;
}

export function startFrameSampler(
  video: HTMLVideoElement,
  send: (jpeg: ArrayBuffer, captureTsMs: number) => void,
  { fps = 1, maxEdge = 512, quality = 0.75, shouldSend, clock = 'live' }: FrameSamplerOptions = {},
): FrameSampler {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const sigCanvas = document.createElement('canvas');
  const sigCtx = sigCanvas.getContext('2d', { willReadFrequently: true });
  let lastSig: Uint8ClampedArray | null = null;
  let lastLiveSampleWall = 0;
  let busy = false;
  let stopped = false;

  // --- segmented session clock ---
  let mode: SamplerClock = clock;
  let baseMs = 0;
  let anchorWall = performance.now();
  let anchorMedia = video.currentTime * 1000;
  let lastEmittedMs = 0; // largest ts ever handed to send() — the monotone floor
  let lastSampledMediaS = -Infinity; // media pacing gate (per segment)

  const currentTs = (): number => {
    if (mode === 'live') return baseMs + (performance.now() - anchorWall);
    let ts = baseMs + (video.currentTime * 1000 - anchorMedia);
    if (ts < lastEmittedMs) {
      // backward seek/replay inside a media segment: re-anchor at the floor
      // so the session clock never rewinds
      baseMs = lastEmittedMs;
      anchorMedia = video.currentTime * 1000;
      ts = baseMs;
    }
    return ts;
  };

  const setClock = (next: SamplerClock) => {
    baseMs = Math.max(lastEmittedMs, currentTs());
    anchorWall = performance.now();
    anchorMedia = video.currentTime * 1000;
    lastSampledMediaS = -Infinity;
    mode = next;
  };

  const capture = () => {
    if (stopped || busy || !ctx) return;
    if (video.readyState < 2 || video.videoWidth === 0) return; // no decodable frame yet
    if (shouldSend && !shouldSend()) return;
    if (mode === 'media') {
      // only sample when playback advanced a full interval; dt < 0 (seek-back)
      // falls through — currentTs() re-anchors and the frame sends
      const dt = video.currentTime - lastSampledMediaS;
      if (dt >= 0 && dt < 1 / Math.max(0.1, fps) - 0.02) return;
      lastSampledMediaS = video.currentTime;
    }

    const edge = typeof maxEdge === 'function' ? maxEdge() : maxEdge;
    const q = typeof quality === 'function' ? quality() : quality;
    // live-mode pacing gate: the interval ticks at the constructor cadence;
    // effFps (source-aware profile) slows the EFFECTIVE rate without a rebuild
    if (mode === 'live' && effFps) {
      const eff = Math.max(0.1, effFps());
      const dtWall = (performance.now() - lastLiveSampleWall) / 1000;
      if (lastLiveSampleWall > 0 && dtWall < 1 / eff - 0.02) return;
      lastLiveSampleWall = performance.now();
    }

    // dedup gate (screen share): tiny grayscale signature vs last SENT frame
    if (dedup && dedup()) {
      if (!sigCtx) return; // no 2d context (shouldn't happen)
      sigCanvas.width = 32; sigCanvas.height = 18;
      sigCtx.drawImage(video, 0, 0, 32, 18);
      const data = sigCtx.getImageData(0, 0, 32, 18).data;
      if (lastSig) {
        let diff = 0;
        for (let i = 0; i < data.length; i += 4) {
          diff += Math.abs(data[i] - lastSig[i]); // R channel ~ grayscale proxy
        }
        const meanDiff = diff / (data.length / 4);
        if (meanDiff < 2.5) return; // near-identical: skip encode AND send
      }
      lastSig = data;
    }

    const scale = Math.min(1, edge / Math.max(video.videoWidth, video.videoHeight));
    canvas.width = Math.max(2, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(2, Math.round(video.videoHeight * scale));
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    busy = true;
    const captureTsMs = currentTs();
    lastEmittedMs = Math.max(lastEmittedMs, captureTsMs);
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          busy = false;
          return;
        }
        blob
          .arrayBuffer()
          .then((jpeg) => {
            if (!stopped) send(jpeg, captureTsMs);
          })
          .finally(() => {
            busy = false;
          });
      },
      'image/jpeg',
      q,
    );
  };

  const timer = setInterval(capture, 1000 / Math.max(0.1, fps));
  capture();

  return {
    stop() {
      stopped = true;
      clearInterval(timer);
    },
    setClock,
    currentTs,
    sendStill(jpeg: ArrayBuffer) {
      if (stopped) return;
      const ts = currentTs();
      lastEmittedMs = Math.max(lastEmittedMs, ts);
      send(jpeg, ts);
    },
  };
}
