// Play mode of the browser app (docs/app.md): a processed video with its stage results drawn on top.
// The original video plays as it is; the boxes are drawn on a canvas over it from the saved stage outputs (JSON).

import { LAYERS } from "./layers.js";

const $ = (id) => document.getElementById(id);
const video = $("video");
const overlay = $("overlay");
const timeline = $("timeline");

const state = {
  name: null,
  details: null, // GET /api/videos/<name>: saved stages, settings used, video size / fps / frame count
  layers: {}, // layer name -> {data: saved stage output, frames: frame_idx -> frame}
  enabled: new Set(), // layers switched on
  fps: 30,
  frameCount: 0,
  width: 0, // frame size the results were made on
  height: 0,
  frame: 0, // frame on screen
  highlight: null, // box hovered in the side panel: {kind: "detections" | "rejected", index}
  requested: 0, // frame of the last seek asked for
  pendingSeek: null, // frame to seek to when the current seek is done (dragging on the timeline)
};

const api = (name, path = "") => `/api/videos/${encodeURIComponent(name)}${path}`;
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const clampFrame = (n) => Math.min(Math.max(n, 0), state.frameCount - 1);

async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `${response.status} ${response.statusText}: ${url}`);
  }
  return response.json();
}

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = false;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// ---------------------------------------------------------------------------------------------------------------
// Start: list of videos, or open the one in the URL (?video=IMG_4887&frame=120)

async function main() {
  const params = new URLSearchParams(location.search);
  const name = params.get("video");
  const videos = await getJSON("/api/videos");
  fillVideoSelect(videos, name);
  if (!name) {
    showHome(videos);
    return;
  }
  await openVideo(name, Number(params.get("frame")) || 0);
}

function fillVideoSelect(videos, current) {
  const select = $("video-select");
  const options = videos.map((v) => {
    const option = new Option(v.video_found ? v.name : `${v.name} (video not found)`, v.name, false, v.name === current);
    option.disabled = !v.video_found;
    return option;
  });
  select.replaceChildren(new Option("Choose a video…", ""), ...options);
  select.addEventListener("change", () => {
    location.search = select.value ? `?video=${encodeURIComponent(select.value)}` : "";
  });
}

function showHome(videos) {
  $("home").hidden = false;
  const items = videos.map((v) => {
    const card = el(v.video_found ? "a" : "div", v.video_found ? "video-card" : "video-card disabled");
    if (v.video_found) card.href = `?video=${encodeURIComponent(v.name)}`;
    card.append(el("strong", "", v.name), el("span", "stages", v.video_found ? v.stages.join(" → ") : "video file not found"));
    const li = el("li");
    li.append(card);
    return li;
  });
  $("video-list").replaceChildren(...(items.length ? items : [el("li", "muted", "No results yet.")]));
}

async function openVideo(name, startFrame) {
  state.name = name;
  document.title = `${name} · swim-analytics`;
  state.details = await getJSON(api(name));

  // Load the saved output of every stage that has a drawing layer
  const available = Object.keys(LAYERS).filter((layer) => state.details.stages.includes(LAYERS[layer].stage));
  if (!available.length) throw new Error(`No layer can draw the saved results of '${name}' (${state.details.stages.join(", ")})`);
  await Promise.all(available.map(async (layer) => {
    const data = await getJSON(api(name, `/results/${LAYERS[layer].stage}`));
    const frames = [];
    for (const f of data.frames) {
      f.rejected ??= []; // files saved before the filtering stage existed have no "rejected"
      frames[f.frame_idx] = f;
    }
    state.layers[layer] = { data, frames };
  }));

  const first = state.layers[available[0]].data;
  state.fps = first.fps;
  state.width = first.width;
  state.height = first.height;
  state.frameCount = state.details.video?.frame_count || first.frames.length;
  // Default: only the last stage's layer, as on the command line (earlier layers show the same boxes before later
  // stages changed them)
  state.enabled = new Set([available.at(-1)]);

  $("player").style.setProperty("--ar", state.width / state.height);
  $("viewer").hidden = false;
  buildLayerToggles();
  showSettings();
  checkResults();
  setupVideo(startFrame);
  setupControls();
  setupTimeline();
}

// ---------------------------------------------------------------------------------------------------------------
// Which frame is on screen
//
// video.currentTime is not precise enough to know the frame. requestVideoFrameCallback reports the presentation
// time of the frame actually shown (mediaTime); frame_idx = round(mediaTime * fps). Measured on our videos, real
// frame timestamps differ from frame_idx / fps by at most 0.08 frames, so this gives the frame the results are for.

const frameAtMediaTime = (t) => clampFrame(Math.round(t * state.fps));

function setupVideo(startFrame) {
  video.addEventListener("loadedmetadata", () => {
    if (video.videoWidth !== state.width || video.videoHeight !== state.height) {
      addWarning(`The browser shows the video as ${video.videoWidth}×${video.videoHeight}, the results are for ` +
        `${state.width}×${state.height}: boxes will be in the wrong place`);
    }
    seekToFrame(startFrame);
  }, { once: true });
  video.addEventListener("error", () => {
    showError(state.details.video
      ? "This browser can't play this video. iPhone videos are HEVC: use Safari or Chrome on macOS."
      : `Video file not found: ${state.details.video_path}`);
  });
  video.addEventListener("seeked", onSeeked);
  video.addEventListener("play", updatePlayButton);
  video.addEventListener("pause", () => {
    updatePlayButton();
    updateUrl();
  });

  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
    const onFrame = (_now, meta) => {
      showFrame(frameAtMediaTime(meta.mediaTime));
      video.requestVideoFrameCallback(onFrame);
    };
    video.requestVideoFrameCallback(onFrame);
  } else {
    addWarning("This browser can't report which frame is shown (no requestVideoFrameCallback): " +
      "while playing, boxes may be a frame off");
    const tick = () => {
      if (!video.paused) showFrame(clampFrame(Math.floor(video.currentTime * state.fps)));
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  video.src = api(state.name, "/video");
}

function seekToFrame(n) {
  n = clampFrame(n);
  state.requested = n;
  if (video.seeking) {
    state.pendingSeek = n; // only the latest target, once the current seek is done
    return;
  }
  // The middle of the frame: real timestamps differ slightly from frame_idx / fps, so seeking to the exact start
  // could show the frame before
  video.currentTime = (n + 0.5) / state.fps;
}

function onSeeked() {
  if (state.pendingSeek !== null) {
    const n = state.pendingSeek;
    state.pendingSeek = null;
    seekToFrame(n);
    return;
  }
  // requestVideoFrameCallback reports the new frame too; this covers browsers that skip it after a seek while paused
  if (video.paused) showFrame(clampFrame(Math.floor(video.currentTime * state.fps)));
}

function step(delta) {
  video.pause();
  const from = video.seeking || state.pendingSeek !== null ? state.requested : state.frame;
  seekToFrame(from + delta);
}

function togglePlay() {
  if (video.paused || video.ended) video.play().catch((e) => showError(`Can't play: ${e.message}`));
  else video.pause();
}

let lastShown = null;

function showFrame(n) {
  const changed = n !== lastShown;
  lastShown = n;
  state.frame = n;
  drawOverlay();
  drawTimeline();
  if (!changed) return;
  updatePosition();
  updateFrameBoxes();
  if (video.paused) updateUrl();
}

function updateUrl() {
  history.replaceState(null, "", `?video=${encodeURIComponent(state.name)}&frame=${state.frame}`);
}

// ---------------------------------------------------------------------------------------------------------------
// Drawing

// The last switched-on layer in pipeline order: its boxes are listed in the side panel and colored in the timeline
function primaryLayer() {
  return Object.keys(LAYERS).filter((layer) => state.enabled.has(layer)).at(-1) ?? null;
}

function frameData(layer, n) {
  // Frames missing from the results are drawn as "nothing found"
  return state.layers[layer].frames[n] ?? { frame_idx: n, detections: [], rejected: [] };
}

// Size the canvas to its on-screen size in device pixels (sharp on high-DPI screens) and clear it
function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(canvas.clientWidth * dpr);
  const h = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return ctx;
}

function drawOverlay() {
  const ctx = fitCanvas(overlay);
  const scale = overlay.width / state.width; // canvas pixels per video pixel
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  const px = (window.devicePixelRatio || 1) / scale; // one screen pixel, in video pixels
  const primary = primaryLayer();
  for (const layer of Object.keys(LAYERS)) {
    if (!state.enabled.has(layer)) continue;
    const highlight = layer === primary ? state.highlight : null;
    LAYERS[layer].draw(ctx, frameData(layer, state.frame), { px, width: state.width, height: state.height, highlight });
  }
}

// Timeline: one bar per frame, colored by the number of persons kept; a mark on top where boxes were rejected.
// The bars only change with the layer or the size, so they are drawn once into `timelineBars` and copied.
let timelineBars = null;

function drawTimeline() {
  const ctx = fitCanvas(timeline);
  const layer = primaryLayer() ?? Object.keys(state.layers).at(-1);
  if (!timelineBars || timelineBars.width !== timeline.width || timelineBars.height !== timeline.height ||
      timelineBars.layer !== layer) {
    timelineBars = drawTimelineBars(layer, timeline.width, timeline.height);
    $("timeline-layer").textContent = `timeline: ${layer}`;
  }
  ctx.drawImage(timelineBars.canvas, 0, 0);

  const dpr = window.devicePixelRatio || 1;
  const x = ((state.frame + 0.5) / state.frameCount) * timeline.width;
  ctx.fillStyle = css("--tl-playhead");
  ctx.fillRect(Math.round(x - dpr), 0, Math.round(2 * dpr), timeline.height);
}

function drawTimelineBars(layer, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const markH = Math.round(5 * dpr);
  const markW = Math.round(2 * dpr); // at least 2 screen pixels, so a single frame's mark stays visible
  const gap = Math.round(2 * dpr);
  const colors = [css("--tl-gap"), css("--tl-one"), css("--tl-many")];
  const rejectedColor = css("--tl-rejected");
  const n = state.frameCount;
  for (let i = 0; i < n; i++) {
    const f = frameData(layer, i);
    const x0 = Math.floor((i / n) * width);
    const w = Math.max(Math.floor(((i + 1) / n) * width) - x0, 1);
    ctx.fillStyle = colors[Math.min(f.detections.length, 2)];
    ctx.fillRect(x0, markH + gap, w, height - markH - gap);
    if (f.rejected.length) {
      ctx.fillStyle = rejectedColor;
      ctx.fillRect(x0, 0, Math.max(w, markW), markH);
    }
  }
  return { canvas, width, height, layer };
}

// ---------------------------------------------------------------------------------------------------------------
// Controls, timeline, side panel

function setupControls() {
  $("play").addEventListener("click", togglePlay);
  $("prev").addEventListener("click", () => step(-1));
  $("next").addEventListener("click", () => step(1));
  // Without focus, Space doesn't also "click" the last clicked button
  for (const button of document.querySelectorAll(".controls button")) {
    button.addEventListener("click", () => button.blur());
  }
  $("speed").addEventListener("change", (e) => {
    video.playbackRate = Number(e.target.value);
    e.target.blur();
  });

  document.addEventListener("keydown", (e) => {
    if (e.target.closest("select, input, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
    const actions = {
      " ": togglePlay,
      ArrowLeft: () => step(e.shiftKey ? -10 : -1),
      ArrowRight: () => step(e.shiftKey ? 10 : 1),
      Home: () => step(-state.frameCount),
      End: () => step(state.frameCount),
    };
    if (actions[e.key]) {
      e.preventDefault();
      actions[e.key]();
    }
  });
}

function updatePlayButton() {
  $("play").classList.toggle("playing", !video.paused && !video.ended);
}

function updatePosition() {
  const position = $("position");
  position.replaceChildren(
    `frame ${state.frame}`, el("span", "muted", ` / ${state.frameCount - 1}`),
    el("span", "time", `${(state.frame / state.fps).toFixed(2)} s`),
    el("span", "muted", ` / ${(state.frameCount / state.fps).toFixed(2)} s`),
  );
}

function setupTimeline() {
  const frameAt = (e) => {
    const rect = timeline.getBoundingClientRect();
    return Math.floor(((e.clientX - rect.left) / rect.width) * state.frameCount);
  };
  timeline.addEventListener("pointerdown", (e) => {
    timeline.setPointerCapture(e.pointerId);
    video.pause();
    seekToFrame(frameAt(e));
  });
  timeline.addEventListener("pointermove", (e) => {
    if (timeline.hasPointerCapture(e.pointerId)) seekToFrame(frameAt(e));
  });
  new ResizeObserver(() => {
    drawOverlay();
    drawTimeline();
  }).observe($("player"));
}

function buildLayerToggles() {
  const toggles = Object.entries(LAYERS).map(([layer, { stage, description }]) => {
    const input = el("input");
    input.type = "checkbox";
    input.checked = state.enabled.has(layer);
    input.disabled = !state.layers[layer];
    input.addEventListener("change", () => {
      if (input.checked) state.enabled.add(layer);
      else state.enabled.delete(layer);
      lastShown = null; // re-list the side panel's boxes for the new primary layer
      showFrame(state.frame);
      input.blur();
    });
    const text = el("span");
    text.append(el("strong", "", layer), el("small", "", state.layers[layer] ? description : `${stage} not run yet`));
    const label = el("label", "layer");
    label.append(input, text);
    return label;
  });
  $("layers").replaceChildren(...toggles);
}

function updateFrameBoxes() {
  state.highlight = null;
  const layer = primaryLayer();
  $("frame-title").textContent = layer ? `Frame ${state.frame} · ${layer}` : `Frame ${state.frame}`;
  const list = $("frame-boxes");
  if (!layer) {
    list.replaceChildren(el("li", "summary", "Switch a layer on to see its boxes."));
    return;
  }
  const f = frameData(layer, state.frame);
  const rows = [
    ...f.detections.map((d, i) => boxRow("detections", i, d, d.label)),
    ...f.rejected.map((r, i) => boxRow("rejected", i, r.detection, r.reason)),
  ];
  const rejected = f.rejected.length ? `${f.rejected.length} rejected` : null;
  const summary = f.detections.length
    ? el("li", "summary", [`${f.detections.length} ${f.detections.length === 1 ? "person" : "persons"}`, rejected]
      .filter(Boolean).join(", "))
    : el("li", "empty", rejected ? `No person in this frame (${rejected})` : "No person in this frame");
  list.replaceChildren(summary, ...rows);
}

function boxRow(kind, index, detection, name) {
  const { x1, y1, x2, y2 } = detection.bbox;
  const li = el("li", kind === "rejected" ? "box rejected" : "box");
  li.append(
    el("span", "swatch"),
    el("span", "name", name),
    el("span", "score", detection.score.toFixed(2)),
    el("span", "size", `${Math.round(x2 - x1)} × ${Math.round(y2 - y1)} px at (${Math.round(x1)}, ${Math.round(y1)})`),
  );
  li.addEventListener("mouseenter", () => {
    state.highlight = { kind, index };
    drawOverlay();
  });
  li.addEventListener("mouseleave", () => {
    state.highlight = null;
    drawOverlay();
  });
  return li;
}

function showSettings() {
  const { config, stages } = state.details;
  const dl = $("settings");
  if (!config) {
    dl.replaceChildren(el("dd", "muted", "No config.json"));
    return;
  }
  // Only stages with saved results: config.json may also hold settings of stages that were not run
  const rows = stages.filter((stage) => config[stage]).flatMap((stage) => [
    el("dt", "", stage),
    el("dd", "", Object.entries(config[stage]).map(([key, value]) => `${key} ${value}`).join(" · ")),
  ]);
  dl.replaceChildren(...rows);
}

function checkResults() {
  const v = state.details.video;
  if (!v) {
    addWarning(`Video file not found: ${state.details.video_path}`);
    return;
  }
  for (const [layer, { data }] of Object.entries(state.layers)) {
    if (data.width !== v.width || data.height !== v.height) {
      addWarning(`${layer}: made on a ${data.width}×${data.height} video, this one is ${v.width}×${v.height} (wrong results folder?)`);
    }
    if (data.frames.length !== v.frame_count) {
      addWarning(`${layer}: results for ${data.frames.length} frames, the video has ${v.frame_count}`);
    }
  }
}

function addWarning(text) {
  $("warnings-section").hidden = false;
  $("warnings").append(el("li", "", text));
}

main().catch((e) => showError(e.message));
