// Labels page (docs/labeling.md): check and fix the pool labels SAM3 made for frames picked for training.
// A frame's labels are shapes: the pool's water as polygons, each lane rope as a straight line with a width at each
// end, in pixels of the full-size frame. Edits are points: drag corners and rope ends, add, delete, join.

import { $, el, hideError, requestJSON, showError } from "./dom.js";

const canvas = $("canvas");
const ctx = canvas.getContext("2d");

// Same colors as the segmentation videos (research/pool-segmentation/): water blue, rope orange
const WATER = { fill: [0, 110, 255], line: "rgb(120, 200, 255)" };
const ROPE = { fill: [255, 140, 0], line: "rgb(255, 190, 80)" };
const HANDLE = 6; // handle radius, screen pixels
const HIT = 8; // how close (screen pixels) a click must be to a point or line

const state = {
  frames: [], // GET /api/labels: name, video, frame_idx, status, counts of shapes
  frame: null, // the open frame: name, video, frame_idx, width, height
  shapes: null, // {water: [[[x, y], ...]], ropes: [{points: [[x, y], [x, y]], widths: [w1, w2]}]}
  status: "sam3",
  sam3: null, // the model's shapes, to tell "fixed" from "accepted" and for "Back to SAM3's labels"
  saved: "", // shapes as last saved (JSON), to know whether there are unsaved changes
  image: null,
  view: { scale: 1, x: 0, y: 0 }, // screen = image point * scale + (x, y), in CSS pixels
  selection: null, // {kind: "polygon" | "rope", index, corner?, end?}
  mode: "edit", // "edit" | "add-rope" | "add-water" | "join"
  draft: [], // points clicked while adding a shape
  mouse: null, // last mouse position over the canvas (screen)
  drag: null, // {type: "pan" | "corner" | "end" | "width", ...}
  undo: [],
  redo: [],
  show: true,
  opacity: 0.3,
  filter: "all",
};

const shapesJSON = (shapes) => JSON.stringify(shapes);
const rounded = (shapes) => JSON.stringify(shapes, (_, v) => (typeof v === "number" ? Math.round(v * 10) / 10 : v));
const changed = () => state.shapes && shapesJSON(state.shapes) !== state.saved;
const isChecked = (status) => status === "fixed" || status === "accepted";

// ---------------------------------------------------------------------------------------------------------------
// Start

async function main() {
  const listing = await requestJSON("/api/labels");
  state.frames = listing.frames;
  if (!state.frames.length) {
    $("labeling").hidden = true;
    $("no-dataset").hidden = false;
    $("dataset-dir").textContent = listing.dataset;
    return;
  }
  setupControls();
  setupCanvas();
  const fromUrl = new URLSearchParams(location.search).get("frame");
  const first = state.frames.find((f) => f.name === fromUrl) ?? state.frames.find((f) => !isChecked(f.status)) ?? state.frames[0];
  renderList();
  await openFrame(first.name);
}

async function openFrame(name) {
  const data = await requestJSON(`/api/labels/${encodeURIComponent(name)}`);
  const image = new Image();
  image.src = `/api/labels/${encodeURIComponent(name)}/image`;
  await image.decode();
  state.frame = data.frame;
  state.image = image;
  state.shapes = { water: data.labels.water, ropes: data.labels.ropes };
  state.status = data.labels.status;
  state.sam3 = data.sam3 ? { water: data.sam3.water, ropes: data.sam3.ropes } : null;
  state.saved = shapesJSON(state.shapes);
  state.selection = null;
  state.undo = [];
  state.redo = [];
  setMode("edit");
  history.replaceState(null, "", `?frame=${encodeURIComponent(name)}`);
  document.title = `${name} · Labels · swim-analytics`;
  fit();
  updatePanel();
  renderList();
  $("message").textContent = "";
}

// Save the open frame: "fixed" if its shapes differ from SAM3's, "accepted" if not
async function save() {
  const status = state.sam3 && rounded(state.shapes) === rounded(state.sam3) ? "accepted" : "fixed";
  const body = { status, ...state.shapes };
  const result = await requestJSON(`/api/labels/${encodeURIComponent(state.frame.name)}`, body, "PUT");
  state.status = result.labels.status;
  state.shapes = { water: result.labels.water, ropes: result.labels.ropes };
  state.saved = shapesJSON(state.shapes);
  const item = state.frames.find((f) => f.name === state.frame.name);
  Object.assign(item, { status: state.status, water: state.shapes.water.length, ropes: state.shapes.ropes.length });
  $("message").textContent = `Saved as ${state.status}.`;
  updatePanel();
  renderList();
  draw();
}

// Go to another frame; unsaved changes are saved first (they are a check of this frame)
async function go(name) {
  try {
    if (changed()) await save();
    await openFrame(name);
    hideError();
  } catch (error) {
    showError(error.message);
  }
}

function neighbour(delta) {
  const list = visibleFrames();
  const k = list.findIndex((f) => f.name === state.frame.name);
  return list[k + delta] ?? null;
}

// ---------------------------------------------------------------------------------------------------------------
// Frame list and panel

const visibleFrames = () => state.frames.filter((f) => state.filter === "all"
  || (state.filter === "todo" ? !isChecked(f.status) : f.status === state.filter) || f.name === state.frame?.name);

function renderList() {
  const items = visibleFrames().map((f) => {
    const li = el("li", `frame-item ${f.status.replace(" ", "-")}${f.name === state.frame?.name ? " current" : ""}`);
    li.append(el("span", "dot"), el("span", "frame-item-name", f.name), el("span", "frame-item-status", f.status));
    li.title = `${f.video}, frame ${f.frame_idx}: ${f.water} water area(s), ${f.ropes} rope(s)`;
    li.addEventListener("click", () => go(f.name));
    return li;
  });
  $("frame-list").replaceChildren(...items);
  $("frame-list").querySelector(".current")?.scrollIntoView({ block: "nearest" });
  const checked = state.frames.filter((f) => isChecked(f.status)).length;
  $("progress").textContent = `${checked} of ${state.frames.length} checked`;
}

function updatePanel() {
  const f = state.frame;
  $("frame-name").textContent = f.name;
  $("frame-info").textContent = `${f.video}, frame ${f.frame_idx} (${f.width}×${f.height})`;
  const labels = { sam3: "SAM3, not checked", fixed: "fixed", accepted: "accepted (SAM3 was right)" };
  $("status").textContent = labels[state.status] ?? state.status;
  $("status").className = `status ${isChecked(state.status) ? "ok" : "warn"}`;
  $("unsaved").hidden = !changed();
  $("reset").hidden = !state.sam3;
  $("undo").disabled = !state.undo.length;
  $("redo").disabled = !state.redo.length;
  $("join").disabled = state.selection?.kind !== "rope";
  $("delete").disabled = !state.selection;
  $("shape-counts").textContent = `${state.shapes.water.length} water area(s), ${state.shapes.ropes.length} rope(s)`;
}

function setupControls() {
  $("filter").addEventListener("change", (e) => {
    state.filter = e.target.value;
    renderList();
  });
  $("save").addEventListener("click", () => save().catch((e) => showError(e.message)));
  $("accept").addEventListener("click", saveAndNext);
  $("reset").addEventListener("click", () => edit(() => {
    state.shapes = structuredClone(state.sam3);
    state.selection = null;
  }));
  $("add-rope").addEventListener("click", () => setMode("add-rope"));
  $("add-water").addEventListener("click", () => setMode("add-water"));
  $("cut-water").addEventListener("click", () => setMode("cut-water"));
  $("join").addEventListener("click", () => setMode("join"));
  $("delete").addEventListener("click", deleteSelection);
  $("undo").addEventListener("click", undo);
  $("redo").addEventListener("click", redo);
  $("opacity").addEventListener("input", (e) => {
    state.opacity = Number(e.target.value);
    draw();
  });
  $("show").addEventListener("change", (e) => {
    state.show = e.target.checked;
    draw();
  });
  for (const button of document.querySelectorAll(".label-panel button")) button.addEventListener("click", () => button.blur());

  document.addEventListener("keydown", (e) => {
    if (e.target.closest("input, select, textarea")) return;
    const key = e.key.toLowerCase();
    const command = e.metaKey || e.ctrlKey;
    const actions = {
      arrowleft: () => neighbour(-1) && go(neighbour(-1).name),
      arrowright: () => neighbour(1) && go(neighbour(1).name),
      a: saveAndNext,
      s: () => save().catch((err) => showError(err.message)),
      z: () => (e.shiftKey ? redo() : undo()),
      y: redo,
      r: () => setMode("add-rope"),
      w: () => setMode("add-water"),
      e: () => setMode("cut-water"),
      j: () => state.selection?.kind === "rope" && setMode("join"),
      h: () => {
        state.show = !state.show;
        $("show").checked = state.show;
        draw();
      },
      f: fit,
      0: fit,
      delete: deleteSelection,
      backspace: deleteSelection,
      enter: finishWater,
      escape: () => {
        if (state.mode !== "edit") setMode("edit");
        else state.selection = null;
        updatePanel();
        draw();
      },
    };
    if (command && key !== "z") return; // leave the browser's own shortcuts alone
    if (actions[key]) {
      e.preventDefault();
      actions[key]();
    }
  });
  window.addEventListener("beforeunload", (e) => {
    if (changed()) e.preventDefault();
  });
}

async function saveAndNext() {
  try {
    await save();
    const next = neighbour(1) ?? state.frames.find((f) => !isChecked(f.status));
    if (next && next.name !== state.frame.name) await openFrame(next.name);
  } catch (error) {
    showError(error.message);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Editing: every change goes through edit(), which keeps the undo history

function edit(change) {
  state.undo.push(shapesJSON(state.shapes));
  state.redo = [];
  change();
  updatePanel();
  draw();
}

function undo() {
  if (!state.undo.length) return;
  state.redo.push(shapesJSON(state.shapes));
  state.shapes = JSON.parse(state.undo.pop());
  state.selection = null;
  updatePanel();
  draw();
}

function redo() {
  if (!state.redo.length) return;
  state.undo.push(shapesJSON(state.shapes));
  state.shapes = JSON.parse(state.redo.pop());
  state.selection = null;
  updatePanel();
  draw();
}

function setMode(mode) {
  state.mode = mode;
  state.draft = [];
  const hints = {
    "add-rope": "Click the two ends of the rope (Esc to cancel)",
    "add-water": "Click around the area to add to the water, Enter to finish (Esc to cancel)",
    "cut-water": "Click around the area to remove from the water, Enter to finish (Esc to cancel)",
    join: "Click the rope to join with the selected one (Esc to cancel)",
  };
  $("mode-hint").hidden = mode === "edit";
  $("mode-hint").textContent = hints[mode] ?? "";
  canvas.style.cursor = mode === "edit" ? "grab" : "crosshair";
  for (const [id, m] of [["add-rope", "add-rope"], ["add-water", "add-water"], ["cut-water", "cut-water"], ["join", "join"]]) {
    $(id).classList.toggle("active", state.mode === m);
  }
  draw();
}

function deleteSelection() {
  const s = state.selection;
  if (!s) return;
  edit(() => {
    if (s.kind === "rope") state.shapes.ropes.splice(s.index, 1);
    else if (s.corner !== undefined && state.shapes.water[s.index].length > 3) state.shapes.water[s.index].splice(s.corner, 1);
    else state.shapes.water.splice(s.index, 1); // a polygon, or a corner of a triangle: the whole area
    state.selection = null;
  });
}

// + Water / - Water: the server merges the drawn area into the water or cuts it out, and simplifies the outlines
async function finishWater() {
  if ((state.mode !== "add-water" && state.mode !== "cut-water") || state.draft.length < 3) return;
  const body = { water: state.shapes.water, [state.mode === "add-water" ? "add" : "subtract"]: state.draft };
  setMode("edit");
  try {
    const { water } = await requestJSON(`/api/labels/${encodeURIComponent(state.frame.name)}/water`, body);
    edit(() => {
      state.shapes.water = water;
      state.selection = null;
    });
  } catch (error) {
    showError(error.message);
  }
}

// Width of a new rope: the median of the frame's other ropes, else 8 px
function defaultWidth() {
  const widths = state.shapes.ropes.flatMap((r) => r.widths).sort((a, b) => a - b);
  return widths.length ? widths[Math.floor(widths.length / 2)] : 8;
}

// One rope from two: the two farthest ends along the longer rope, with the widths at those ends
function joinRopes(a, b) {
  const longer = length(a) >= length(b) ? a : b;
  const [p, q] = longer.points;
  const d = [(q[0] - p[0]) / length(longer), (q[1] - p[1]) / length(longer)];
  const ends = [...a.points.map((pt, k) => [pt, a.widths[k]]), ...b.points.map((pt, k) => [pt, b.widths[k]])];
  const along = ends.map(([pt]) => pt[0] * d[0] + pt[1] * d[1]);
  const first = ends[along.indexOf(Math.min(...along))];
  const last = ends[along.indexOf(Math.max(...along))];
  return { points: [first[0], last[0]], widths: [first[1], last[1]] };
}

const length = (rope) => Math.hypot(rope.points[1][0] - rope.points[0][0], rope.points[1][1] - rope.points[0][1]);

// ---------------------------------------------------------------------------------------------------------------
// View: fit, zoom, pan

const toScreen = ([x, y]) => [x * state.view.scale + state.view.x, y * state.view.scale + state.view.y];
const toImage = ([x, y]) => [(x - state.view.x) / state.view.scale, (y - state.view.y) / state.view.scale];
const clampToImage = ([x, y]) => [Math.min(Math.max(x, 0), state.frame.width), Math.min(Math.max(y, 0), state.frame.height)];

function fit() {
  if (!state.frame) return;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const scale = Math.min(w / state.frame.width, h / state.frame.height) * 0.98;
  state.view = { scale, x: (w - state.frame.width * scale) / 2, y: (h - state.frame.height * scale) / 2 };
  draw();
}

function zoomAt(point, factor) {
  const fitScale = Math.min(canvas.clientWidth / state.frame.width, canvas.clientHeight / state.frame.height);
  const scale = Math.min(Math.max(state.view.scale * factor, fitScale * 0.5), 20);
  const [ix, iy] = toImage(point);
  state.view = { scale, x: point[0] - ix * scale, y: point[1] - iy * scale };
  draw();
}

// ---------------------------------------------------------------------------------------------------------------
// Drawing

function draw() {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(canvas.clientWidth * dpr);
  const h = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!state.image) return;
  ctx.setTransform(dpr * state.view.scale, 0, 0, dpr * state.view.scale, dpr * state.view.x, dpr * state.view.y);
  ctx.drawImage(state.image, 0, 0);
  if (state.show) {
    const px = 1 / state.view.scale; // one screen pixel, in image pixels
    state.shapes.water.forEach((polygon, i) => {
      const selected = state.selection?.kind === "polygon" && state.selection.index === i;
      ctx.beginPath();
      polygon.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.closePath();
      ctx.fillStyle = `rgba(${WATER.fill}, ${state.opacity})`;
      ctx.fill();
      ctx.strokeStyle = selected ? "#ffffff" : WATER.line;
      ctx.lineWidth = (selected ? 3 : 2) * px;
      ctx.stroke();
    });
    state.shapes.ropes.forEach((rope, i) => {
      const selected = state.selection?.kind === "rope" && state.selection.index === i;
      const corners = ropeCorners(rope);
      ctx.beginPath();
      corners.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.closePath();
      ctx.fillStyle = `rgba(${ROPE.fill}, ${Math.min(state.opacity + 0.25, 0.9)})`;
      ctx.fill();
      ctx.strokeStyle = selected ? "#ffffff" : ROPE.line;
      ctx.lineWidth = (selected ? 2 : 1) * px;
      ctx.stroke();
    });
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // handles: screen pixels, same size at any zoom
  if (state.show) drawHandles();
  drawDraft();
}

function drawHandles() {
  const sel = state.selection;
  state.shapes.water.forEach((polygon, i) => {
    for (const p of polygon) square(toScreen(p), HANDLE - 1, WATER.line);
    if (sel?.kind === "polygon" && sel.index === i) {
      if (sel.corner !== undefined) square(toScreen(polygon[sel.corner]), HANDLE + 1, "#ffffff");
      polygon.forEach((p, k) => plus(toScreen(midpoint(p, polygon[(k + 1) % polygon.length]))));
    }
  });
  state.shapes.ropes.forEach((rope, i) => {
    for (const p of rope.points) circle(toScreen(p), HANDLE, ROPE.line);
    if (sel?.kind === "rope" && sel.index === i) {
      for (const [end, side] of widthHandles()) diamond(toScreen(widthHandlePoint(rope, end, side)), "#ffffff");
    }
  });
}

function drawDraft() {
  if (state.mode === "edit" || !state.draft.length) return;
  const pts = [...state.draft.map(toScreen), ...(state.mouse ? [state.mouse] : [])];
  ctx.strokeStyle = state.mode === "add-rope" ? ROPE.line : state.mode === "cut-water" ? "#ff6b6b" : WATER.line;
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  pts.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.stroke();
  ctx.setLineDash([]);
  for (const p of state.draft) circle(toScreen(p), HANDLE - 1, "#ffffff");
}

function square([x, y], r, color) {
  ctx.fillStyle = color;
  ctx.strokeStyle = "#0b1220";
  ctx.lineWidth = 1;
  ctx.fillRect(x - r, y - r, 2 * r, 2 * r);
  ctx.strokeRect(x - r, y - r, 2 * r, 2 * r);
}

function circle([x, y], r, color) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = "#0b1220";
  ctx.lineWidth = 1;
  ctx.stroke();
}

function diamond([x, y], color) {
  const r = HANDLE;
  ctx.beginPath();
  ctx.moveTo(x, y - r);
  ctx.lineTo(x + r, y);
  ctx.lineTo(x, y + r);
  ctx.lineTo(x - r, y);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = "#0b1220";
  ctx.stroke();
}

function plus([x, y]) {
  ctx.beginPath();
  ctx.arc(x, y, HANDLE - 1, 0, 2 * Math.PI);
  ctx.fillStyle = "rgba(11, 18, 32, 0.75)";
  ctx.fill();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(x - 3, y);
  ctx.lineTo(x + 3, y);
  ctx.moveTo(x, y - 3);
  ctx.lineTo(x, y + 3);
  ctx.stroke();
}

const midpoint = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];

// The rope's band (4 corners), as swim/labeling/labels.py rope_corners()
function ropeCorners(rope) {
  const [[x1, y1], [x2, y2]] = rope.points;
  const len = Math.hypot(x2 - x1, y2 - y1) || 1;
  const [ax, ay] = [-(y2 - y1) / len, (x2 - x1) / len];
  const [w1, w2] = rope.widths.map((w) => w / 2);
  return [[x1 + ax * w1, y1 + ay * w1], [x2 + ax * w2, y2 + ay * w2], [x2 - ax * w2, y2 - ay * w2], [x1 - ax * w1, y1 - ay * w1]];
}

const widthHandles = () => [[0, 1], [0, -1], [1, 1], [1, -1]];

function widthHandlePoint(rope, end, side) {
  const [[x1, y1], [x2, y2]] = rope.points;
  const len = Math.hypot(x2 - x1, y2 - y1) || 1;
  const [ax, ay] = [-(y2 - y1) / len, (x2 - x1) / len];
  const [x, y] = rope.points[end];
  const r = rope.widths[end] / 2 + 6 / state.view.scale; // just outside the band
  return [x + side * ax * r, y + side * ay * r];
}

// ---------------------------------------------------------------------------------------------------------------
// Mouse: what is under the pointer, dragging, clicking

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

function distanceToSegment(p, a, b) {
  const [dx, dy] = [b[0] - a[0], b[1] - a[1]];
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy || 1)));
  return dist(p, [a[0] + t * dx, a[1] + t * dy]);
}

function insidePolygon([x, y], polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

// What a click at screen point `s` hits, most specific first
function hitTest(s) {
  const sel = state.selection;
  if (sel?.kind === "rope") {
    const rope = state.shapes.ropes[sel.index];
    for (const [end, side] of widthHandles()) {
      if (dist(s, toScreen(widthHandlePoint(rope, end, side))) <= HIT) return { kind: "width", index: sel.index, end };
    }
  }
  for (let i = state.shapes.ropes.length - 1; i >= 0; i--) {
    const end = state.shapes.ropes[i].points.findIndex((p) => dist(s, toScreen(p)) <= HIT);
    if (end >= 0) return { kind: "end", index: i, end };
  }
  for (let i = state.shapes.water.length - 1; i >= 0; i--) {
    const corner = state.shapes.water[i].findIndex((p) => dist(s, toScreen(p)) <= HIT);
    if (corner >= 0) return { kind: "corner", index: i, corner };
  }
  if (sel?.kind === "polygon") {
    const polygon = state.shapes.water[sel.index];
    for (let k = 0; k < polygon.length; k++) {
      if (dist(s, toScreen(midpoint(polygon[k], polygon[(k + 1) % polygon.length]))) <= HIT) {
        return { kind: "mid", index: sel.index, edge: k };
      }
    }
  }
  const p = toImage(s);
  for (let i = state.shapes.ropes.length - 1; i >= 0; i--) {
    const rope = state.shapes.ropes[i];
    const reach = Math.max(Math.max(...rope.widths) / 2, HIT / state.view.scale);
    if (distanceToSegment(p, ...rope.points) <= reach) return { kind: "rope", index: i };
  }
  for (let i = state.shapes.water.length - 1; i >= 0; i--) {
    if (insidePolygon(p, state.shapes.water[i])) return { kind: "polygon", index: i };
  }
  return null;
}

function setupCanvas() {
  const point = (e) => {
    const r = canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };
  canvas.addEventListener("pointerdown", (e) => {
    if (!state.frame) return;
    canvas.setPointerCapture(e.pointerId);
    const s = point(e);
    if (state.mode !== "edit") {
      clickInMode(s);
      return;
    }
    const hit = e.button === 0 && state.show ? hitTest(s) : null; // hidden labels can't be edited
    const before = shapesJSON(state.shapes);
    if (hit?.kind === "corner" || hit?.kind === "end" || hit?.kind === "width") {
      state.selection = hit.kind === "corner" ? { kind: "polygon", index: hit.index, corner: hit.corner }
        : { kind: "rope", index: hit.index, end: hit.end };
      state.drag = { type: hit.kind, hit, before, moved: false };
    } else if (hit?.kind === "mid") {
      const polygon = state.shapes.water[hit.index];
      polygon.splice(hit.edge + 1, 0, midpoint(polygon[hit.edge], polygon[(hit.edge + 1) % polygon.length]));
      state.selection = { kind: "polygon", index: hit.index, corner: hit.edge + 1 };
      state.drag = { type: "corner", hit: { kind: "corner", index: hit.index, corner: hit.edge + 1 }, before, moved: true };
    } else {
      // A click selects the rope or water area under it; a drag moves the view
      state.drag = { type: "pan", start: s, last: s, hit, moved: false };
      canvas.style.cursor = "grabbing";
    }
    updatePanel();
    draw();
  });
  canvas.addEventListener("pointermove", (e) => {
    const s = point(e);
    state.mouse = s;
    const d = state.drag;
    if (!d) {
      if (state.mode !== "edit") draw();
      else canvas.style.cursor = hoverCursor(state.show ? hitTest(s) : null);
      return;
    }
    if (d.type === "pan") {
      if (dist(s, d.start) > 3) d.moved = true;
      state.view.x += s[0] - d.last[0];
      state.view.y += s[1] - d.last[1];
      d.last = s;
    } else {
      d.moved = true;
      const p = clampToImage(toImage(s));
      if (d.type === "corner") state.shapes.water[d.hit.index][d.hit.corner] = p;
      if (d.type === "end") state.shapes.ropes[d.hit.index].points[d.hit.end] = p;
      if (d.type === "width") {
        const rope = state.shapes.ropes[d.hit.index];
        const [a, b] = rope.points;
        const len = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
        const across = Math.abs(((p[0] - rope.points[d.hit.end][0]) * -(b[1] - a[1]) + (p[1] - rope.points[d.hit.end][1]) * (b[0] - a[0])) / len);
        rope.widths[d.hit.end] = Math.max(1, 2 * across - 12 / state.view.scale); // the handle sits 6 px outside the band
      }
      updatePanel();
    }
    draw();
  });
  const end = () => {
    const d = state.drag;
    state.drag = null;
    if (!d) return;
    canvas.style.cursor = "grab";
    if (d.type === "pan") {
      if (!d.moved) {
        state.selection = d.hit && (d.hit.kind === "rope" || d.hit.kind === "polygon") ? { kind: d.hit.kind, index: d.hit.index } : null;
      }
    } else if (d.moved) {
      // The whole drag is one step of undo
      state.undo.push(d.before);
      state.redo = [];
    }
    updatePanel();
    draw();
  };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);
  canvas.addEventListener("dblclick", () => finishWater());
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    zoomAt(point(e), Math.exp(-e.deltaY * 0.0015));
  }, { passive: false });
  canvas.addEventListener("pointerleave", () => {
    state.mouse = null;
    draw();
  });
  new ResizeObserver(() => draw()).observe(canvas);
}

function hoverCursor(hit) {
  if (!hit) return "grab";
  if (hit.kind === "corner" || hit.kind === "end" || hit.kind === "width" || hit.kind === "mid") return "move";
  return "pointer";
}

function clickInMode(s) {
  if (state.mode === "join") {
    const hit = hitTest(s);
    const sel = state.selection;
    const other = hit && (hit.kind === "rope" || hit.kind === "end") ? hit.index : null;
    if (sel?.kind === "rope" && other !== null && other !== sel.index) {
      edit(() => {
        const joined = joinRopes(state.shapes.ropes[sel.index], state.shapes.ropes[other]);
        state.shapes.ropes = state.shapes.ropes.filter((_, k) => k !== sel.index && k !== other);
        state.shapes.ropes.push(joined);
        state.selection = { kind: "rope", index: state.shapes.ropes.length - 1 };
      });
      setMode("edit");
    }
    return;
  }
  state.draft.push(clampToImage(toImage(s)));
  if (state.mode === "add-rope" && state.draft.length === 2) {
    const points = state.draft;
    const width = defaultWidth();
    edit(() => {
      state.shapes.ropes.push({ points, widths: [width, width] });
      state.selection = { kind: "rope", index: state.shapes.ropes.length - 1 };
    });
    setMode("edit");
  }
  draw();
}

// For the browser test: image point -> screen point, and the shapes being edited
window.__labels = { toScreen: (x, y) => toScreen([x, y]), shapes: () => state.shapes, frame: () => state.frame?.name };

main().catch((error) => showError(error.message));
