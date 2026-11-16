// Labels page (docs/labeling.md): check and fix the pool labels SAM3 made for frames picked for training.
// A frame's labels are polygons, in pixels of the full-size frame: the pool's water, and each lane rope's outline.
// Edits are dots: click dots around an area to add it, drag corners, add corners on edges, delete, join ropes.

import { $, el, hideError, requestJSON, showError } from "./dom.js";

const canvas = $("canvas");
const ctx = canvas.getContext("2d");

// Same colors as the segmentation videos (research/pool-segmentation/): water blue, rope orange
const WATER = { fill: [0, 110, 255], line: "rgb(120, 200, 255)" };
const ROPE = { fill: [255, 140, 0], line: "rgb(255, 190, 80)" };
const HANDLE = 6; // handle radius, screen pixels
const HIT = 8; // how close (screen pixels) a click must be to a point or line
const CLOSE = 5; // screen pixels: while adding an area, a click this close to the first dot closes it
const SAME_DOT = 3; // screen pixels: a click this close to the last dot is the same dot (a double-click)

const state = {
  frames: [], // GET /api/labels: name, video, frame_idx, status, counts of shapes
  frame: null, // the open frame: name, video, frame_idx, width, height
  shapes: null, // {water: [polygon, ...], ropes: [polygon, ...]}, polygon: [[x, y], ...]
  status: "sam3",
  sam3: null, // the model's shapes, to tell "fixed" from "accepted" and for "Back to SAM3's labels"
  saved: "", // shapes as last saved (JSON), to know whether there are unsaved changes
  image: null,
  view: { scale: 1, x: 0, y: 0 }, // screen = image point * scale + (x, y), in CSS pixels
  selection: null, // {layer: "water" | "ropes", index, corner?}
  mode: "edit", // "edit" | "add-rope" | "add-water" | "cut-water" | "join"
  draft: [], // dots clicked while adding an area
  mouse: null, // last mouse position over the canvas (screen)
  drag: null, // {type: "pan" | "corner", ...}
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
  $("join").disabled = state.selection?.layer !== "ropes";
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
    hideClassTip();
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
      z: () => (DRAW_MODES.includes(state.mode) ? deleteKey() : e.shiftKey ? redo() : undo()), // while adding: the last dot
      y: redo,
      r: () => setMode("add-rope"),
      w: () => setMode("add-water"),
      e: () => setMode("cut-water"),
      j: () => state.selection?.layer === "ropes" && setMode("join"),
      h: () => {
        state.show = !state.show;
        $("show").checked = state.show;
        hideClassTip();
        draw();
      },
      f: fit,
      0: fit,
      delete: deleteKey,
      backspace: deleteKey,
      enter: finishDraft,
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
  updateHint();
  hideClassTip();
  canvas.style.cursor = mode === "edit" ? "grab" : "crosshair";
  for (const m of ["add-rope", "add-water", "cut-water", "join"]) $(m).classList.toggle("active", state.mode === m);
  draw();
}

// What to do now, above the frame
function updateHint() {
  const finish = "click the first dot or press Enter to finish (drag: move the frame; Backspace: remove the last dot; Esc: cancel)";
  const hints = {
    "add-rope": `Rope: click dots around it; ${finish}`,
    "add-water": `Add water: click dots around the area; ${finish}`,
    "cut-water": `Remove water: click dots around the area; ${finish}`,
    join: "Click the rope to join with the selected one (Esc: cancel)",
  };
  $("mode-hint").hidden = state.mode === "edit";
  $("mode-hint").textContent = hints[state.mode] ?? "";
}

const DRAW_MODES = ["add-rope", "add-water", "cut-water"];

// Delete / Backspace: while drawing, the last dot; else the selected corner or area
function deleteKey() {
  if (DRAW_MODES.includes(state.mode)) {
    state.draft.pop();
    draw();
  } else {
    deleteSelection();
  }
}

function deleteSelection() {
  const s = state.selection;
  if (!s) return;
  edit(() => {
    const polygons = state.shapes[s.layer];
    if (s.corner !== undefined && polygons[s.index].length > 3) polygons[s.index].splice(s.corner, 1);
    else polygons.splice(s.index, 1); // the area, or a corner of a triangle: the whole area
    state.selection = null;
  });
}

// The dots clicked so far become an area: a new rope, or water merged in / cut out
function finishDraft() {
  if (!DRAW_MODES.includes(state.mode) || state.draft.length < 3) return;
  if (state.mode === "add-rope") {
    const rope = state.draft;
    edit(() => {
      state.shapes.ropes.push(rope);
      state.selection = { layer: "ropes", index: state.shapes.ropes.length - 1 };
    });
    setMode("edit");
  } else {
    finishWater();
  }
}

// + Water / - Water: the server merges the drawn area into the water or cuts it out, and simplifies the outlines
async function finishWater() {
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

// One rope from two (e.g. SAM3 found a rope as two pieces): the outline around both
function convexHull(points) {
  const sorted = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (list) => {
    const hull = [];
    for (const p of list) {
      while (hull.length >= 2 && cross(hull[hull.length - 2], hull[hull.length - 1], p) <= 0) hull.pop();
      hull.push(p);
    }
    hull.pop();
    return hull;
  };
  return [...half(sorted), ...half(sorted.reverse())];
}

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

const LAYERS = {
  water: { color: WATER, alpha: () => state.opacity, width: 2 },
  ropes: { color: ROPE, alpha: () => Math.min(state.opacity + 0.25, 0.9), width: 1 },
};

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
  const px = 1 / state.view.scale; // one screen pixel, in image pixels
  if (state.show) {
    for (const [layer, look] of Object.entries(LAYERS)) {
      state.shapes[layer].forEach((polygon, i) => {
        const selected = state.selection?.layer === layer && state.selection.index === i;
        polygonPath(polygon);
        ctx.fillStyle = `rgba(${look.color.fill}, ${look.alpha()})`;
        ctx.fill();
        ctx.strokeStyle = selected ? "#ffffff" : look.color.line;
        ctx.lineWidth = (selected ? look.width + 1 : look.width) * px;
        ctx.stroke();
      });
    }
  }
  drawDraft(px);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // handles: screen pixels, same size at any zoom
  if (state.show) drawHandles();
  drawDraftDots();
}

function polygonPath(polygon) {
  ctx.beginPath();
  polygon.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
}

// Corners of every area (water: squares, ropes: circles); the selected area also gets + on its edges
function drawHandles() {
  const sel = state.selection;
  for (const [layer, look] of Object.entries(LAYERS)) {
    const mark = layer === "ropes" ? circle : square;
    state.shapes[layer].forEach((polygon, i) => {
      for (const p of polygon) mark(toScreen(p), HANDLE - 1, look.color.line);
      if (sel?.layer === layer && sel.index === i) {
        if (sel.corner !== undefined) mark(toScreen(polygon[sel.corner]), HANDLE + 1, "#ffffff");
        polygon.forEach((p, k) => plus(toScreen(midpoint(p, polygon[(k + 1) % polygon.length]))));
      }
    });
  }
}

const draftColor = () => (state.mode === "add-rope" ? ROPE : state.mode === "cut-water" ? { fill: [255, 107, 107], line: "#ff6b6b" } : WATER);

// The area being added: the dots connected in order, filled from the third dot on (the closing edge dashed)
function drawDraft(px) {
  if (!DRAW_MODES.includes(state.mode) || state.draft.length < 2) return;
  const color = draftColor();
  if (state.draft.length >= 3) {
    polygonPath(state.draft);
    ctx.fillStyle = `rgba(${color.fill}, 0.4)`;
    ctx.fill();
  }
  ctx.strokeStyle = color.line;
  ctx.lineWidth = 2 * px;
  ctx.beginPath();
  state.draft.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.stroke();
  if (state.draft.length >= 3) {
    const [first, last] = [state.draft[0], state.draft[state.draft.length - 1]];
    ctx.setLineDash([6 * px, 4 * px]);
    ctx.beginPath();
    ctx.moveTo(...last);
    ctx.lineTo(...first);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

// The dots, in screen pixels; the first one larger once it can close the area
function drawDraftDots() {
  if (!DRAW_MODES.includes(state.mode)) return;
  state.draft.forEach((p, k) => {
    const closes = k === 0 && state.draft.length >= 3;
    circle(toScreen(p), closes ? CLOSE : HANDLE - 2, closes ? draftColor().line : "#ffffff");
  });
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

// The rope under image point `p` (index, or -1): inside its outline, or a few screen pixels from it (far ropes are thin)
function ropeAt(p) {
  const reach = HIT / 2 / state.view.scale;
  for (let i = state.shapes.ropes.length - 1; i >= 0; i--) {
    const polygon = state.shapes.ropes[i];
    if (insidePolygon(p, polygon) || polygon.some((q, k) => distanceToSegment(p, q, polygon[(k + 1) % polygon.length]) <= reach)) {
      return i;
    }
  }
  return -1;
}

// What a click at screen point `s` hits, most specific first: {kind: "corner" | "mid" | "area", layer, index, ...}
function hitTest(s) {
  for (const layer of ["ropes", "water"]) { // ropes are drawn on top
    const polygons = state.shapes[layer];
    for (let i = polygons.length - 1; i >= 0; i--) {
      const corner = polygons[i].findIndex((p) => dist(s, toScreen(p)) <= HIT);
      if (corner >= 0) return { kind: "corner", layer, index: i, corner };
    }
  }
  const sel = state.selection;
  if (sel) {
    const polygon = state.shapes[sel.layer][sel.index];
    for (let k = 0; k < polygon.length; k++) {
      if (dist(s, toScreen(midpoint(polygon[k], polygon[(k + 1) % polygon.length]))) <= HIT) {
        return { kind: "mid", layer: sel.layer, index: sel.index, edge: k };
      }
    }
  }
  const p = toImage(s);
  const rope = ropeAt(p);
  if (rope >= 0) return { kind: "area", layer: "ropes", index: rope };
  const water = state.shapes.water.findLastIndex((polygon) => insidePolygon(p, polygon));
  return water >= 0 ? { kind: "area", layer: "water", index: water } : null;
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
    state.mouse = s;
    if (state.mode !== "edit") {
      // Adding: a click is a dot (or, joining, the rope to join); a drag moves the frame
      state.drag = { type: "pan", start: s, last: s, moved: false, click: e.button === 0 };
      return;
    }
    const hit = e.button === 0 && state.show ? hitTest(s) : null; // hidden labels can't be edited
    const before = shapesJSON(state.shapes);
    if (hit?.kind === "corner") {
      state.selection = { layer: hit.layer, index: hit.index, corner: hit.corner };
      state.drag = { type: "corner", hit, before, moved: false };
    } else if (hit?.kind === "mid") {
      const polygon = state.shapes[hit.layer][hit.index];
      polygon.splice(hit.edge + 1, 0, midpoint(polygon[hit.edge], polygon[(hit.edge + 1) % polygon.length]));
      state.selection = { layer: hit.layer, index: hit.index, corner: hit.edge + 1 };
      state.drag = { type: "corner", hit: { ...state.selection, kind: "corner" }, before, moved: true };
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
      if (state.mode === "edit") canvas.style.cursor = hoverCursor(state.show ? hitTest(s) : null);
      showClassTip(s);
      return;
    }
    hideClassTip();
    if (d.type === "pan") {
      if (dist(s, d.start) > 3 && !d.moved) {
        d.moved = true;
        canvas.style.cursor = "grabbing";
      }
      state.view.x += s[0] - d.last[0];
      state.view.y += s[1] - d.last[1];
      d.last = s;
    } else {
      d.moved = true;
      state.shapes[d.hit.layer][d.hit.index][d.hit.corner] = clampToImage(toImage(s));
      updatePanel();
    }
    draw();
  });
  const end = () => {
    const d = state.drag;
    state.drag = null;
    if (!d) return;
    canvas.style.cursor = state.mode === "edit" ? "grab" : "crosshair";
    if (d.type === "pan" && state.mode !== "edit") {
      if (!d.moved && d.click) clickInMode(d.start);
      return;
    }
    if (d.type === "pan") {
      if (!d.moved) state.selection = d.hit?.kind === "area" ? { layer: d.hit.layer, index: d.hit.index } : null;
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
  canvas.addEventListener("dblclick", finishDraft);
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    zoomAt(point(e), Math.exp(-e.deltaY * 0.0015));
    showClassTip(point(e)); // the shape under the pointer changes with the zoom
  }, { passive: false });
  canvas.addEventListener("pointerleave", () => {
    state.mouse = null;
    hideClassTip();
  });
  new ResizeObserver(() => draw()).observe(canvas);
}

// ---------------------------------------------------------------------------------------------------------------
// Class tooltip: the class under the pointer, as the training mask will have it (rope over water over other)

const CLASS_NAMES = { ropes: "Lane rope", water: "Water", other: "Other" };
const HANDLE_HINTS = { corner: "drag to move this corner", mid: "drag to add a corner here" };

// The class at screen point `s`: {layer: "ropes" | "water" | "other", index}, or null outside the frame
function classAt(s) {
  const p = toImage(s);
  if (p[0] < 0 || p[1] < 0 || p[0] > state.frame.width || p[1] > state.frame.height) return null;
  const rope = ropeAt(p);
  if (rope >= 0) return { layer: "ropes", index: rope };
  const water = state.shapes.water.findIndex((polygon) => insidePolygon(p, polygon));
  return water >= 0 ? { layer: "water", index: water } : { layer: "other" };
}

function showClassTip(s) {
  const tip = $("class-tip");
  const found = state.frame && state.show && state.mode === "edit" ? classAt(s) : null;
  if (!found) {
    hideClassTip();
    return;
  }
  // Over a handle, the class of the area the handle belongs to
  const hit = hitTest(s);
  const handle = hit && HANDLE_HINTS[hit.kind] ? hit : null;
  const { layer, index } = handle ?? found;
  const name = layer === "ropes" ? `${CLASS_NAMES.ropes} ${index + 1}` : CLASS_NAMES[layer];
  tip.replaceChildren(el("strong", "", name));
  if (handle) tip.append(el("span", "", ` · ${HANDLE_HINTS[handle.kind]}`));
  tip.className = `class-tip ${layer === "ropes" ? "rope" : layer}`;
  tip.hidden = false;
  // Next to the pointer, kept inside the editor
  const box = canvas.getBoundingClientRect();
  const x = Math.min(s[0] + 16, box.width - tip.offsetWidth - 4);
  const y = s[1] + 20 + tip.offsetHeight > box.height ? s[1] - tip.offsetHeight - 10 : s[1] + 20;
  tip.style.transform = `translate(${Math.max(x, 4)}px, ${y}px)`;
}

function hideClassTip() {
  $("class-tip").hidden = true;
}

function hoverCursor(hit) {
  if (!hit) return "grab";
  return hit.kind === "area" ? "pointer" : "move";
}

function clickInMode(s) {
  if (state.mode === "join") {
    const hit = hitTest(s);
    const sel = state.selection;
    const other = hit?.layer === "ropes" ? hit.index : null;
    if (sel?.layer === "ropes" && other !== null && other !== sel.index) {
      edit(() => {
        const joined = convexHull([...state.shapes.ropes[sel.index], ...state.shapes.ropes[other]]);
        state.shapes.ropes = state.shapes.ropes.filter((_, k) => k !== sel.index && k !== other);
        state.shapes.ropes.push(joined);
        state.selection = { layer: "ropes", index: state.shapes.ropes.length - 1 };
      });
      setMode("edit");
    }
    return;
  }
  // Drawing: each click is a dot; a click on the first dot closes the area (on the last dot: nothing, e.g. a
  // double-click's second click). Small distances, so the dots of a thin rope can be close together.
  const within = (p, pixels) => dist(toScreen(p), s) <= pixels;
  if (state.draft.length >= 3 && within(state.draft[0], CLOSE)) {
    finishDraft();
    return;
  }
  if (state.draft.length && within(state.draft[state.draft.length - 1], SAME_DOT)) return;
  state.draft.push(clampToImage(toImage(s)));
  draw();
}

// For the browser test: image point -> screen point, and the shapes being edited
window.__labels = { toScreen: (x, y) => toScreen([x, y]), shapes: () => state.shapes, frame: () => state.frame?.name };

main().catch((error) => showError(error.message));
