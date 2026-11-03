// Videos page of the browser app (docs/app.md): the videos in a folder, whether they have results and whether those
// are up to date with the settings chosen for processing. Processing runs on the server in the background, one video
// at a time (POST /api/jobs); this page shows its progress and opens processed videos in the Play page.

import { $, el, hideError, requestJSON, showError } from "./dom.js";
import { changedSettings, markSettings, setSettingValue, settingRow } from "./settings.js";

const STORAGE = { folder: "swim.videos.folder", settings: "swim.videos.settings" };

const state = {
  listing: null, // GET /api/library: folder, parent, subfolders, videos with their results state, stages
  jobs: [], // GET /api/jobs: every job of the server's session, oldest first
  changes: {}, // settings for processing that differ from the defaults: {stage: {name: value}}
  selected: new Set(), // paths of the ticked videos
  rows: new Map(), // video path -> elements of its row that change with the jobs
  polling: null,
};

// Remembered in this browser only (last folder, settings); the page works without it (private window)
const store = {
  get(key) {
    try { return localStorage.getItem(key); } catch { return null; }
  },
  set(key, value) {
    try {
      if (value === null) localStorage.removeItem(key);
      else localStorage.setItem(key, value);
    } catch { /* storage blocked */ }
  },
};

const isActive = (job) => job.state === "queued" || job.state === "running";
const latestJob = (path) => state.jobs.findLast((job) => job.video === path) ?? null;

// ---------------------------------------------------------------------------------------------------------------
// Start

async function main() {
  buildSettings(await requestJSON("/api/settings"));
  setupFolderBar();
  setupToolbar();
  window.addEventListener("popstate", () => loadFolder(new URLSearchParams(location.search).get("folder"), "none"));

  const fromUrl = new URLSearchParams(location.search).get("folder");
  const remembered = store.get(STORAGE.folder);
  try {
    await loadFolder(fromUrl ?? remembered, "replace");
  } catch (error) {
    if (!fromUrl && !remembered) throw error;
    // The remembered folder was moved or deleted: open the default one, keep the message
    await loadFolder(null, "replace");
    showError(error.message);
  }
  await pollJobs();
}

// Load a folder (null = the server's input folder). `history`: "push" (the user moved to another folder: Back
// returns), "replace" (first load), "none" (Back / Forward, or a refresh)
async function loadFolder(folder, history = "none") {
  const query = new URLSearchParams();
  if (folder) query.set("folder", folder);
  if (Object.keys(state.changes).length) query.set("settings", JSON.stringify(state.changes));
  const listing = await requestJSON(`/api/library?${query}`);
  const moved = listing.folder !== state.listing?.folder;
  state.listing = listing;
  store.set(STORAGE.folder, listing.folder);
  const url = `?folder=${encodeURIComponent(listing.folder)}`;
  if (history === "push" && moved) window.history.pushState(null, "", url);
  if (history === "replace") window.history.replaceState(null, "", url);
  if (moved) state.selected.clear();
  renderFolder();
  renderVideos();
}

// Errors of actions the user started (a folder that doesn't exist, a job the server refused) go to the banner
function attempt(promise) {
  promise.then(hideError, (error) => showError(error.message));
}

// ---------------------------------------------------------------------------------------------------------------
// Folder bar: path field, parent folder, subfolders

function setupFolderBar() {
  $("folder-form").addEventListener("submit", (event) => {
    event.preventDefault();
    attempt(loadFolder($("folder-path").value.trim(), "push"));
  });
  $("folder-up").addEventListener("click", () => attempt(loadFolder(state.listing.parent, "push")));
}

function renderFolder() {
  const { folder, parent, folders, output } = state.listing;
  $("folder-path").value = folder;
  $("folder-up").disabled = !parent;
  $("output-dir").textContent = output;
  const chips = folders.map((f) => {
    const button = el("button", "folder-chip", f.name);
    button.type = "button";
    button.addEventListener("click", () => attempt(loadFolder(f.path, "push")));
    return button;
  });
  $("subfolders").replaceChildren(...chips);
  $("subfolders").hidden = !chips.length;
}

// ---------------------------------------------------------------------------------------------------------------
// Settings for processing: every stage's settings, starting from the defaults (changes remembered in this browser)

function buildSettings(defaults) {
  const groups = Object.entries(defaults).map(([stage, fields]) => {
    const group = el("fieldset", "settings-group");
    group.append(el("legend", "", stage), ...fields.map((f) => settingRow(stage, f, onSettingsChange)));
    return group;
  });
  $("settings").replaceChildren(...groups);

  let saved = {};
  try {
    saved = JSON.parse(store.get(STORAGE.settings) ?? "{}") ?? {};
  } catch { /* not JSON: ignore */ }
  for (const [stage, values] of Object.entries(saved)) {
    for (const [name, value] of Object.entries(values ?? {})) setSettingValue($("settings"), stage, name, value);
  }
  const { valid } = markSettings($("settings"));
  if (!valid) {
    // A remembered value no longer allowed (the range changed): start from the defaults
    store.set(STORAGE.settings, null);
    buildSettings(defaults);
    return;
  }
  state.changes = changedSettings($("settings"));
  if (Object.keys(state.changes).length) $("process-settings").open = true;
  showSettingsSummary();

  $("reset-settings").onclick = () => {
    store.set(STORAGE.settings, null);
    buildSettings(defaults);
    attempt(loadFolder(state.listing?.folder));
  };
}

let reloadTimer = null;

function onSettingsChange() {
  const { valid } = markSettings($("settings"));
  if (!valid) return; // keep the last valid settings until the field is fixed
  state.changes = changedSettings($("settings"));
  store.set(STORAGE.settings, Object.keys(state.changes).length ? JSON.stringify(state.changes) : null);
  showSettingsSummary();
  // The videos' state depends on the settings: ask again once the typing stops
  clearTimeout(reloadTimer);
  reloadTimer = setTimeout(() => attempt(loadFolder(state.listing?.folder)), 300);
}

function showSettingsSummary() {
  const changed = Object.entries(state.changes).flatMap(([stage, values]) =>
    Object.entries(values).map(([name, value]) => `${stage}.${name} ${value}`));
  $("settings-summary").textContent = changed.length ? `· changed: ${changed.join(", ")}` : "· defaults";
  $("reset-settings").hidden = !changed.length;
}

// ---------------------------------------------------------------------------------------------------------------
// Videos: one row each. The row is built when the folder is loaded; its job, buttons and checkbox change with the
// jobs (updateRows).

function renderVideos() {
  state.rows.clear();
  const rows = state.listing.videos.map(videoRow);
  $("videos").replaceChildren(...(rows.length ? rows : [el("li", "videos-empty", "No videos in this folder (.mov, .mp4, .m4v).")]));
  updateRows();
}

function videoRow(v) {
  const li = el("li", "video-row");
  const checkbox = el("input");
  checkbox.type = "checkbox";
  checkbox.setAttribute("aria-label", `Select ${v.file}`);
  checkbox.addEventListener("change", () => {
    if (checkbox.checked) state.selected.add(v.path);
    else state.selected.delete(v.path);
    updateRows();
  });

  const thumbBox = el("div", "thumb-box");
  if (v.info) {
    const img = el("img", "thumb");
    img.loading = "lazy";
    img.alt = "";
    img.src = `/api/thumbnail?path=${encodeURIComponent(v.path)}&v=${v.modified}`;
    img.addEventListener("error", () => img.remove());
    thumbBox.append(img);
  }

  const title = el("div", "video-title");
  const name = el("a", "video-file", v.file);
  const status = describeResults(v);
  title.append(name, el("span", `status ${status.kind}`, status.text));

  const main = el("div", "video-main");
  main.append(title, el("div", "video-meta", videoMeta(v)));
  if (v.info && v.results.stages.length && !v.results.made_from) main.append(stageChips(v));
  if (status.detail) main.append(el("div", `video-detail ${status.kind}`, status.detail));
  const job = el("div", "job");
  main.append(job);

  const open = el("a", "btn-secondary", "Open");
  open.href = `/play?video=${encodeURIComponent(v.results.name)}`;
  const action = el("button", "btn-primary", actionLabel(v));
  action.type = "button";
  action.title = v.info ? `Runs ${state.listing.stages.slice(state.listing.stages.indexOf(v.results.start ?? state.listing.stages[0])).join(" → ")}` : "";
  action.addEventListener("click", () => processVideos([v]));
  const actions = el("div", "video-actions");
  actions.append(open, action);

  li.append(checkbox, thumbBox, main, actions);
  state.rows.set(v.path, { li, checkbox, name, job, open, action });
  return li;
}

function videoMeta(v) {
  const parts = [];
  if (v.info) {
    const seconds = v.info.frame_count / v.info.fps;
    parts.push(`${Math.floor(seconds / 60)}:${String(Math.round(seconds % 60)).padStart(2, "0")}`,
      `${v.info.frame_count} frames`, `${v.info.width}×${v.info.height}`, `${Math.round(v.info.fps)} fps`);
  }
  parts.push(v.size >= 1e9 ? `${(v.size / 1e9).toFixed(1)} GB` : `${Math.round(v.size / 1e6)} MB`,
    new Date(v.modified * 1000).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" }));
  return parts.join(" · ");
}

// The state of the video's results, compared with the settings for processing: a label and what is missing
function describeResults(v) {
  const r = v.results;
  if (!v.info) return { kind: "bad", text: "Can't read this video" };
  if (!r.stages.length) return { kind: "none", text: "Not processed" };
  if (r.made_from) {
    return { kind: "warn", text: "Results of another video",
      detail: `${r.name}'s results were made from ${r.made_from}. Processing this video replaces them.` };
  }
  if (r.start === null) return { kind: "ok", text: "Up to date" };
  const reasons = [];
  if (!r.settings_saved) reasons.push("settings not recorded (no config.json)");
  for (const d of r.differences) reasons.push(`${d.stage}.${d.name} ${d.saved} → ${d.wanted}`);
  for (const stage of state.listing.stages) if (!r.stages.includes(stage)) reasons.push(`${stage} not run`);
  return { kind: "warn", text: "Needs update", detail: reasons.join(" · ") };
}

// One chip per stage: saved (✓), saved with other settings (≠), not run (–)
function stageChips(v) {
  const r = v.results;
  const changed = new Set(r.differences.map((d) => d.stage));
  const chips = el("div", "stage-chips");
  for (const stage of state.listing.stages) {
    const kind = !r.stages.includes(stage) ? "missing" : changed.has(stage) || !r.settings_saved ? "changed" : "saved";
    const chip = el("span", `chip ${kind}`, stage);
    chip.title = { saved: "saved", changed: "saved, with other settings", missing: "not run" }[kind];
    chips.append(chip);
  }
  return chips;
}

function actionLabel(v) {
  const r = v.results;
  if (!r.stages.length || r.made_from) return "Process";
  return r.start === null ? "Re-process" : "Update";
}

// What changes with the jobs and the selection: each row's job line, buttons and checkbox, and the toolbar
function updateRows() {
  if (!state.listing) return;
  for (const v of state.listing.videos) {
    const row = state.rows.get(v.path);
    const job = latestJob(v.path);
    const active = job && isActive(job);
    const openable = v.results.stages.length > 0 && !v.results.made_from && job?.state !== "running";
    row.checkbox.disabled = active || !v.info;
    row.checkbox.checked = state.selected.has(v.path) && !row.checkbox.disabled;
    row.li.classList.toggle("selected", row.checkbox.checked);
    row.li.classList.toggle("active", Boolean(active));
    row.open.hidden = !openable;
    row.action.hidden = Boolean(active);
    row.action.disabled = !v.info;
    if (openable) row.name.href = row.open.href;
    else row.name.removeAttribute("href");
    renderJob(row.job, job);
  }
  updateToolbar();
}

function renderJob(box, job) {
  box.className = "job";
  if (!job) {
    box.replaceChildren();
    return;
  }
  const cancel = el("button", "btn-text", "Cancel");
  cancel.type = "button";
  cancel.addEventListener("click", () => attempt(cancelJob(job.id)));
  if (job.state === "queued") {
    box.replaceChildren(el("span", "", `Waiting · ${ordinal(job.position)} in line`), cancel);
  } else if (job.state === "running") {
    const bar = el("progress");
    bar.max = Math.max(job.total, 1);
    bar.value = job.stage === "detection" ? Math.min(job.done, job.total) : bar.max;
    const text = job.cancel_requested ? "Cancelling…"
      : job.stage === "detection" && job.done > 0
        ? `detection · ${job.done} / ${job.total} frames${job.seconds_left !== null ? ` · ${duration(job.seconds_left)} left` : ""}`
        : `${job.stage ?? "starting"}…`;
    box.classList.add("running");
    box.replaceChildren(bar, el("span", "", text), job.cancel_requested ? "" : cancel);
  } else if (job.state === "done") {
    box.classList.add("done");
    box.replaceChildren(el("span", "", `Processed in ${duration(job.seconds)}`));
  } else if (job.state === "failed") {
    box.classList.add("failed");
    box.replaceChildren(el("span", "", `Processing failed: ${job.error}`));
  } else {
    box.replaceChildren(el("span", "", "Cancelled"));
  }
}

const ordinal = (n) => `${n}${n % 10 === 1 && n !== 11 ? "st" : n % 10 === 2 && n !== 12 ? "nd" : n % 10 === 3 && n !== 13 ? "rd" : "th"}`;

function duration(seconds) {
  if (seconds < 60) return `${Math.max(Math.round(seconds), 1)} s`;
  return `${Math.floor(seconds / 60)} min ${String(Math.round(seconds % 60)).padStart(2, "0")} s`;
}

// ---------------------------------------------------------------------------------------------------------------
// Toolbar: select all, process the selected videos or all that need it

// Videos that can be queued: readable, not already waiting or running
const queueable = () => state.listing.videos.filter((v) => v.info && !(latestJob(v.path) && isActive(latestJob(v.path))));
// ... of which those whose results are missing or not up to date (another video's results need an explicit choice)
const needing = () => queueable().filter((v) => v.results.start !== null && !v.results.made_from);
const selectedVideos = () => queueable().filter((v) => state.selected.has(v.path));

function setupToolbar() {
  $("select-all").addEventListener("change", (event) => {
    for (const v of queueable()) {
      if (event.target.checked) state.selected.add(v.path);
      else state.selected.delete(v.path);
    }
    updateRows();
  });
  $("process-selected").addEventListener("click", () => processVideos(selectedVideos()));
  $("process-needed").addEventListener("click", () => processVideos(needing()));
}

function updateToolbar() {
  const videos = state.listing.videos;
  const processed = videos.filter((v) => v.results.stages.length && !v.results.made_from).length;
  const upToDate = videos.filter((v) => v.results.stages.length && v.results.start === null).length;
  $("video-count").textContent = videos.length
    ? `${videos.length} ${videos.length === 1 ? "video" : "videos"} · ${processed} with results · ${upToDate} up to date`
    : "No videos";
  const selectable = queueable();
  const selected = selectedVideos().length;
  $("select-all").disabled = !selectable.length;
  $("select-all").checked = selectable.length > 0 && selected === selectable.length;
  $("select-all").indeterminate = selected > 0 && selected < selectable.length;
  $("process-selected").disabled = !selected;
  $("process-selected").textContent = selected ? `Process selected (${selected})` : "Process selected";
  const needed = needing().length;
  $("process-needed").disabled = !needed;
  $("process-needed").textContent = needed ? `Process all that need it (${needed})` : "All up to date";
}

// ---------------------------------------------------------------------------------------------------------------
// Jobs: queue, cancel, follow the progress

async function processVideos(videos) {
  if (!videos.length) return;
  const others = videos.filter((v) => v.results.made_from);
  if (others.length) {
    const list = others.map((v) => `• ${v.file}: the results in ${v.results.name}/ were made from ${v.results.made_from}`).join("\n");
    if (!window.confirm(`Results with the same name already exist for another video:\n\n${list}\n\nReplace them?`)) return;
  }
  try {
    state.jobs = await requestJSON("/api/jobs", {
      videos: videos.map((v) => v.path), settings: state.changes, replace: others.length > 0,
    });
    hideError();
  } catch (error) {
    showError(error.message);
    return;
  }
  for (const v of videos) state.selected.delete(v.path);
  updateRows();
  showJobsSummary();
  startPolling();
}

async function cancelJob(id) {
  state.jobs = await requestJSON(`/api/jobs/${id}/cancel`, {});
  updateRows();
}

function startPolling() {
  state.polling ??= setInterval(pollJobs, 1000);
}

let connectionLost = false;

async function pollJobs() {
  const before = new Set(state.jobs.filter(isActive).map((j) => j.id));
  try {
    state.jobs = await requestJSON("/api/jobs");
    if (connectionLost) hideError();
    connectionLost = false;
  } catch (error) {
    connectionLost = true;
    showError(`Can't reach the server (${error.message}). Is python -m swim.app still running?`);
    return;
  }
  const active = new Set(state.jobs.filter(isActive).map((j) => j.id));
  if (active.size) startPolling();
  else if (state.polling) {
    clearInterval(state.polling);
    state.polling = null;
  }
  // A job ended: its video's results changed, ask for the folder's state again
  if ([...before].some((id) => !active.has(id)) && state.listing) await loadFolder(state.listing.folder).catch((e) => showError(e.message));
  else updateRows();
  showJobsSummary();
}

// In the top bar, wherever the videos are: what is being processed and how many are waiting
function showJobsSummary() {
  const running = state.jobs.find((j) => j.state === "running");
  const waiting = state.jobs.filter((j) => j.state === "queued").length;
  const parts = [];
  if (running) {
    const pct = running.stage === "detection" && running.total ? ` ${Math.floor((100 * running.done) / running.total)}%` : "";
    parts.push(`Processing ${running.name}: ${running.stage ?? "starting"}${pct}`);
  }
  if (waiting) parts.push(`${waiting} waiting`);
  $("jobs-summary").textContent = parts.join(" · ");
}

main().catch((error) => showError(error.message));
