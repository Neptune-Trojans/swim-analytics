// Settings form rows, shared by the Play page (change the fast stages' settings and re-run them) and the Videos page
// (settings for processing). Each setting is described as {name, value, type, help, min, max, step, choices}
// (describe() in swim/pipeline/config.py).
//
// An editable row's input carries data-stage, data-name, data-type and data-saved: the value it is compared with to
// mark the row "changed" (the saved setting on the Play page, the default on the Videos page).

import { el } from "./dom.js";

// A row for one setting: a number (slider + field), a choice (select) or, for other text, a fixed value
export function settingRow(stage, f, onChange, saved = f.value) {
  if (f.choices) return choiceSetting(stage, f, onChange, saved);
  if (typeof f.value === "number") return numberSetting(stage, f, onChange, saved);
  return fixedSetting(f);
}

function numberSetting(stage, f, onChange, saved) {
  const id = `setting-${stage}-${f.name}`;
  const range = el("input");
  range.type = "range";
  range.tabIndex = -1; // the number field next to it is the one for the keyboard
  const number = el("input");
  number.type = "number";
  number.id = id;
  number.required = true;
  for (const input of [range, number]) {
    if (f.min !== null) input.min = f.min;
    if (f.max !== null) input.max = f.max;
    input.step = f.step ?? "any";
    input.value = f.value;
  }
  Object.assign(number.dataset, { stage, name: f.name, saved, type: f.type });
  range.addEventListener("input", () => {
    number.value = range.value;
    onChange();
  });
  number.addEventListener("input", () => {
    range.value = number.value;
    onChange();
  });

  const label = el("label", "setting-name", f.name);
  label.htmlFor = id;
  const controls = el("div", "setting-controls");
  controls.append(range, number);
  const row = el("div", "setting");
  row.append(label, controls, el("p", "setting-help", f.help));
  return row;
}

function choiceSetting(stage, f, onChange, saved) {
  const id = `setting-${stage}-${f.name}`;
  const select = el("select");
  select.id = id;
  select.replaceChildren(...f.choices.map((choice) => new Option(choice, choice, false, choice === f.value)));
  Object.assign(select.dataset, { stage, name: f.name, saved, type: f.type });
  select.addEventListener("change", onChange);

  const head = el("div", "setting-head");
  const label = el("label", "setting-name", f.name);
  label.htmlFor = id;
  head.append(label, select);
  const row = el("div", "setting");
  row.append(head, el("p", "setting-help", f.help));
  return row;
}

export function fixedSetting(f) {
  const head = el("div", "setting-head");
  head.append(el("span", "setting-name", f.name), el("span", "setting-value", String(f.value)));
  const row = el("div", "setting");
  row.append(head, el("p", "setting-help", f.help));
  return row;
}

// The editable inputs under `root`
export const settingInputs = (root) => [...root.querySelectorAll("[data-name]")];

const parseSetting = (type, text) => (type === "str" ? text : Number(text));
export const settingValue = (input) => parseSetting(input.dataset.type, input.value);
export const isChanged = (input) =>
  input.value !== "" && settingValue(input) !== parseSetting(input.dataset.type, input.dataset.saved);

// A value is usable if present and in range; any decimal is fine (not only multiples of the slider step), except for
// whole-number settings
export function usable(input) {
  const v = input.validity;
  if (v.valueMissing || v.badInput || v.rangeUnderflow || v.rangeOverflow) return false;
  return input.dataset.type !== "int" || Number.isInteger(Number(input.value));
}

// {stage: {name: value}} of the changed inputs under `root`
export function changedSettings(root) {
  const changes = {};
  for (const input of settingInputs(root)) {
    if (isChanged(input)) (changes[input.dataset.stage] ??= {})[input.dataset.name] = settingValue(input);
  }
  return changes;
}

// Mark changed and invalid rows; returns whether anything changed and whether every value is usable
export function markSettings(root) {
  const inputs = settingInputs(root);
  for (const input of inputs) {
    const row = input.closest(".setting");
    row.classList.toggle("changed", isChanged(input));
    row.classList.toggle("invalid", !usable(input));
  }
  return { changed: inputs.some(isChanged), valid: inputs.every(usable) };
}

// Set an input (and its slider) from code; false if there is no such setting
export function setSettingValue(root, stage, name, value) {
  const input = root.querySelector(`[data-stage="${stage}"][data-name="${name}"]`);
  if (!input) return false;
  input.value = value;
  const range = input.closest(".setting").querySelector('input[type="range"]');
  if (range) range.value = value;
  return true;
}
