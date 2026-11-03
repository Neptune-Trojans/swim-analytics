// Page helpers shared by the Videos page (videos.js) and the Play page (viewer.js): elements, errors, JSON requests.

export const $ = (id) => document.getElementById(id);

export function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export function showError(message) {
  $("error").textContent = message;
  $("error").hidden = false;
}

export function hideError() {
  $("error").hidden = true;
}

// GET (no body) or POST (body sent as JSON); the server's error message ("detail") becomes the Error's message
export async function requestJSON(url, body) {
  const options = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(typeof data.detail === "string" ? data.detail : `${response.status} ${response.statusText}: ${url}`);
  }
  return data;
}
