// Drawing layers of the browser viewer, one per stage output (see docs/app.md).
// Mirrors the Python renderer (swim/visualization/ and LAYERS in swim/visualization/render.py): same layer names,
// same colors, so the browser and the exported video.mp4 show the same thing. A new stage adds its layer in both.

// Same colors as swim/visualization/detections.py (there in BGR)
const KEPT = { color: "rgb(255, 0, 0)", text: "#ffffff" };
const REJECTED = { color: "rgb(170, 170, 170)", text: "#111111" };

// Track colors, chosen to stand out on blue pool water: track N gets TRACK_COLORS[(N - 1) % length].
// Same list as TRACK_COLORS in swim/visualization/tracks.py
const TRACK_COLORS = [
  { color: "rgb(255, 214, 10)", text: "#111111" }, // yellow
  { color: "rgb(255, 61, 242)", text: "#111111" }, // magenta
  { color: "rgb(255, 140, 0)", text: "#111111" }, // orange
  { color: "rgb(124, 255, 0)", text: "#111111" }, // lime
  { color: "rgb(255, 255, 255)", text: "#111111" }, // white
  { color: "rgb(184, 77, 255)", text: "#ffffff" }, // violet
  { color: "rgb(0, 230, 118)", text: "#111111" }, // mint
  { color: "rgb(30, 30, 30)", text: "#ffffff" }, // black
];

export const trackStyle = (trackId) => TRACK_COLORS[(trackId - 1) % TRACK_COLORS.length];

// Layer name -> the stage whose saved output it draws, and the function that draws one frame of it. Pipeline order.
export const LAYERS = {
  detections: { stage: "detection", description: "everything the detector found", draw: drawDetections },
  filtered: { stage: "filtering", description: "kept boxes; removed ones in gray with the reason", draw: drawDetections },
  tracks: { stage: "tracking", description: "each track in its own color with its number; untracked boxes in gray", draw: drawTracks },
};

/**
 * Draw every detection's box with label and score, rejected ones in gray with the reason.
 *
 * `ctx` is scaled to video pixels. Options: `px` = one screen pixel in video pixels (lines and labels keep the same
 * size on screen at any window size), `width` / `height` = frame size, `highlight` = {kind: "detections" |
 * "rejected", index} of a box to emphasize (hovered in the side panel).
 */
export function drawDetections(ctx, frame, { px, width, height, highlight }) {
  // Rejected first, so kept boxes are drawn on top; their label goes below the box, so it doesn't collide with the
  // label of a kept box it duplicates
  frame.rejected.forEach((r, i) => {
    const label = `${r.reason} ${r.detection.score.toFixed(2)}`;
    const emphasized = highlight?.kind === "rejected" && highlight.index === i;
    drawBox(ctx, r.detection.bbox, REJECTED, label, "below", { px, width, height, emphasized });
  });
  frame.detections.forEach((d, i) => {
    const label = `${d.label} ${d.score.toFixed(2)}`;
    const emphasized = highlight?.kind === "detections" && highlight.index === i;
    drawBox(ctx, d.bbox, KEPT, label, "above", { px, width, height, emphasized });
  });
}

/**
 * Draw every tracked box in its track's color with "#id score", boxes the tracker didn't use thin and gray.
 * Options as for drawDetections.
 */
export function drawTracks(ctx, frame, { px, width, height, highlight }) {
  // Untracked first, so tracks are drawn on top; their label goes below the box
  frame.rejected.forEach((r, i) => {
    const label = `${r.reason} ${r.detection.score.toFixed(2)}`;
    const emphasized = highlight?.kind === "rejected" && highlight.index === i;
    drawBox(ctx, r.detection.bbox, REJECTED, label, "below", { px, width, height, emphasized, thin: true });
  });
  frame.detections.forEach((d, i) => {
    const label = `#${d.track_id} ${d.score.toFixed(2)}`;
    const emphasized = highlight?.kind === "detections" && highlight.index === i;
    drawBox(ctx, d.bbox, trackStyle(d.track_id), label, "above", { px, width, height, emphasized });
  });
}

function drawBox(ctx, bbox, style, label, labelSide, { px, width, height, emphasized, thin = false }) {
  const w = bbox.x2 - bbox.x1;
  const h = bbox.y2 - bbox.y1;
  if (emphasized) {
    ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
    ctx.lineWidth = 6 * px;
    ctx.strokeRect(bbox.x1, bbox.y1, w, h);
  }
  ctx.strokeStyle = style.color;
  ctx.lineWidth = (emphasized ? 3 : thin ? 1 : 2) * px;
  ctx.strokeRect(bbox.x1, bbox.y1, w, h);

  // Label on a tag in the box color, above or below the box, kept inside the frame
  ctx.font = `600 ${12 * px}px system-ui, -apple-system, sans-serif`;
  const pad = 4 * px;
  const tagH = 18 * px;
  const tagW = ctx.measureText(label).width + 2 * pad;
  const x = Math.min(Math.max(bbox.x1 - px, 0), width - tagW);
  const y = Math.min(Math.max(labelSide === "above" ? bbox.y1 - tagH : bbox.y2, 0), height - tagH);
  ctx.fillStyle = style.color;
  ctx.fillRect(x, y, tagW, tagH);
  ctx.fillStyle = style.text;
  ctx.textBaseline = "middle";
  ctx.fillText(label, x + pad, y + tagH / 2);
}
