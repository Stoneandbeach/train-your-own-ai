// ---- Config (mirrors server/config.py) ----
// MODEL_GRID_SIZE is the network's own fixed input resolution - what
// training, checkpoints, saliency maps, and the connection-line input count
// all use, regardless of mode. Digits draws natively at this resolution;
// Drawings draws DRAW_DOWNSCALE_FACTOR times finer and gets block-downscaled
// by fill-count on the server before it ever reaches the model - see
// server/preprocessing.py's downscale_by_fill_count and server/main.py's
// draw_update handler. `drawGridSize` (under State, below) tracks whichever
// of DRAW_GRID_SIZE_DIGITS/DRAWINGS applies to the current mode.
const MODEL_GRID_SIZE = 28;
const DRAW_DOWNSCALE_FACTOR = 2;
const DRAW_GRID_SIZE_DIGITS = MODEL_GRID_SIZE;
const DRAW_GRID_SIZE_DRAWINGS = MODEL_GRID_SIZE * DRAW_DOWNSCALE_FACTOR;
const MIN_NODES_PER_LAYER = 10;
// Drawings mode allows twice the hidden-layer capacity of Digits. The
// per-node pixel sizes below (PX_PER_NODE_*/NODE_RADIUS_GAP_*) scale down by
// the same ratio the node cap scales up (NODE_SIZE_SCALE_DRAWINGS), so
// maxNodesPerLayer * pxPerNode - the config panel's height budget, see
// applyNodeLayoutForMode() below - stays identical across modes: more nodes,
// same panel height, just packed tighter (node circles keep the same
// on-screen proportions, just smaller).
const MAX_NODES_PER_LAYER_DIGITS = 40;
const MAX_NODES_PER_LAYER_DRAWINGS = 80;
const NODE_STEP = 1;
const MIN_LAYERS = 0;
const MAX_LAYERS = 4;
const DEFAULT_LAYER_WIDTHS = [10];
const DRAW_SEND_INTERVAL_MS = 80;
const PX_PER_NODE_DIGITS = 18; // vertical pixels per node circle, for the resizable layer blocks (1.2x the original 15)
const NODE_RADIUS_GAP_DIGITS = 3; // gap subtracted from PX_PER_NODE/2 to get circle radius (1.2x the original 2.5)
const NODE_SIZE_SCALE_DRAWINGS = MAX_NODES_PER_LAYER_DIGITS / MAX_NODES_PER_LAYER_DRAWINGS;
const PX_PER_NODE_DRAWINGS = PX_PER_NODE_DIGITS * NODE_SIZE_SCALE_DRAWINGS;
const NODE_RADIUS_GAP_DRAWINGS = NODE_RADIUS_GAP_DIGITS * NODE_SIZE_SCALE_DRAWINGS;
const LABEL_HEIGHT = 20; // space reserved above the node column for the count label
// The resize handle (.resize-handle in style.css) is 28px tall and straddles
// the block's bottom border via bottom:-14px, so it reaches 14px up into
// this clearance - below that it starts overlapping the bottom node circle,
// badly so in Drawings mode (9px/node - the icon would cover the node
// almost entirely). +2px so it's a clearance, not an exact touch.
const MIN_HANDLE_CLEARANCE = 16;
// Below this per-connection magnitude (as a fraction of the frame's max), skip
// drawing the line entirely rather than just fading it - the input layer can
// contribute up to MODEL_GRID_SIZE^2 * MAX_NODES_PER_LAYER_DRAWINGS lines in one transition,
// and most of those are negligible, so this keeps the canvas from turning to
// mush (and keeps redraws fast) without changing how the strong connections look.
const CONNECTION_MIN_MAGNITUDE = 0.03;

// ---- Localization ----
// TRANSLATIONS comes from i18n.js, loaded before this file - see that file
// for every UI string in both languages.
let currentLanguage = "sv"; // default in all cases - see setLanguage() below

function t(key, ...args) {
  const dict = TRANSLATIONS[currentLanguage] || TRANSLATIONS.en;
  const value = key in dict ? dict[key] : TRANSLATIONS.en[key];
  return typeof value === "function" ? value(...args) : value;
}

// ---- State ----
// numClasses/classLabels describe the current task (10 digits, or 8 Quick
// Draw category names) - set from the server's mode_selected message. Every
// loop that used to hardcode 10/"0"-"9" reads these instead. classLabelsSv
// is the same set of classes translated to Swedish (server/main.py's
// class_names_sv - a no-op passthrough for Digits, since digit labels don't
// need translating) - currentClassLabels() below picks whichever matches
// currentLanguage for display, everywhere classLabels used to be read
// directly.
let numClasses = 0;
let classLabels = [];
let classLabelsSv = [];

function currentClassLabels() {
  return currentLanguage === "sv" && classLabelsSv.length === classLabels.length ? classLabelsSv : classLabels;
}
let currentMode = null; // "digits" | "drawings" | null (menu showing, no mode chosen yet)
let drawGridSize = DRAW_GRID_SIZE_DIGITS; // resolution of `pixels` below - set per mode by updateDrawResolution()
let pixels = new Array(drawGridSize * drawGridSize).fill(0);
let layerWidths = [...DEFAULT_LAYER_WIDTHS];
// Node cap/sizing for the config panel - set per mode by applyNodeLayoutForMode().
let maxNodesPerLayer = MAX_NODES_PER_LAYER_DIGITS;
let pxPerNode = PX_PER_NODE_DIGITS;
let nodeRadiusGap = NODE_RADIUS_GAP_DIGITS;
// Blank space below the last node circle, so the resize handle (see
// .resize-handle in style.css) doesn't overlap it - a full node height
// (tracks pxPerNode rather than being its own fixed constant), floored at
// MIN_HANDLE_CLEARANCE for modes where a node height alone isn't enough.
let handleClearance = Math.max(PX_PER_NODE_DIGITS, MIN_HANDLE_CLEARANCE);
let lastSendTime = 0;
let sendPending = false;
let trainLossHistory = []; // training loss, one point per pushed metrics message
let valLossPoints = []; // {x: index into trainLossHistory, y: val_loss}, one per completed epoch
let accuracyHistory = []; // validation accuracy (0-1), one point per completed epoch
let currentActivations = []; // normalized (0-1) per-layer node activations from the latest classification
let layerCanvases = []; // canvas element per layer, index-aligned with layerWidths
let rawActivations = []; // un-normalized per-layer node activations, for connection-line strength
let edgeWeights = []; // one matrix [dst][src] per layer transition (hidden->hidden, then last hidden->output)

// ---- WebSocket ----
const ws = new WebSocket(`ws://${location.host}/ws`);

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  switch (msg.type) {
    case "classification":
      renderClassification(msg.probs, msg.predicted);
      currentActivations = msg.activations || [];
      updateLayerVisuals();
      rawActivations = msg.raw_activations || [];
      drawConnections();
      break;
    case "training_metrics":
      handleTrainingMetrics(msg);
      break;
    case "training_status":
      handleTrainingStatus(msg);
      break;
    case "checkpoint_loaded":
      edgeWeights = msg.edge_weights || [];
      drawConnections();
      break;
    case "debug_sample":
      handleDebugSample(msg);
      break;
    case "saliency":
      if (msg.class_idx === explainedClass) drawSaliencyOverlay(msg.map);
      break;
    case "node_saliency":
      if (selectedNode && msg.layer === selectedNode.layer && msg.node === selectedNode.node) {
        drawSaliencyOverlay(msg.map);
      }
      break;
    case "mode_selected":
      handleModeSelected(msg);
      break;
    case "kiosk_reset":
      // The server has confirmed the shared kiosk state is back to "no mode
      // chosen" (see server/main.py's reset_kiosk handler) - reload rather
      // than hand-resetting every piece of client state individually, so
      // this client (and any other connected one) lands on a genuinely
      // fresh page load, the menu, same as a brand new visitor would see.
      location.reload();
      break;
    default:
      break;
  }
};

function sendMessage(obj) {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

// ---- Draw pane ----
const drawCanvas = document.getElementById("draw-canvas");
const drawCtx = drawCanvas.getContext("2d");
// Cell size for painting - mode-dependent (drawGridSize), recomputed by
// updateDrawResolution() on every mode switch. Distinct from
// saliencyCellSize below, which always covers the canvas in
// MODEL_GRID_SIZE cells since saliency maps come straight from the model's
// own fixed resolution regardless of draw resolution.
let cellSize = drawCanvas.width / drawGridSize;
function updateDrawResolution(mode) {
  drawGridSize = mode === "drawings" ? DRAW_GRID_SIZE_DRAWINGS : DRAW_GRID_SIZE_DIGITS;
  cellSize = drawCanvas.width / drawGridSize;
}
let isDrawing = false;
let eraseMode = false;
let lastPointerPos = null; // {x, y} in client coords, for interpolating fast strokes

// ---- "What would help/hurt this?" saliency overlay ----
const saliencyCanvas = document.getElementById("saliency-canvas");
const saliencyCtx = saliencyCanvas.getContext("2d");
const saliencyCellSize = drawCanvas.width / MODEL_GRID_SIZE;
let explainedClass = null; // class index currently being explained, or null
let selectedNode = null; // {layer, node} of a hidden-layer node being explained, or null

function clearSaliencyOverlay() {
  saliencyCtx.clearRect(0, 0, saliencyCanvas.width, saliencyCanvas.height);
}

function updateExplainHighlight() {
  for (let d = 0; d < numClasses; d++) {
    const dot = document.getElementById(`node-dot-${d}`);
    if (!dot) continue;
    dot.classList.toggle("explained", d === explainedClass);
  }
}

function resetExplain() {
  explainedClass = null;
  clearSaliencyOverlay();
  updateExplainHighlight();
}

function toggleExplain(classIdx) {
  resetSelectedNode(); // only one explanation active at a time, sharing the overlay canvas
  if (explainedClass === classIdx) {
    resetExplain();
    return;
  }
  explainedClass = classIdx;
  sendMessage({ type: "explain_prediction", class_idx: classIdx });
  updateExplainHighlight();
}

function resetSelectedNode() {
  if (!selectedNode) return;
  selectedNode = null;
  clearSaliencyOverlay();
  updateLayerVisuals();
}

// Click a hidden-layer node to see which drawing areas drive its activation.
function toggleNodeSelect(layer, node) {
  resetExplain(); // only one explanation active at a time, sharing the overlay canvas
  if (selectedNode && selectedNode.layer === layer && selectedNode.node === node) {
    resetSelectedNode();
    return;
  }
  selectedNode = { layer, node };
  sendMessage({ type: "explain_node", layer, node });
  updateLayerVisuals();
}

// map: 28x28 array of raw (signed, un-normalized) importance scores from the
// server - per patch, score(fully inked) - score(fully blank), independent
// of whatever's currently drawn there. Normalized symmetrically around zero
// (like the connection lines): red = inking there would raise the score,
// blue = it would lower it, with 0 (no effect) as a neutral midpoint.
function drawSaliencyOverlay(map) {
  let maxAbs = 1e-9;
  for (const row of map) {
    for (const v of row) {
      if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
    }
  }
  clearSaliencyOverlay();
  for (let gy = 0; gy < MODEL_GRID_SIZE; gy++) {
    for (let gx = 0; gx < MODEL_GRID_SIZE; gx++) {
      const norm = clamp(0.5 + map[gy][gx] / (2 * maxAbs), 0, 1);
      const magnitude = Math.abs(norm - 0.5) * 2;
      const alpha = 0.15 + magnitude * 0.6;
      saliencyCtx.fillStyle = colorForActivationAlpha(norm, alpha);
      saliencyCtx.fillRect(gx * saliencyCellSize, gy * saliencyCellSize, saliencyCellSize, saliencyCellSize);
    }
  }
}

function paintCell(gx, gy, intensity, erase) {
  if (gx < 0 || gx >= drawGridSize || gy < 0 || gy >= drawGridSize) return;
  const idx = gy * drawGridSize + gx;
  pixels[idx] = erase ? 0 : Math.min(255, Math.max(pixels[idx], intensity));
}

function paintPointAtClient(clientX, clientY, erase) {
  const rect = drawCanvas.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width) * drawGridSize;
  const y = ((clientY - rect.top) / rect.height) * drawGridSize;
  const gx = Math.floor(x);
  const gy = Math.floor(y);

  for (let i = -1; i <= 1; i++) {
    paintCell(gx+i, gy, 255/(1+Math.abs(i)), erase);
    paintCell(gx, gy+i, 255/(1+Math.abs(i)), erase);
  }
  if (erase) {
    // Wider eraser footprint so it actually feels like erasing, not just
    // undrawing a single stroke's width.
    paintCell(gx - 1, gy - 1, 0, erase);
    paintCell(gx + 1, gy - 1, 0, erase);
    paintCell(gx - 1, gy + 1, 0, erase);
    paintCell(gx + 1, gy + 1, 0, erase);
  }
}

// Paints every grid cell along the segment from the last recorded pointer
// position to (clientX, clientY) rather than just the endpoint, so a fast
// stroke (fewer pointermove samples than grid cells crossed) doesn't leave
// gaps in the line.
function paintStrokeSegment(clientX, clientY, erase) {
  if (lastPointerPos) {
    const dx = clientX - lastPointerPos.x;
    const dy = clientY - lastPointerPos.y;
    const dist = Math.hypot(dx, dy);
    const steps = Math.max(1, Math.ceil(dist / (cellSize / 2)));
    for (let i = 1; i <= steps; i++) {
      const t = i / steps;
      paintPointAtClient(lastPointerPos.x + dx * t, lastPointerPos.y + dy * t, erase);
    }
  } else {
    paintPointAtClient(clientX, clientY, erase);
  }
  lastPointerPos = { x: clientX, y: clientY };
}

function redrawCanvas() {
  // Display is inverted (black ink on white) for visual comfort, but the
  // underlying pixels array keeps the original MNIST convention (0 = blank,
  // 255 = fully inked) unchanged, since that's what's sent to the server.
  drawCtx.fillStyle = "white";
  drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  for (let gy = 0; gy < drawGridSize; gy++) {
    for (let gx = 0; gx < drawGridSize; gx++) {
      const v = pixels[gy * drawGridSize + gx];
      if (v > 0) {
        const shade = 255 - v;
        drawCtx.fillStyle = `rgb(${shade},${shade},${shade})`;
        drawCtx.fillRect(gx * cellSize, gy * cellSize, cellSize, cellSize);
      }
    }
  }
}

// If a digit or node explanation is currently selected, it should track the
// drawing live rather than going stale - re-request it at the same throttled
// cadence as draw_update.
function refreshActiveExplanation() {
  if (explainedClass !== null) {
    sendMessage({ type: "explain_prediction", class_idx: explainedClass });
  } else if (selectedNode) {
    sendMessage({ type: "explain_node", layer: selectedNode.layer, node: selectedNode.node });
  }
}

function scheduleDrawSend() {
  const now = performance.now();
  if (now - lastSendTime >= DRAW_SEND_INTERVAL_MS) {
    lastSendTime = now;
    sendMessage({ type: "draw_update", pixels });
    refreshActiveExplanation();
  } else if (!sendPending) {
    sendPending = true;
    setTimeout(() => {
      sendPending = false;
      lastSendTime = performance.now();
      sendMessage({ type: "draw_update", pixels });
      refreshActiveExplanation();
    }, DRAW_SEND_INTERVAL_MS - (now - lastSendTime));
  }
}

drawCanvas.addEventListener("contextmenu", (e) => e.preventDefault());

drawCanvas.addEventListener("pointerdown", (e) => {
  isDrawing = true;
  eraseMode = e.button === 2;
  lastPointerPos = null;
  // A selected explanation persists across strokes now - it tracks the
  // drawing live via refreshActiveExplanation(), only cleared by deselecting
  // or pressing Reset.
  paintStrokeSegment(e.clientX, e.clientY, eraseMode);
  redrawCanvas();
  scheduleDrawSend();
});
drawCanvas.addEventListener("pointermove", (e) => {
  if (!isDrawing) return;
  // getCoalescedEvents exposes every raw sample the OS captured between
  // frames (there can be several on a fast stroke) instead of just the
  // latest one, so we fill gaps using real mouse positions, not guesses.
  const coalesced = e.getCoalescedEvents ? e.getCoalescedEvents() : [];
  const samples = coalesced.length ? coalesced : [e];
  for (const sample of samples) {
    paintStrokeSegment(sample.clientX, sample.clientY, eraseMode);
  }
  redrawCanvas();
  scheduleDrawSend();
});
window.addEventListener("pointerup", () => {
  isDrawing = false;
  lastPointerPos = null;
});

// Shared by the Reset button and a mode switch (server/main.py resets
// last_pixels to None in both cases) - blanks the canvas and everything
// downstream of it (classification, activations, connection lines,
// any active explanation), without touching training-progress state.
function resetDrawingState() {
  pixels = new Array(drawGridSize * drawGridSize).fill(0);
  redrawCanvas();
  renderClassification(new Array(numClasses).fill(0), null);
  currentActivations = [];
  updateLayerVisuals();
  rawActivations = [];
  drawConnections();
  resetExplain();
  resetSelectedNode();
}

document.getElementById("reset-btn").addEventListener("click", () => {
  resetDrawingState();
  sendMessage({ type: "draw_update", pixels });
});

redrawCanvas();

// ---- Config pane: vertical layer blocks, drag the bottom edge to resize ----
const layersContainerEl = document.getElementById("layers-container");

// maxNodesPerLayer * pxPerNode is invariant across modes by construction
// (see NODE_SIZE_SCALE_DRAWINGS above); the panel height below adds
// handleClearance on top of that, which - now that it tracks pxPerNode - is
// NOT quite invariant across modes (18px in Digits vs the 16px floor in
// Drawings, see MIN_HANDLE_CLEARANCE), so the panel is a couple px taller in
// Digits mode. Small enough not to bother decoupling it from the per-block
// clearance term below.
function applyNodeLayoutForMode(mode) {
  const isDrawings = mode === "drawings";
  maxNodesPerLayer = isDrawings ? MAX_NODES_PER_LAYER_DRAWINGS : MAX_NODES_PER_LAYER_DIGITS;
  pxPerNode = isDrawings ? PX_PER_NODE_DRAWINGS : PX_PER_NODE_DIGITS;
  nodeRadiusGap = isDrawings ? NODE_RADIUS_GAP_DRAWINGS : NODE_RADIUS_GAP_DIGITS;
  handleClearance = Math.max(pxPerNode, MIN_HANDLE_CLEARANCE);
  layersContainerEl.style.setProperty(
    "--layers-container-height",
    `${maxNodesPerLayer * pxPerNode + LABEL_HEIGHT + handleClearance + 8}px`
  );
}
applyNodeLayoutForMode(currentMode);

function nodesToHeight(nodeCount) {
  return nodeCount * pxPerNode;
}

// v in [0, 1]: 0 -> blue, 1 -> red.
function colorForActivation(v) {
  const r = Math.round(clamp(v, 0, 1) * 255);
  const b = Math.round((1 - clamp(v, 0, 1)) * 255);
  return `rgb(${r},0,${b})`;
}

function drawLayerNodes(canvas, count, layerIndex) {
  const ctx = canvas.getContext("2d");
  canvas.height = nodesToHeight(count);
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const acts = currentActivations[layerIndex];
  const matches = acts && acts.length === count && currentActivations.length === layerWidths.length;

  const cx = canvas.width / 2;
  const radius = Math.max(1, pxPerNode / 2 - nodeRadiusGap);
  for (let i = 0; i < count; i++) {
    const cy = i * pxPerNode + pxPerNode / 2;
    ctx.fillStyle = matches ? colorForActivation(acts[i]) : "#556";
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
    ctx.fill();
    if (selectedNode && selectedNode.layer === layerIndex && selectedNode.node === i) {
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cx, cy, radius + 2, 0, 2 * Math.PI);
      ctx.stroke();
    }
  }
}

// Hit-test a click's local (canvas-space) y coordinate against the same
// layout drawLayerNodes uses, returning the clicked node index or null if
// the click landed in the gap between circles.
function nodeIndexAtY(y, count) {
  const radius = Math.max(1, pxPerNode / 2 - nodeRadiusGap);
  const i = Math.round((y - pxPerNode / 2) / pxPerNode);
  if (i < 0 || i >= count) return null;
  const cy = i * pxPerNode + pxPerNode / 2;
  return Math.abs(y - cy) <= radius ? i : null;
}

function updateLayerVisuals() {
  layerCanvases.forEach((canvas, i) => {
    if (canvas) drawLayerNodes(canvas, layerWidths[i], i);
  });
}

// Always renders exactly MAX_LAYERS columns - the first layerWidths.length
// are real layer blocks, the rest are invisible .layer-slot-empty spacers -
// so a config with fewer than MAX_LAYERS layers leaves real, fixed-width
// empty column(s) to the right instead of the real blocks stretching to fill
// the row. renderConfigButtons() below relies on this: it lays out the same
// MAX_LAYERS grid (same slot count, same gap) so +Add/-Remove always land at
// exactly the width and position of a real layer column.
function renderLayersList() {
  layersContainerEl.innerHTML = "";
  layerCanvases = [];
  for (let i = 0; i < MAX_LAYERS; i++) {
    if (i >= layerWidths.length) {
      const spacer = document.createElement("div");
      spacer.className = "layer-slot-empty";
      layersContainerEl.appendChild(spacer);
      continue;
    }

    const width = layerWidths[i];
    const block = document.createElement("div");
    block.className = "layer-block";
    block.style.height = `${nodesToHeight(width) + LABEL_HEIGHT + handleClearance}px`;

    const label = document.createElement("span");
    label.className = "node-count";
    label.textContent = width;
    block.appendChild(label);

    const canvas = document.createElement("canvas");
    canvas.className = "layer-canvas";
    canvas.addEventListener("click", (e) => {
      const nodeIdx = nodeIndexAtY(e.offsetY, layerWidths[i]);
      if (nodeIdx !== null) toggleNodeSelect(i, nodeIdx);
    });
    block.appendChild(canvas);
    layerCanvases[i] = canvas;

    const handle = document.createElement("div");
    handle.className = "resize-handle";
    block.appendChild(handle);

    attachResizeHandlers(handle, block, label, canvas, i);

    layersContainerEl.appendChild(block);
  }

  // Blocks are flexed to share the container's full width; their rendered
  // width is only known once they're in the DOM, so size+draw each canvas
  // after layout settles (flex:1 makes every block the same width).
  layerCanvases.forEach((canvas, i) => {
    canvas.width = canvas.clientWidth;
    drawLayerNodes(canvas, layerWidths[i], i);
  });
  drawConnections();
  renderConfigButtons();
}

function attachResizeHandlers(handle, block, label, canvas, layerIndex) {
  let startY = 0;
  let startNodeCount = 0;
  let dragging = false;

  const onPointerMove = (e) => {
    if (!dragging) return;
    const deltaY = e.clientY - startY;
    const deltaSteps = Math.round(deltaY / pxPerNode);
    const newCount = clamp(
      startNodeCount + deltaSteps * NODE_STEP,
      MIN_NODES_PER_LAYER,
      maxNodesPerLayer
    );
    if (newCount !== layerWidths[layerIndex]) {
      layerWidths[layerIndex] = newCount;
      block.style.height = `${nodesToHeight(newCount) + LABEL_HEIGHT + handleClearance}px`;
      label.textContent = newCount;
      drawLayerNodes(canvas, newCount, layerIndex);
      drawConnections();
    }
  };

  const onPointerUp = () => {
    if (!dragging) return;
    dragging = false;
    block.classList.remove("dragging");
    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerup", onPointerUp);
    sendConfigUpdate();
  };

  handle.addEventListener("pointerdown", (e) => {
    dragging = true;
    startY = e.clientY;
    startNodeCount = layerWidths[layerIndex];
    block.classList.add("dragging");
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    e.preventDefault();
  });
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

// ---- Connection lines: MODEL_GRID_SIZE^2 input pixels -> node(layer 0), node(layer i) ->
// node(layer i+1), and the last hidden layer (or, with no hidden layers, the
// input pixels directly) -> the 10 output nodes in the Classification pane ----
const mainRowEl = document.getElementById("main-row");
const connectionsCanvas = document.getElementById("connections-canvas");
const connectionsCtx = connectionsCanvas.getContext("2d");

// v in [0, 1]: 0 -> blue, 1 -> red, with the given alpha.
function colorForActivationAlpha(v, alpha) {
  const r = Math.round(clamp(v, 0, 1) * 255);
  const b = Math.round((1 - clamp(v, 0, 1)) * 255);
  return `rgba(${r},0,${b},${alpha})`;
}

function getLayerNodeCenters(canvas, count) {
  const rect = canvas.getBoundingClientRect();
  const cx = rect.left + canvas.width / 2;
  const centers = [];
  for (let i = 0; i < count; i++) {
    centers.push({ x: cx, y: rect.top + i * pxPerNode + pxPerNode / 2 });
  }
  return centers;
}

// One point per input pixel, spread evenly over the drawing canvas's own
// height (not showing MODEL_GRID_SIZE^2 separate nodes - just where each pixel's
// connection line starts), anchored to its right edge so lines flow into the
// Configure pane the same way every other layer transition does.
function getInputNodeCenters() {
  const rect = drawCanvas.getBoundingClientRect();
  const n = MODEL_GRID_SIZE * MODEL_GRID_SIZE;
  const centers = new Array(n);
  for (let i = 0; i < n; i++) {
    centers[i] = { x: rect.right, y: rect.top + ((i + 0.5) / n) * rect.height };
  }
  return centers;
}

function getOutputNodeCenters() {
  const centers = [];
  for (let d = 0; d < numClasses; d++) {
    const dot = document.getElementById(`node-dot-${d}`);
    const rect = dot.getBoundingClientRect();
    centers.push({ x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 });
  }
  return centers;
}

// Transitions run input -> layer[0] -> ... -> layer[n-1] -> output, so there's
// always exactly one more weight matrix than there are configured layers
// (zero hidden layers still has the single input -> output matrix).
function edgesValid() {
  if (edgeWeights.length !== layerWidths.length + 1) return false;
  if (rawActivations.length !== layerWidths.length) return false;
  for (let i = 0; i < edgeWeights.length; i++) {
    const expectedSrc = i === 0 ? MODEL_GRID_SIZE * MODEL_GRID_SIZE : layerWidths[i - 1];
    const expectedDst = i < layerWidths.length ? layerWidths[i] : numClasses;
    const matrix = edgeWeights[i];
    if (!matrix || matrix.length !== expectedDst) return false;
    if (matrix.length > 0 && matrix[0].length !== expectedSrc) return false;
  }
  for (let i = 0; i < layerWidths.length; i++) {
    if (!rawActivations[i] || rawActivations[i].length !== layerWidths[i]) return false;
  }
  return true;
}

function drawConnections() {
  const rowRect = mainRowEl.getBoundingClientRect();
  connectionsCanvas.style.width = `${rowRect.width}px`;
  connectionsCanvas.style.height = `${rowRect.height}px`;
  connectionsCanvas.width = rowRect.width;
  connectionsCanvas.height = rowRect.height;
  connectionsCtx.clearRect(0, 0, connectionsCanvas.width, connectionsCanvas.height);

  if (!edgesValid()) return;

  const hiddenCenters = layerCanvases.map((canvas, i) => getLayerNodeCenters(canvas, layerWidths[i]));
  const outputCenters = getOutputNodeCenters();
  const srcCentersByTransition = [getInputNodeCenters(), ...hiddenCenters];
  const normalizedPixels = pixels.map((p) => p / 255);

  // First pass: compute every edge's (source activation * weight), and the
  // largest magnitude seen this frame, to normalize colors/opacity by.
  const transitions = [];
  let maxAbs = 1e-9;
  for (let i = 0; i < edgeWeights.length; i++) {
    const srcCenters = srcCentersByTransition[i];
    const dstCenters = i < layerWidths.length ? hiddenCenters[i] : outputCenters;
    const weights = edgeWeights[i];
    const acts = i === 0 ? normalizedPixels : rawActivations[i - 1];
    const values = [];
    for (let d = 0; d < dstCenters.length; d++) {
      const row = new Array(srcCenters.length);
      for (let s = 0; s < srcCenters.length; s++) {
        const v = acts[s] * weights[d][s];
        row[s] = v;
        if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
      }
      values.push(row);
    }
    transitions.push({ srcCenters, dstCenters, values });
  }

  // Second pass: draw. Weak/near-zero connections are skipped entirely (see
  // CONNECTION_MIN_MAGNITUDE) so the strongest signal paths stand out instead
  // of the view turning to mush - especially important for the input
  // transition, which can be tens of thousands of lines.
  const offsetX = rowRect.left;
  const offsetY = rowRect.top;
  transitions.forEach(({ srcCenters, dstCenters, values }) => {
    for (let d = 0; d < dstCenters.length; d++) {
      for (let s = 0; s < srcCenters.length; s++) {
        const norm = clamp(0.5 + values[d][s] / (2 * maxAbs), 0, 1);
        const magnitude = Math.abs(norm - 0.5) * 2;
        if (magnitude < CONNECTION_MIN_MAGNITUDE) continue;
        const alpha = 0.06 + magnitude * 0.7;
        connectionsCtx.strokeStyle = colorForActivationAlpha(norm, alpha);
        connectionsCtx.lineWidth = 1;
        connectionsCtx.beginPath();
        connectionsCtx.moveTo(srcCenters[s].x - offsetX, srcCenters[s].y - offsetY);
        connectionsCtx.lineTo(dstCenters[d].x - offsetX, dstCenters[d].y - offsetY);
        connectionsCtx.stroke();
      }
    }
  });
}

function sendConfigUpdate() {
  sendMessage({ type: "config_update", layers: layerWidths });
}

const addLayerBtn = document.getElementById("add-layer-btn");
const removeLayerBtn = document.getElementById("remove-layer-btn");
const configButtonsEl = document.getElementById("config-buttons");

// Places +Add in the empty column where the next layer would land (slot
// layerWidths.length, only while under MAX_LAYERS) and -Remove under the
// rightmost existing layer (slot layerWidths.length - 1) - on the same
// MAX_LAYERS-slot grid renderLayersList() draws #layers-container with, so
// both buttons are always exactly as wide as a real layer column. At 0
// layers only +Add shows (in slot 0); at MAX_LAYERS layers only -Remove
// shows (in the last slot).
function renderConfigButtons() {
  configButtonsEl.innerHTML = "";
  const layerCount = layerWidths.length;
  for (let slot = 0; slot < MAX_LAYERS; slot++) {
    const slotEl = document.createElement("div");
    slotEl.className = "config-button-slot";
    if (slot === layerCount && layerCount < MAX_LAYERS) {
      addLayerBtn.hidden = false;
      slotEl.appendChild(addLayerBtn);
    } else if (slot === layerCount - 1) {
      removeLayerBtn.hidden = false;
      slotEl.appendChild(removeLayerBtn);
    }
    configButtonsEl.appendChild(slotEl);
  }
}

addLayerBtn.addEventListener("click", () => {
  if (layerWidths.length >= MAX_LAYERS) return;
  layerWidths.push(MIN_NODES_PER_LAYER);
  renderLayersList();
  sendConfigUpdate();
});

removeLayerBtn.addEventListener("click", () => {
  if (layerWidths.length <= MIN_LAYERS) return;
  layerWidths.pop();
  renderLayersList();
  sendConfigUpdate();
});

document.getElementById("train-btn").addEventListener("click", () => {
  sendMessage({ type: "retrain" });
  resetTrainingPlots("trainingEllipsis");
  resetExplain();
  resetSelectedNode();
});

renderLayersList();
sendConfigUpdate();

// ---- Classification pane ----
const predictedClassEl = document.getElementById("predicted-class");
const probBarsEl = document.getElementById("prob-bars");

// Rebuilt from scratch on every mode_selected (initial connect, an explicit
// mode switch, or a Drawings-mode reroll) since numClasses/classLabels
// change. The output-node circle stays a small fixed-size geometry anchor
// (just the label's first letter, so getOutputNodeCenters()/the connection-
// line math are unaffected by label length) with the full class name in an
// adjacent text label - for Digits this reduces to exactly the digit glyph,
// since classLabels[d] already equals "0".."9".
function initProbBars() {
  probBarsEl.innerHTML = "";
  const labels = currentClassLabels();
  // Every .class-label gets fixed to the width of the longest name in this
  // round (in ch, ~1 character wide) via a shared CSS custom property, so
  // every row's prob-bar-track lines up at the same x position - see
  // .class-label in style.css.
  const maxLabelLength = labels.reduce((max, label) => Math.max(max, label.length), 0);
  probBarsEl.style.setProperty("--class-label-width", `${maxLabelLength}ch`);
  for (let d = 0; d < numClasses; d++) {
    const label = labels[d];
    const glyph = label.length ? label[0].toUpperCase() : "?";
    const row = document.createElement("div");
    row.className = "prob-bar-row";
    row.innerHTML = `
      <div class="digit-node">
        <span class="output-node-dot" id="node-dot-${d}" title="${label}">${glyph}</span>
      </div>
      <span class="class-label">${label}</span>
      <div class="prob-bar-track"><div class="prob-bar-fill" id="bar-${d}"></div></div>
    `;
    probBarsEl.appendChild(row);
    const dotEl = document.getElementById(`node-dot-${d}`);
    dotEl.addEventListener("click", () => toggleExplain(d));
  }
}

// Remembered so a language switch can replay this exact classification (via
// initProbBars() rebuilding the rows, then this re-running) instead of the
// bars going blank until the next live classification tick.
let lastClassificationProbs = null;
let lastClassificationPredicted = null;

// Unlike the hidden-layer nodes (normalized per forward pass), the output
// nodes use the softmax probability directly as the 0-1 color value, so the
// color is comparable across separate classifications, not just within one.
function renderClassification(probs, predicted) {
  lastClassificationProbs = probs;
  lastClassificationPredicted = predicted;
  for (let d = 0; d < numClasses; d++) {
    const p = probs[d] || 0;
    const color = colorForActivation(p);
    const pct = Math.round(p * 100);
    const barEl = document.getElementById(`bar-${d}`);
    barEl.style.width = `${pct}%`;
    barEl.style.background = color;
    const dot = document.getElementById(`node-dot-${d}`);
    if (dot) dot.style.background = color;
  }
  predictedClassEl.textContent =
    predicted === null || predicted === undefined ? "-" : currentClassLabels()[predicted];
}

// ---- AI status pane ----
const progressLabelEl = document.getElementById("progress-label");
const progressBarFillEl = document.getElementById("progress-bar-fill");

// Canvases are backed at devicePixelRatio resolution so text/lines stay
// sharp on hi-DPI displays; drawing code below still works in CSS-pixel
// units (lossW/lossH etc.) thanks to the ctx.scale(dpr, dpr).
function setupCanvasHiDPI(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvas.clientWidth || canvas.width;
  const cssHeight = canvas.height; // height isn't stretched by CSS, so the attribute is the intended logical size
  canvas.width = Math.round(cssWidth * dpr);
  canvas.height = Math.round(cssHeight * dpr);
  canvas.style.height = `${cssHeight}px`;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  return { ctx, w: cssWidth, h: cssHeight };
}

const lossCanvas = document.getElementById("loss-canvas");
const { ctx: lossCtx, w: lossW, h: lossH } = setupCanvasHiDPI(lossCanvas);
const accuracyCanvas = document.getElementById("accuracy-canvas");
const { ctx: accuracyCtx, w: accuracyW, h: accuracyH } = setupCanvasHiDPI(accuracyCanvas);

// key is a TRANSLATIONS key (see i18n.js), not resolved text - remembered in
// currentProgressKey so refreshProgressLabel() (called on a language switch)
// can re-render this exact same state in the new language.
let currentProgressKey = "idle";

function setProgress(fraction, key) {
  progressBarFillEl.style.width = `${Math.round(clamp(fraction, 0, 1) * 100)}%`;
  currentProgressKey = key;
  progressLabelEl.textContent = t(key);
}

function refreshProgressLabel() {
  progressLabelEl.textContent = t(currentProgressKey);
}

// Shared by the Train button and a mode switch (both start a fresh
// training history from the visitor's point of view). key is a
// TRANSLATIONS key, same as setProgress().
function resetTrainingPlots(key) {
  trainLossHistory = [];
  valLossPoints = [];
  accuracyHistory = [];
  setProgress(0, key);
  drawLossPlot();
  drawAccuracyPlot();
  renderTrainingSummary({});
}

function handleTrainingStatus(msg) {
  renderTrainingSummary(msg);
  if (msg.state === "running") {
    setProgress(0, "trainingEllipsis");
  } else if (msg.state === "error") {
    setProgress(0, "trainingError");
  } else if (msg.state === "idle") {
    setProgress(1, "trainingComplete");
  }
}

function handleTrainingMetrics(msg) {
  trainLossHistory.push(msg.loss);
  if (msg.val_loss != null) {
    valLossPoints.push({ x: trainLossHistory.length - 1, y: msg.val_loss });
  }
  if (msg.accuracy != null) {
    accuracyHistory.push(msg.accuracy);
  }
  // Each epoch pushes a mid-epoch sample (no val_loss yet) and one
  // end-of-epoch sample (val_loss present); use that to tell how far
  // through the current epoch training is.
  const fraction = msg.val_loss != null ? (msg.epoch + 1) / msg.total_epochs : msg.epoch / msg.total_epochs;
  setProgress(fraction, "trainingEllipsis");
  drawLossPlot();
  drawAccuracyPlot();
}

function drawLossPlot() {
  const w = lossW;
  const h = lossH;
  lossCtx.fillStyle = "black";
  lossCtx.fillRect(0, 0, w, h);

  if (trainLossHistory.length < 2) return;

  const allLosses = trainLossHistory.concat(valLossPoints.map((p) => p.y));
  const maxLoss = Math.max(...allLosses, 0.001);
  const step = w / (trainLossHistory.length - 1);
  const toY = (loss) => h - (loss / maxLoss) * (h - 10) - 5;

  lossCtx.strokeStyle = "#2a6df4";
  lossCtx.lineWidth = 2;
  lossCtx.beginPath();
  trainLossHistory.forEach((loss, i) => {
    const x = i * step;
    const y = toY(loss);
    if (i === 0) lossCtx.moveTo(x, y);
    else lossCtx.lineTo(x, y);
  });
  lossCtx.stroke();

  if (valLossPoints.length > 0) {
    lossCtx.strokeStyle = "#f4a52a";
    lossCtx.lineWidth = 2;
    lossCtx.beginPath();
    valLossPoints.forEach((p, i) => {
      const x = p.x * step;
      const y = toY(p.y);
      if (i === 0) lossCtx.moveTo(x, y);
      else lossCtx.lineTo(x, y);
    });
    lossCtx.stroke();

    lossCtx.fillStyle = "#f4a52a";
    valLossPoints.forEach((p) => {
      lossCtx.beginPath();
      lossCtx.arc(p.x * step, toY(p.y), 3, 0, 2 * Math.PI);
      lossCtx.fill();
    });
  }

  drawLossLegend();
}

function drawLossLegend() {
  const legendY = 14;
  lossCtx.font = "12px system-ui, sans-serif";

  lossCtx.fillStyle = "#2a6df4";
  lossCtx.fillRect(10, legendY - 8, 12, 4);
  lossCtx.fillStyle = "#eee";
  lossCtx.fillText("train loss", 26, legendY);

  lossCtx.fillStyle = "#f4a52a";
  lossCtx.fillRect(110, legendY - 8, 12, 4);
  lossCtx.fillStyle = "#eee";
  lossCtx.fillText("val loss", 126, legendY);
}

function drawAccuracyPlot() {
  const w = accuracyW;
  const h = accuracyH;
  accuracyCtx.fillStyle = "black";
  accuracyCtx.fillRect(0, 0, w, h);

  if (accuracyHistory.length < 2) return;

  const step = w / (accuracyHistory.length - 1);
  const toY = (acc) => h - acc * (h - 10) - 5; // fixed 0-1 scale

  accuracyCtx.strokeStyle = "#3ecf6e";
  accuracyCtx.lineWidth = 2;
  accuracyCtx.beginPath();
  accuracyHistory.forEach((acc, i) => {
    const x = i * step;
    const y = toY(acc);
    if (i === 0) accuracyCtx.moveTo(x, y);
    else accuracyCtx.lineTo(x, y);
  });
  accuracyCtx.stroke();

  const last = accuracyHistory[accuracyHistory.length - 1];
  accuracyCtx.font = "12px system-ui, sans-serif";
  accuracyCtx.fillStyle = "#eee";
  accuracyCtx.fillText(`${(last * 100).toFixed(1)}%`, 10, 14);
}

// ---- Debug mode ----
const debugToggleBtn = document.getElementById("debug-toggle-btn");
const debugOptionsEl = document.getElementById("debug-options");
const loadSampleBtn = document.getElementById("load-sample-btn");
const sampleInfoEl = document.getElementById("sample-info");
const trainingSummaryEl = document.getElementById("training-summary");
const trainingPlotsEl = document.getElementById("training-plots");

const STOP_REASON_KEYS = {
  early_stopping: "stopReasonEarlyStopping",
  max_epochs: "stopReasonMaxEpochs",
  stopped_by_user: "stopReasonStoppedByUser",
};

// Remembered so a language switch can re-render the same summary via
// renderTrainingSummary(lastTrainingStatusMsg) instead of leaving stale text
// in the old language.
let lastTrainingStatusMsg = {};

// msg: a training_status message, or {} to clear the summary (a fresh
// training run's stats haven't landed yet, or none has ever completed in
// this mode since the page loaded - see resetTrainingPlots()). epochs_trained
// only appears on a training_status message once a run has actually ended
// (idle/stopped/error - never "running"), so its absence is what gates this.
function renderTrainingSummary(msg) {
  lastTrainingStatusMsg = msg;
  if (msg.epochs_trained == null) {
    trainingSummaryEl.textContent = "";
    return;
  }
  const parts = [t("epochsTrained", msg.epochs_trained)];
  if (msg.best_val_accuracy != null) {
    parts.push(t("valAccuracy", (msg.best_val_accuracy * 100).toFixed(1)));
  }
  if (msg.stop_reason) {
    const reasonKey = STOP_REASON_KEYS[msg.stop_reason];
    parts.push(t("stoppingCondition", reasonKey ? t(reasonKey) : msg.stop_reason));
  }
  trainingSummaryEl.textContent = parts.join(" | ");
}

debugToggleBtn.addEventListener("click", () => {
  const isActive = debugToggleBtn.classList.toggle("active");
  debugOptionsEl.hidden = !isActive;
  trainingPlotsEl.hidden = !isActive;
});

loadSampleBtn.addEventListener("click", () => {
  sendMessage({ type: "debug_load_sample" });
});

// Nearest-neighbor upscale of a flattened srcSize x srcSize array to
// dstSize x dstSize (dstSize a whole multiple of srcSize) - used only to
// display a native-resolution dataset sample on a finer draw canvas
// (Drawings mode); a no-op whenever the two resolutions already match.
function upscalePixels(src, srcSize, dstSize) {
  if (srcSize === dstSize) return src.slice();
  const factor = dstSize / srcSize;
  const dst = new Array(dstSize * dstSize);
  for (let y = 0; y < dstSize; y++) {
    const sy = Math.floor(y / factor);
    for (let x = 0; x < dstSize; x++) {
      dst[y * dstSize + x] = src[sy * srcSize + Math.floor(x / factor)];
    }
  }
  return dst;
}

// Remembered (both languages, like class_names/class_names_sv) so a
// language switch can re-render this line via refreshDebugSampleInfo()
// instead of leaving stale text in the old language.
let lastDebugSampleLabel = null;
let lastDebugSampleLabelSv = null;

function handleDebugSample(msg) {
  // msg.pixels always arrives at MODEL_GRID_SIZE (that's the dataset's own
  // resolution) - upscale to the current draw resolution so it displays and
  // can be drawn over consistently with the rest of the canvas.
  pixels = upscalePixels(msg.pixels, MODEL_GRID_SIZE, drawGridSize);
  redrawCanvas();
  lastDebugSampleLabel = msg.label;
  lastDebugSampleLabelSv = msg.label_sv;
  refreshDebugSampleInfo();
  refreshActiveExplanation();
}

function refreshDebugSampleInfo() {
  if (lastDebugSampleLabel == null) return;
  const label = currentLanguage === "sv" && lastDebugSampleLabelSv != null ? lastDebugSampleLabelSv : lastDebugSampleLabel;
  sampleInfoEl.textContent = t("trueLabel", label);
}

// ---- Menu / mode selection ----
const menuOverlayEl = document.getElementById("menu-overlay");
const menuLoadingEl = document.getElementById("menu-loading");
const headerRowEl = document.getElementById("header-row");
const menuModeBtns = document.querySelectorAll(".menu-mode-btn");
const drawHeadingEl = document.getElementById("draw-heading");
const explainHintEl = document.getElementById("explain-hint");

// Mode-dependent copy - keyed off TRANSLATIONS' own drawHeading*/explainHint*/
// loadSample* naming (see i18n.js), so adding a mode just means adding the
// matching *Digits/*Drawings keys there, nothing to change here.
function applyModeCopy(mode) {
  const suffix = mode === "drawings" ? "Drawings" : "Digits";
  drawHeadingEl.textContent = t(`drawHeading${suffix}`);
  explainHintEl.textContent = t(`explainHint${suffix}`);
  loadSampleBtn.textContent = t(`loadSample${suffix}`);
}

// Re-renders every piece of UI chrome text in the current language - the
// static headings/buttons/hints below, plus everything that depends on
// runtime state (applyModeCopy needs currentMode; the rest replay whatever
// they last rendered via their own remembered key/message - see
// refreshProgressLabel, renderTrainingSummary, refreshDebugSampleInfo).
// Called once at load (the static HTML already matches the Swedish default,
// so this is a no-op paint the first time - see index.html) and again on
// every language switch.
function applyTranslations() {
  document.title = t("pageTitle");
  document.getElementById("menu-title").textContent = t("pageTitle");
  document.getElementById("header-title").textContent = t("pageTitle");
  document.getElementById("menu-subtitle").textContent = t("menuSubtitle");
  menuModeBtns.forEach((btn) => {
    btn.textContent = t(btn.dataset.mode === "drawings" ? "drawingsMode" : "digitsMode");
  });
  menuLoadingEl.textContent = t("loading");
  document.getElementById("menu-btn").textContent = t("backToMenu");
  document.getElementById("reset-btn").textContent = t("reset");
  document.getElementById("config-heading").textContent = t("configureHeading");
  addLayerBtn.textContent = t("addLayer");
  removeLayerBtn.textContent = t("removeLayer");
  document.getElementById("config-hint").textContent = t("configHint");
  document.getElementById("classification-heading").textContent = t("classificationHeading");
  document.getElementById("status-heading").textContent = t("aiStatusHeading");
  document.getElementById("loss-plot-label").textContent = t("lossLabel");
  document.getElementById("accuracy-plot-label").textContent = t("validationAccuracyLabel");
  document.getElementById("debug-heading").textContent = t("debugHeading");
  debugToggleBtn.textContent = t("debugToggle");
  document.getElementById("inactivity-heading").textContent = t("inactivityHeading");
  document.getElementById("inactivity-cancel-btn").textContent = t("inactivityCancel");

  applyModeCopy(currentMode);

  // Rebuilds the classification pane's rows against currentClassLabels(),
  // which is now the newly-selected language's names (see class_names_sv in
  // server/main.py) - then replays whatever was last actually classified/
  // explained so the bars don't go blank until the next live classification
  // tick.
  initProbBars();
  if (lastClassificationProbs) renderClassification(lastClassificationProbs, lastClassificationPredicted);
  updateExplainHighlight();

  refreshProgressLabel();
  renderTrainingSummary(lastTrainingStatusMsg);
  refreshDebugSampleInfo();
}

function setLanguage(lang) {
  currentLanguage = lang;
  document.documentElement.lang = lang;
  document.querySelectorAll(".lang-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.lang === lang);
  });
  applyTranslations();
}

document.querySelectorAll(".lang-btn").forEach((btn) => {
  btn.addEventListener("click", () => setLanguage(btn.dataset.lang));
});

menuModeBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    menuModeBtns.forEach((b) => (b.disabled = true));
    menuLoadingEl.hidden = false;
    sendMessage({ type: "select_mode", mode: btn.dataset.mode });
  });
});

document.getElementById("menu-btn").addEventListener("click", () => {
  // Only re-shows the overlay - does NOT send select_mode, so backing out
  // of the menu without picking anything leaves the current scene intact.
  menuOverlayEl.hidden = false;
});

// Handles every mode_selected the server ever sends: the initial sync right
// after connecting (see the server's ws_endpoint), an explicit menu pick, or
// a Drawings-mode reroll (re-picking "Drawings" while already in it).
function handleModeSelected(msg) {
  numClasses = msg.num_classes;
  classLabels = msg.class_names.slice();
  classLabelsSv = (msg.class_names_sv || msg.class_names).slice();
  currentMode = msg.mode;
  updateDrawResolution(msg.mode);
  applyNodeLayoutForMode(msg.mode);

  applyModeCopy(msg.mode);
  initProbBars();

  // Make the panels visible before anything below measures their layout
  // (canvas sizing in renderLayersList, getBoundingClientRect in
  // drawConnections) - while #main-row is still hidden, every element in it
  // reports zero size, which corrupts canvas pixel buffers (they render as
  // squashed bars instead of circles until something else happens to
  // re-render them, e.g. add/remove layer).
  menuModeBtns.forEach((b) => (b.disabled = false));
  menuLoadingEl.hidden = true;
  menuOverlayEl.hidden = true;
  headerRowEl.hidden = false;
  mainRowEl.hidden = false;

  // Widths set while in the other mode may exceed this mode's own node cap
  // (see MAX_NODES_PER_LAYER_DIGITS/DRAWINGS) - re-clamp and, if that
  // actually changed anything, push the correction to the server the same
  // way a manual resize would.
  const clampedWidths = layerWidths.map((w) => clamp(w, MIN_NODES_PER_LAYER, maxNodesPerLayer));
  const widthsChanged = clampedWidths.some((w, i) => w !== layerWidths[i]);
  layerWidths = clampedWidths;
  renderLayersList();
  if (widthsChanged) sendConfigUpdate();

  resetDrawingState();
  // select_mode() on the server always loads *something* now - a real
  // checkpoint, or else a fresh random-init model (see
  // server/inference.py's load_random()) - so edge_weights here reflects
  // whichever one, letting the connection-line view light up immediately
  // instead of staying dark until the first real Train completes.
  edgeWeights = msg.edge_weights || [];
  drawConnections();
  resetTrainingPlots(msg.checkpoint_ready ? "idle" : "noTrainedModel");
}

// Establishes the Swedish default for real (not just via the static HTML's
// own already-Swedish text - see index.html) - sets .lang-btn.active and
// document.documentElement.lang consistently with currentLanguage, and
// replays applyModeCopy/refreshProgressLabel/etc against real state. A
// no-op repaint the first time, since the static HTML already matches.
setLanguage(currentLanguage);

// ---- Inactivity timeout ----
// After this long with no activity anywhere on the page (menu or a live
// session), show a warning; after that, this many seconds of no response
// resets the whole kiosk back to the menu (see server/main.py's reset_kiosk
// handler) for the next visitor.
const INACTIVITY_WARNING_MS = 60 * 1000;
const INACTIVITY_COUNTDOWN_SECONDS = 60;

const inactivityOverlayEl = document.getElementById("inactivity-overlay");
const inactivityMessageEl = document.getElementById("inactivity-message");
const inactivityCancelBtn = document.getElementById("inactivity-cancel-btn");

let inactivityWarningTimer = null;
let inactivityCountdownInterval = null;
let inactivityCountdownRemaining = 0;

function armInactivityTimer() {
  clearTimeout(inactivityWarningTimer);
  inactivityWarningTimer = setTimeout(showInactivityWarning, INACTIVITY_WARNING_MS);
}

function updateInactivityMessage() {
  inactivityMessageEl.textContent = t("inactivityMessage", inactivityCountdownRemaining);
}

function showInactivityWarning() {
  inactivityCountdownRemaining = INACTIVITY_COUNTDOWN_SECONDS;
  updateInactivityMessage();
  inactivityOverlayEl.hidden = false;
  inactivityCountdownInterval = setInterval(() => {
    inactivityCountdownRemaining -= 1;
    if (inactivityCountdownRemaining <= 0) {
      clearInterval(inactivityCountdownInterval);
      sendMessage({ type: "reset_kiosk" });
      return;
    }
    updateInactivityMessage();
  }, 1000);
}

function cancelInactivityWarning() {
  clearInterval(inactivityCountdownInterval);
  inactivityOverlayEl.hidden = true;
  armInactivityTimer();
}

inactivityCancelBtn.addEventListener("click", cancelInactivityWarning);

// Broadly scoped on purpose (drawing, dragging a resize handle, clicking any
// button, typing) - anything counts as "still here". Only postpones the
// *next* warning, though: once the warning is actually showing, only the
// explicit Cancel button above dismisses it, so idly resting a finger on
// the dimmed backdrop doesn't silently reset the countdown.
["pointerdown", "pointermove", "keydown", "wheel"].forEach((eventName) => {
  window.addEventListener(eventName, () => {
    if (inactivityOverlayEl.hidden) armInactivityTimer();
  });
});

armInactivityTimer();
