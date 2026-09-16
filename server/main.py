import asyncio
import json
import logging
import os
import random

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from server.config import (
    CHECKPOINT_PATH_DIGITS,
    CHECKPOINT_PATH_DRAWINGS,
    DEFAULT_LAYER_WIDTHS,
    DRAW_DOWNSCALE_FACTOR,
    GRID_SIZE,
    MAX_LAYERS,
    MAX_NODES_PER_LAYER_DIGITS,
    MAX_NODES_PER_LAYER_DRAWINGS,
    MIN_LAYERS,
    MIN_NODES_PER_LAYER,
    NODE_STEP,
    NUM_CLASSES,
    QUICKDRAW_NUM_CLASSES,
    TITLE_HELP_SAMPLE_COUNT,
)
from server.data import get_raw_test_samples, get_raw_train_samples, prewarm_mnist
from server.help_messages import HELP_MESSAGES
from server.inference import InferenceService
from server.model_interface import ArchitectureSpec, LayerSpec
from server.preprocessing import downscale_by_fill_count
from server.quickdraw_categories import CATEGORY_POOL, CATEGORY_TRANSLATIONS_SV
from server.quickdraw_data import (
    get_quickdraw_raw_train_samples,
    get_quickdraw_raw_val_samples,
    pick_slice_starts,
    prewarm_categories,
)
from server.training_manager import TrainingManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kiosk")

app = FastAPI()

# This module is the only place that knows "Digits" and "Drawings" exist as
# concepts - InferenceService, TrainingManager/train_worker, and
# quickdraw_data all operate on generic (checkpoint_path, class_names,
# dataset) parameters, never a mode string.
DIGIT_CLASS_NAMES: list[str] = [str(i) for i in range(NUM_CLASSES)]

inference = InferenceService()
clients: set[WebSocket] = set()

# Current config, updated as the visitor edits it; used when Retrain is clicked.
current_layer_widths: list[int] = list(DEFAULT_LAYER_WIDTHS)

# Last-drawn pixels (single shared kiosk state), so a fresh checkpoint can be
# re-applied to whatever is currently on the canvas without waiting for the
# visitor to draw again.
last_pixels: list[int] | None = None

training_manager: TrainingManager | None = None

# Loaded lazily on startup; used by debug-mode "load a real sample" tooling
# in Digits mode.
mnist_test_images = None
mnist_test_labels = None
# Loaded once at startup too (same fixed TRAIN_SUBSET_SEED subset every time -
# see get_raw_train_samples()) - the title help dialog's Digits-mode sample
# grid picks randomly from this.
mnist_train_images = None
mnist_train_labels = None

# Current mode/task (shared kiosk state, same as last_pixels/current_layer_widths
# above - one kiosk, one active mode at a time, broadcast to every client).
# Starts at None (no mode chosen yet, not "digits") so a client's very first
# connection to a freshly booted server sees the menu instead of the server
# silently defaulting into Digits before the visitor gets to choose - a
# reconnect *after* a mode has been picked (page reload, second tab) does
# still sync straight to whatever mode is already active, see ws_endpoint.
current_mode: str | None = None
current_class_names: list[str] = []
# Quick Draw only: per-category slice start chosen at select_mode time, reused
# for both training (see the "retrain" handler) and the debug sample val set
# below, so the two can never disagree about where the train/val split falls.
current_slice_starts: dict[str, int] = {}
quickdraw_val_images = None
quickdraw_val_labels = None
# Same slice_starts as above, but the train portion - see _fetch_quickdraw_task.
# The title help dialog's Drawings-mode sample grid picks randomly from this.
quickdraw_train_images = None
quickdraw_train_labels = None


def checkpoint_path_for_mode(mode: str) -> str:
    return CHECKPOINT_PATH_DRAWINGS if mode == "drawings" else CHECKPOINT_PATH_DIGITS


def clear_checkpoints() -> None:
    """Kiosk starts fresh every boot - no pre-trained model should carry over
    from a previous run/visitor. Called once at startup, before any client
    can connect and pick a mode."""
    for path in (CHECKPOINT_PATH_DIGITS, CHECKPOINT_PATH_DRAWINGS):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def max_nodes_per_layer_for_mode(mode: str | None) -> int:
    return MAX_NODES_PER_LAYER_DRAWINGS if mode == "drawings" else MAX_NODES_PER_LAYER_DIGITS


def draw_grid_size_for_mode(mode: str) -> int:
    """Resolution of the `pixels` array a draw_update is expected to carry.
    Digits draws natively at the model's own GRID_SIZE; Drawings draws
    DRAW_DOWNSCALE_FACTOR times finer and gets block-downscaled back down to
    GRID_SIZE in the draw_update handler before it ever reaches last_pixels."""
    return GRID_SIZE * DRAW_DOWNSCALE_FACTOR if mode == "drawings" else GRID_SIZE


def dataset_for_mode(mode: str) -> str:
    return "quickdraw" if mode == "drawings" else "mnist"


def class_names_sv_for(class_names: list[str], mode: str) -> list[str]:
    """Swedish display names for class_names, same order - digit labels
    ("0".."9") need no translating; Quick Draw category names are looked up
    in CATEGORY_TRANSLATIONS_SV (built from quickdraw_curated_list.csv's own
    "swedish" column - see server/quickdraw_categories.py). The .get(name,
    name) fallback never actually triggers today (Drawings' class_names are
    always sampled from CATEGORY_POOL, which is exactly
    CATEGORY_TRANSLATIONS_SV's key set - see select_mode) but keeps this from
    ever crashing on prod-would-be-fine-degraded-instead if that ever stops
    holding."""
    if mode != "drawings":
        return list(class_names)
    return [CATEGORY_TRANSLATIONS_SV.get(name, name) for name in class_names]


def pick_title_help_samples() -> list[dict]:
    """Random TITLE_HELP_SAMPLE_COUNT-image sample (with truth labels) of
    whatever current_mode actually trains on right now - see
    get_raw_train_samples()/get_quickdraw_raw_train_samples(). Powers the
    title help dialog's "here's what it's trained on" grid (see
    static/app.js) with the real data, not a canned example. Called fresh
    every time that dialog is opened (see the "get_title_help_samples"
    handler in ws_endpoint below), not just once per mode_selected, so
    reopening it reshuffles the picture."""
    if current_mode == "drawings":
        images, labels = quickdraw_train_images, quickdraw_train_labels
    else:
        images, labels = mnist_train_images, mnist_train_labels
    if images is None or len(images) == 0:
        return []

    count = min(TITLE_HELP_SAMPLE_COUNT, len(images))
    samples = []
    for idx in random.sample(range(len(images)), count):
        label = current_class_names[int(labels[idx])] if current_mode == "drawings" else str(int(labels[idx]))
        label_sv = class_names_sv_for([label], current_mode)[0]
        samples.append({"pixels": images[idx].flatten().tolist(), "label": label, "label_sv": label_sv})
    return samples


def build_mode_selected_message() -> dict:
    assert current_mode is not None
    return {
        "type": "mode_selected",
        "mode": current_mode,
        "num_classes": len(current_class_names),
        "class_names": list(current_class_names),
        "class_names_sv": class_names_sv_for(current_class_names, current_mode),
        # is_trained, not is_ready: select_mode() always loads *something*
        # (a real checkpoint, or else a random-init model - see
        # inference.load_random()) so live classification/edge_weights are
        # never blank, but the "no trained model yet" UI text should still
        # only go away once a real training run has actually happened.
        "checkpoint_ready": inference.is_trained,
        "edge_weights": inference.get_edge_weights(),
    }


def clamp_layer_widths(widths: list[int], max_nodes: int) -> list[int]:
    widths = widths[:MAX_LAYERS] if len(widths) > MAX_LAYERS else widths
    if len(widths) < MIN_LAYERS:
        widths = list(DEFAULT_LAYER_WIDTHS)

    def snap(w: int) -> int:
        w = max(MIN_NODES_PER_LAYER, min(max_nodes, w))
        return round(w / NODE_STEP) * NODE_STEP

    return [snap(w) for w in widths]


async def broadcast(message: dict) -> None:
    dead = []
    payload = json.dumps(message)
    for ws in clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


def normalize_activations(activations: list[np.ndarray]) -> list[list[float]]:
    """Min-max normalize hidden-layer activation values to [0, 1], using the
    min/max across all layers of this single forward pass (so node colors are
    comparable across the whole network, not just within one layer)."""
    if not activations:
        return []
    all_vals = np.concatenate([a.flatten() for a in activations])
    lo, hi = float(all_vals.min()), float(all_vals.max())
    rng = hi - lo
    if rng < 1e-9:
        return [[0.5] * len(layer) for layer in activations]
    return [[(float(v) - lo) / rng for v in layer] for layer in activations]


async def classify_and_broadcast() -> None:
    if last_pixels is None:
        return
    probs, activations = inference.predict_with_activations(last_pixels)
    if probs is None:
        return
    await broadcast(
        {
            "type": "classification",
            "probs": [float(p) for p in probs],
            "predicted": int(probs.argmax()),
            "activations": normalize_activations(activations),
            # Raw (un-normalized) values, so the client can compute
            # connection-line strength as (source activation * edge weight).
            "raw_activations": [[float(v) for v in layer] for layer in activations],
        }
    )


async def handle_training_message(message: dict) -> None:
    if message.get("type") == "checkpoint_loaded":
        msg_checkpoint_path = message.get("checkpoint_path")
        # A training run started in a mode that's since been switched away
        # from (see select_mode's training_manager.stop() call) can still
        # have a final checkpoint_loaded message land after the switch - drop
        # it rather than reloading the wrong mode's checkpoint over the one
        # the visitor is now looking at.
        if msg_checkpoint_path is not None and msg_checkpoint_path != checkpoint_path_for_mode(current_mode):
            return
        inference.reload(checkpoint_path_for_mode(current_mode), current_class_names)
        message = dict(message)
        message["edge_weights"] = inference.get_edge_weights()
        await broadcast(message)
        await classify_and_broadcast()
    else:
        await broadcast(message)


def _fetch_quickdraw_task(class_names: list[str]) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Runs on a worker thread (network I/O) - see select_mode. Picks fresh
    per-category slice offsets and returns them alongside both the held-out
    validation images/labels and the train images/labels drawn from those
    same offsets, so training (which reuses these exact slice_starts), the
    debug sample button, and the title help dialog's sample grid can never
    disagree about where the train/val split falls."""
    slice_starts = pick_slice_starts(class_names)
    val_images, val_labels = get_quickdraw_raw_val_samples(class_names, slice_starts)
    train_images, train_labels = get_quickdraw_raw_train_samples(class_names, slice_starts)
    return slice_starts, val_images, val_labels, train_images, train_labels


async def select_mode(mode: str) -> None:
    global current_mode, current_class_names, current_slice_starts
    global last_pixels, quickdraw_val_images, quickdraw_val_labels, current_layer_widths
    global quickdraw_train_images, quickdraw_train_labels

    assert training_manager is not None
    training_manager.stop()
    last_pixels = None
    # Re-clamp against the new mode's own node cap - widths set while in
    # Drawings (up to MAX_NODES_PER_LAYER_DRAWINGS) would otherwise carry an
    # out-of-range value into Digits (MAX_NODES_PER_LAYER_DIGITS) or vice versa.
    current_layer_widths = clamp_layer_widths(current_layer_widths, max_nodes_per_layer_for_mode(mode))

    if mode == "drawings":
        class_names = random.sample(CATEGORY_POOL, QUICKDRAW_NUM_CLASSES)
        slice_starts, val_images, val_labels, train_images, train_labels = await asyncio.to_thread(
            _fetch_quickdraw_task, class_names
        )
        current_class_names = class_names
        current_slice_starts = slice_starts
        quickdraw_val_images = val_images
        quickdraw_val_labels = val_labels
        quickdraw_train_images = train_images
        quickdraw_train_labels = train_labels
    else:
        current_class_names = list(DIGIT_CLASS_NAMES)
        current_slice_starts = {}

    current_mode = mode
    inference.reset()
    if not inference.reload(checkpoint_path_for_mode(mode), current_class_names):
        # No matching trained checkpoint - load a random-init model instead,
        # so there's always something live to draw against and see the
        # connection weights of right away. checkpoint_ready in
        # build_mode_selected_message() below stays keyed off is_trained, so
        # the UI still correctly says no training has happened yet.
        spec = ArchitectureSpec(
            kind="mlp",
            layers=LayerSpec(widths=list(current_layer_widths)),
            num_classes=len(current_class_names),
            class_names=list(current_class_names),
            dataset=dataset_for_mode(mode),
        )
        inference.load_random(spec)

    await broadcast(build_mode_selected_message())


@app.on_event("startup")
async def startup() -> None:
    global training_manager, mnist_test_images, mnist_test_labels, mnist_train_images, mnist_train_labels
    clear_checkpoints()
    loop = asyncio.get_running_loop()
    training_manager = TrainingManager(loop, handle_training_message)
    # The kiosk runs offline on the event day - pull in everything both
    # modes' training data needs now, while there's still a network, rather
    # than lazily on whatever a visitor happens to pick first. Both only do
    # real work the very first time the server starts (or, for Quick Draw,
    # after a new category is added to the pool); every later start finds it
    # all already cached on disk - see prewarm_mnist()/prewarm_categories()'s
    # docstrings. Blocks the server from accepting connections until done,
    # which is exactly what we want: "server is up" should mean "kiosk is
    # actually ready."
    await asyncio.to_thread(prewarm_mnist)
    await asyncio.to_thread(prewarm_categories, CATEGORY_POOL)
    mnist_test_images, mnist_test_labels = await asyncio.to_thread(get_raw_test_samples)
    mnist_train_images, mnist_train_labels = await asyncio.to_thread(get_raw_train_samples)
    # No mode chosen yet at boot (current_mode is None) - nothing to reload
    # here, select_mode() does the first reload once a visitor picks a mode.


@app.get("/help-messages")
async def get_help_messages() -> dict:
    """Static per-mode/pane/language help dialog copy - see
    server/help_messages.py. Fetched once by static/app.js on load rather
    than threaded through every mode_selected broadcast, since it never
    changes at runtime."""
    return HELP_MESSAGES


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    global last_pixels, current_layer_widths, current_mode, current_class_names, current_slice_starts
    await websocket.accept()
    clients.add(websocket)
    # Sync this (possibly late-joining, or a page reload) client to whatever
    # mode the kiosk is already in - shared kiosk state, same as broadcasting
    # classification/training progress to every connected client. Stays
    # silent if no mode has been chosen yet, so a true first connection sees
    # the menu rather than jumping straight into a default mode.
    if current_mode is not None:
        await websocket.send_text(json.dumps(build_mode_selected_message()))
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")

            if msg_type == "draw_update":
                pixels = msg.get("pixels")
                if current_mode is None or not isinstance(pixels, list):
                    continue
                draw_grid_size = draw_grid_size_for_mode(current_mode)
                if len(pixels) != draw_grid_size * draw_grid_size:
                    continue
                if current_mode == "drawings":
                    image = np.array(pixels, dtype=np.float32).reshape(draw_grid_size, draw_grid_size)
                    last_pixels = downscale_by_fill_count(image, DRAW_DOWNSCALE_FACTOR).flatten().tolist()
                else:
                    last_pixels = pixels
                await classify_and_broadcast()

            elif msg_type == "config_update":
                widths = msg.get("layers", [])
                if isinstance(widths, list) and all(isinstance(w, int) for w in widths):
                    current_layer_widths = clamp_layer_widths(widths, max_nodes_per_layer_for_mode(current_mode))
                    await websocket.send_text(
                        json.dumps({"type": "config_ack", "layers": current_layer_widths})
                    )

            elif msg_type == "select_mode":
                mode = msg.get("mode")
                if mode not in ("digits", "drawings"):
                    continue
                await select_mode(mode)

            elif msg_type == "retrain":
                if current_mode is None:
                    continue
                extra = {"slice_starts": current_slice_starts} if current_mode == "drawings" else {}
                spec = ArchitectureSpec(
                    kind="mlp",
                    layers=LayerSpec(widths=list(current_layer_widths)),
                    num_classes=len(current_class_names),
                    class_names=list(current_class_names),
                    dataset=dataset_for_mode(current_mode),
                    extra=extra,
                )
                assert training_manager is not None
                training_manager.start(spec, checkpoint_path_for_mode(current_mode))

            elif msg_type == "debug_load_sample":
                if current_mode == "drawings":
                    if quickdraw_val_images is None or quickdraw_val_labels is None or len(quickdraw_val_images) == 0:
                        continue
                    idx = random.randrange(len(quickdraw_val_images))
                    sample_pixels = quickdraw_val_images[idx].flatten().tolist()
                    sample_label = current_class_names[int(quickdraw_val_labels[idx])]
                else:
                    if mnist_test_images is None or mnist_test_labels is None:
                        continue
                    idx = random.randrange(len(mnist_test_images))
                    sample_pixels = mnist_test_images[idx].flatten().tolist()
                    sample_label = str(int(mnist_test_labels[idx]))
                last_pixels = sample_pixels

                sample_label_sv = class_names_sv_for([sample_label], current_mode)[0]
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "debug_sample",
                            "pixels": sample_pixels,
                            "label": sample_label,
                            "label_sv": sample_label_sv,
                        }
                    )
                )
                await classify_and_broadcast()

            elif msg_type == "get_title_help_samples":
                # Sent each time the title help dialog is opened (see
                # openHelp() in static/app.js) rather than baked into
                # mode_selected, precisely so reopening it reshuffles which
                # TITLE_HELP_SAMPLE_COUNT real training images are shown.
                if current_mode is None:
                    continue
                await websocket.send_text(
                    json.dumps({"type": "title_help_samples", "samples": pick_title_help_samples()})
                )

            elif msg_type == "explain_prediction":
                class_idx = msg.get("class_idx")
                if not isinstance(class_idx, int) or not (0 <= class_idx < len(current_class_names)):
                    continue
                if last_pixels is None:
                    continue
                saliency = inference.compute_saliency(last_pixels, class_idx)
                if saliency is None:
                    continue
                await broadcast(
                    {
                        "type": "saliency",
                        "class_idx": class_idx,
                        "map": [[float(v) for v in row] for row in saliency],
                    }
                )

            elif msg_type == "explain_node":
                layer = msg.get("layer")
                node = msg.get("node")
                if not isinstance(layer, int) or not isinstance(node, int):
                    continue
                if last_pixels is None:
                    continue
                saliency = inference.compute_node_saliency(last_pixels, layer, node)
                if saliency is None:
                    continue
                await broadcast(
                    {
                        "type": "node_saliency",
                        "layer": layer,
                        "node": node,
                        "map": [[float(v) for v in row] for row in saliency],
                    }
                )

            elif msg_type == "reset_kiosk":
                # Inactivity timeout fired client-side (see static/app.js) -
                # put the shared kiosk state back to "no mode chosen", same
                # as a fresh server boot, then tell every connected client to
                # reload so each one lands back on the menu. Deliberately
                # does NOT touch saved checkpoints on disk - a wandering-off
                # visitor's half-finished config gets discarded, but a
                # genuinely trained model stays available for whoever picks
                # that mode next.
                assert training_manager is not None
                training_manager.stop()
                last_pixels = None
                current_mode = None
                current_class_names = []
                current_slice_starts = {}
                current_layer_widths = list(DEFAULT_LAYER_WIDTHS)
                inference.reset()
                await broadcast({"type": "kiosk_reset"})

    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(websocket)


app.mount("/", StaticFiles(directory="static", html=True), name="static")
