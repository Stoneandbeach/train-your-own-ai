"""Kiosk-wide constants."""

MIN_NODES_PER_LAYER = 10
MAX_NODES_PER_LAYER = 40
NODE_STEP = 1
MIN_LAYERS = 0
MAX_LAYERS = 4

# Native MNIST images are 28x28; downsampled (by block-averaging) to this
# resolution everywhere - the drawing canvas, training data, and debug MNIST
# samples - so GRID_SIZE is the single source of truth for pixel resolution.
GRID_SIZE = 28
INPUT_SIZE = GRID_SIZE * GRID_SIZE
NUM_CLASSES = 10

BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3

# Deliberately small: with the full 60k-image MNIST training set, even a
# single 10-node hidden layer clears >90% val accuracy, so visitors can't see
# model capacity matter. Training on a small, fixed subset instead makes
# capacity visible: too little capacity underfits, too much overfits this
# small a set (no regularization is added on purpose - that tradeoff is part
# of the demo). The held-out validation set stays the full MNIST test split.
TRAIN_SUBSET_SIZE = 600
TRAIN_SUBSET_SEED = 42

# Push a training_metrics message every N batches, in addition to end-of-epoch.
METRICS_PUSH_EVERY_N_BATCHES = 20

# Separate checkpoint files per mode, always threaded explicitly through the
# call chain (never read from a shared global at the point of use) - so a
# Drawings-mode retrain can never overwrite the visitor's Digits-mode model
# or vice versa.
CHECKPOINT_PATH_DIGITS = "checkpoints/checkpoint_digits.pt"
CHECKPOINT_PATH_DRAWINGS = "checkpoints/checkpoint_drawings.pt"
DATA_ROOT = "data"

DEFAULT_LAYER_WIDTHS = [20, 20]

# Occlusion-based saliency ("why did the model predict this digit?"): side
# length in pixels of each square patch that gets zeroed out and re-predicted.
# 1 = full per-pixel resolution (INPUT_SIZE forward passes per map).
OCCLUSION_PATCH_SIZE = 1

# ---- Quick Draw ("Drawings" mode) ----
# Categories are range-fetched directly from Google's public bucket (each
# .npy file there is often 90MB+, so we only ever pull the header plus the
# exact byte slice needed - see server/quickdraw_data.py) and disk-cached
# under QUICKDRAW_DATA_ROOT, mirroring how DATA_ROOT caches MNIST.
QUICKDRAW_BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
QUICKDRAW_DATA_ROOT = "data/quickdraw"
QUICKDRAW_NUM_CLASSES = 8
# Same total training-set size as MNIST's TRAIN_SUBSET_SIZE, split evenly
# across classes, so the "small subset makes capacity visible" pedagogy
# transfers unchanged (see TRAIN_SUBSET_SIZE above) - it's the same lesson,
# just with a different dataset.
QUICKDRAW_TRAIN_PER_CLASS = TRAIN_SUBSET_SIZE // QUICKDRAW_NUM_CLASSES
QUICKDRAW_VAL_PER_CLASS = 100
QUICKDRAW_FETCH_TIMEOUT = 10  # seconds, per HTTP request

# Quick Draw's numpy_bitmap files vs. this project's own pixel convention
# (0 = blank, 255 = fully inked, as drawn on the canvas and used throughout
# server/data.py, server/preprocessing.py, static/app.js): set only after
# visually verifying a real fetched sample - see the verification script
# referenced in server/quickdraw_data.py's module docstring.
QUICKDRAW_INVERT_PIXELS = False
