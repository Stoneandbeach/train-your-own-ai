"""Kiosk-wide constants."""

MIN_NODES_PER_LAYER = 10
# Drawings mode allows twice the hidden-layer capacity of Digits - see
# server/main.py's max_nodes_per_layer_for_mode() and static/app.js's mirror
# of these two constants, which also scales the on-screen node size/spacing
# down by the same ratio so the config panel packs tighter without growing.
MAX_NODES_PER_LAYER_DIGITS = 40
MAX_NODES_PER_LAYER_DRAWINGS = 80
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
# Upper bound only - in practice training stops earlier via early stopping
# (EARLY_STOPPING_PATIENCE below), which is what actually governs run length.
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3

# Stop training once EARLY_STOPPING_PATIENCE consecutive epochs pass with no
# improvement in validation loss - see server/train_worker.py. The checkpoint
# on disk always holds the best (lowest val_loss) epoch's weights, not
# whatever the most recent epoch happened to produce.
EARLY_STOPPING_PATIENCE = 3

# Dropout probability applied after each hidden layer's ReLU, Drawings mode
# only - see server/mlp_classifier.py's build_mlp()/dropout_rate_for_dataset().
# Digits mode stays at implicit 0.0 (a true no-op for nn.Dropout), so its
# capacity-vs-overfitting demo (see TRAIN_SUBSET_SIZE below) is unchanged.
# 0.3 rather than the more common 0.5: layers can be configured as narrow as
# MIN_NODES_PER_LAYER (10 nodes), where dropping half of an already-small
# layer per forward pass during training is aggressive; 0.3 still meaningfully
# regularizes the small (600-image) training set without crippling that end
# of the configurable range.
DRAWINGS_DROPOUT_RATE = 0.3

# Deliberately small: with the full 60k-image MNIST training set, even a
# single 10-node hidden layer clears >90% val accuracy, so visitors can't see
# model capacity matter. Training on a small, fixed subset instead makes
# capacity visible: too little capacity underfits, too much overfits this
# small a set. In Digits mode no regularization is added on purpose - that
# tradeoff is part of the demo; Drawings mode gets dropout (see
# DRAWINGS_DROPOUT_RATE above), which softens but doesn't eliminate it there.
# The held-out validation set stays the full MNIST test split.
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

DEFAULT_LAYER_WIDTHS = [10]

# Drawings mode only: the live draw canvas is this many times finer than the
# model's own GRID_SIZE (thinner strokes, more precision), block-downscaled
# by fill-count before ever reaching the model - see
# server/preprocessing.py's downscale_by_fill_count and server/main.py's
# draw_update handler. Digits mode draws natively at GRID_SIZE, untouched.
DRAW_DOWNSCALE_FACTOR = 2

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

# How many images per category to download and cache locally the first time
# that category is picked - every draw after that (this run or a future one)
# picks a random slice within this cached block and reads it straight from
# disk, never touching the network again for that category (see
# server/quickdraw_data.py's _ensure_local_cache). Comfortably above
# QUICKDRAW_TRAIN_PER_CLASS + QUICKDRAW_VAL_PER_CLASS (175) so repeated
# draws of the same category still land on different images, while staying
# a small fetch (~4MB at 784 bytes/image) even for the most popular
# categories, which can otherwise run 90MB+ in full.
QUICKDRAW_CACHE_SIZE_PER_CATEGORY = 5000

# Quick Draw's numpy_bitmap files vs. this project's own pixel convention
# (0 = blank, 255 = fully inked, as drawn on the canvas and used throughout
# server/data.py, server/preprocessing.py, static/app.js): set only after
# visually verifying a real fetched sample - see the verification script
# referenced in server/quickdraw_data.py's module docstring.
QUICKDRAW_INVERT_PIXELS = False

# Hand-curated (with the user) subset of Quick Draw categories to draw random
# classes from - see server/quickdraw_categories.py, which loads this CSV at
# import time rather than hardcoding the pool, so re-curating (including
# adding languages beyond the "english" column it currently reads) is just an
# edit to this file. Path is relative to the working directory the server is
# run from (repo root - see README), matching DATA_ROOT/QUICKDRAW_DATA_ROOT above.
QUICKDRAW_CURATED_CATEGORIES_CSV = "quickdraw_curated_list.csv"
