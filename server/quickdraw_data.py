"""Quick Draw dataset loading/caching, mirroring server/data.py's contract
(get_data_loaders()-shaped, same (GRID_SIZE, GRID_SIZE) float32 [0, 1] tensor
convention) so server/train_worker.py can call either interchangeably based
on ArchitectureSpec.dataset.

Each category lives in Google's public bucket as a single .npy file of shape
(N, 784) uint8 - often 90MB+ for a popular category - so every fetch here
uses an HTTP Range request for just the bytes needed, never a full download.
See _parse_npy_header for the byte layout (confirmed against a live sample of
quickdraw_dataset/full/numpy_bitmap/cat.npy: NPY format v1.0, 6-byte magic +
2-byte version + 2-byte little-endian header-dict length at offset 8).

Caching is two-tier: the first time a category is used, its first
QUICKDRAW_CACHE_SIZE_PER_CATEGORY images are range-fetched once and saved to
disk as a single local block (see ensure_local_cache) - every slice pick
after that, this run or a future one, is served straight from that local
block with no network call at all, as long as it falls within it (it always
does - pick_slice_start() only ever picks within the cached block's bounds).

The kiosk runs offline on the actual event day, so server/main.py's startup
calls prewarm_categories() for the whole curated pool up front - only the
very first server start (or after adding a new category to the pool) does
any real downloading; every later start just finds every block already on
disk and skips straight past it.

Pixel-polarity verification (0=blank vs 0=inked, relative to this project's
own 0=blank/255=inked convention): run this module directly -
    .venv/bin/python -m server.quickdraw_data
- it fetches a few real "cat" images and prints an ASCII-art dump of one plus
a background-vs-ink pixel-value comparison, so QUICKDRAW_INVERT_PIXELS in
config.py can be set correctly before this module is wired into training.
"""

import ast
import json
import logging
import os
import random
import struct
import tempfile
import urllib.request
from urllib.parse import quote

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from server.config import (
    BATCH_SIZE,
    GRID_SIZE,
    QUICKDRAW_BASE_URL,
    QUICKDRAW_CACHE_SIZE_PER_CATEGORY,
    QUICKDRAW_DATA_ROOT,
    QUICKDRAW_FETCH_TIMEOUT,
    QUICKDRAW_INVERT_PIXELS,
    QUICKDRAW_TRAIN_PER_CLASS,
    QUICKDRAW_VAL_PER_CLASS,
)

logger = logging.getLogger(__name__)

# Quick Draw bitmaps are natively 28x28 with no resizing step here (unlike
# MNIST's downsampling path in data.py) - the user confirmed Drawings mode
# should just fit the existing 28x28 setup, so this module never needs to
# handle a different GRID_SIZE.
assert GRID_SIZE == 28, "Quick Draw bitmaps are natively 28x28; no downsampling is implemented here."

_BYTES_PER_IMAGE = GRID_SIZE * GRID_SIZE
_HEADER_CACHE_PATH = os.path.join(QUICKDRAW_DATA_ROOT, "_headers.json")
_INITIAL_HEADER_FETCH_BYTES = 512


def _npy_url(category: str) -> str:
    return f"{QUICKDRAW_BASE_URL}/{quote(category)}.npy"


def _fetch_range(url: str, start: int, end: int) -> bytes:
    """GETs bytes [start, end] (inclusive) via HTTP Range. Raises if the
    server doesn't honor the Range request (status != 206) - silently
    falling through to a full-body 200 response would mean downloading an
    entire, often 90MB+, Quick Draw file just to read a small slice."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=QUICKDRAW_FETCH_TIMEOUT) as resp:
        if resp.status != 206:
            raise RuntimeError(
                f"expected HTTP 206 Partial Content from {url}, got {resp.status} "
                "- server may not support range requests"
            )
        return resp.read()


def _parse_npy_header(header_bytes: bytes) -> tuple[int, tuple[int, ...]]:
    """Returns (data_offset, shape) for a .npy file given its leading bytes.
    Raises if header_bytes doesn't contain the full header (caller should
    re-fetch with a larger byte range)."""
    if header_bytes[:6] != b"\x93NUMPY":
        raise ValueError("not a .npy file (bad magic bytes)")
    major = header_bytes[6]
    if major == 1:
        header_len = struct.unpack_from("<H", header_bytes, 8)[0]
        data_offset = 10 + header_len
    else:
        header_len = struct.unpack_from("<I", header_bytes, 8)[0]
        data_offset = 12 + header_len
    if data_offset > len(header_bytes):
        raise ValueError("header_bytes too short - re-fetch with a larger range")

    header_text = header_bytes[data_offset - header_len : data_offset].decode("latin1").strip()
    header_dict = ast.literal_eval(header_text)
    if header_dict.get("descr") != "|u1":
        raise ValueError(f"unexpected dtype {header_dict.get('descr')!r} in Quick Draw .npy header")
    if header_dict.get("fortran_order"):
        raise ValueError("unexpected fortran-ordered Quick Draw .npy file")
    return data_offset, header_dict["shape"]


def _load_header_cache() -> dict:
    if not os.path.exists(_HEADER_CACHE_PATH):
        return {}
    try:
        with open(_HEADER_CACHE_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_header_cache(cache: dict) -> None:
    os.makedirs(QUICKDRAW_DATA_ROOT, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=QUICKDRAW_DATA_ROOT, suffix=".tmp")
    os.close(fd)
    with open(tmp_path, "w") as f:
        json.dump(cache, f)
    os.replace(tmp_path, _HEADER_CACHE_PATH)


def get_header_info(category: str) -> tuple[int, int]:
    """Returns (data_offset, total_images) for `category`, from the on-disk
    header cache if present, otherwise a small range-fetch (never the full
    file) that populates the cache for next time."""
    cache = _load_header_cache()
    if category in cache:
        entry = cache[category]
        return entry["data_offset"], entry["total"]

    url = _npy_url(category)
    chunk = _fetch_range(url, 0, _INITIAL_HEADER_FETCH_BYTES - 1)
    try:
        data_offset, shape = _parse_npy_header(chunk)
    except ValueError:
        # Header text didn't fit in the initial probe (very long shape/dtype
        # text) - re-fetch generously and try once more.
        chunk = _fetch_range(url, 0, _INITIAL_HEADER_FETCH_BYTES * 8 - 1)
        data_offset, shape = _parse_npy_header(chunk)

    if len(shape) != 2 or shape[1] != _BYTES_PER_IMAGE:
        raise ValueError(f"unexpected Quick Draw shape {shape} for category {category!r}")
    total = shape[0]

    cache[category] = {"data_offset": data_offset, "total": total}
    _save_header_cache(cache)
    return data_offset, total


def get_category_total(category: str) -> int:
    """Total images available for `category` (cheap after the first call -
    hits the on-disk header cache)."""
    _, total = get_header_info(category)
    return total


def _category_cache_dir(category: str) -> str:
    safe = category.replace(" ", "_").replace("/", "_")
    return os.path.join(QUICKDRAW_DATA_ROOT, safe)


def _local_cache_path(category: str) -> str:
    return os.path.join(_category_cache_dir(category), "cache.npy")


def ensure_local_cache(category: str) -> np.ndarray:
    """Returns this category's locally-cached image block: its first
    QUICKDRAW_CACHE_SIZE_PER_CATEGORY images, or all of them if it has fewer.
    Downloads and saves that block (one range-fetch, never the full category
    file) the first time this category is used; every call after that, this
    run or a future one, loads straight from disk - see the module
    docstring. pick_slice_start() and fetch_category_slice() both go through
    this, so neither ever needs the network once a category's block exists
    locally."""
    cache_path = _local_cache_path(category)
    if os.path.exists(cache_path):
        return np.load(cache_path)

    data_offset, total = get_header_info(category)
    cache_size = min(total, QUICKDRAW_CACHE_SIZE_PER_CATEGORY)
    byte_start = data_offset
    byte_end = data_offset + cache_size * _BYTES_PER_IMAGE - 1
    raw = _fetch_range(_npy_url(category), byte_start, byte_end)
    images = np.frombuffer(raw, dtype=np.uint8).reshape(cache_size, GRID_SIZE, GRID_SIZE).copy()

    os.makedirs(_category_cache_dir(category), exist_ok=True)
    np.save(cache_path, images)
    return images


def prewarm_categories(categories: list[str]) -> None:
    """Calls ensure_local_cache() for every category up front, so the kiosk
    can go fully offline afterward - see server/main.py's startup, which
    calls this for the whole curated pool. Already-cached categories cost
    just a file-exists check each, so a repeat call (every server restart
    after the first) is fast. A single category's fetch failing (no network,
    a renamed/missing category, ...) is logged and skipped rather than
    aborting the rest of the pool or crashing startup - better to come up
    with 49 of 50 categories ready than fail to start at all."""
    logger.info("prewarming %d Quick Draw categories...", len(categories))
    fetched = 0
    failed = []
    for i, category in enumerate(categories, start=1):
        cache_path = _local_cache_path(category)
        already_cached = os.path.exists(cache_path)
        try:
            ensure_local_cache(category)
        except Exception:
            logger.warning("failed to prewarm Quick Draw category %r", category, exc_info=True)
            failed.append(category)
            continue
        if not already_cached:
            fetched += 1
            logger.info("prewarmed %d/%d: %s", i, len(categories), category)
    logger.info(
        "Quick Draw prewarm done: %d newly downloaded, %d already cached, %d failed%s",
        fetched,
        len(categories) - fetched - len(failed),
        len(failed),
        f" ({failed})" if failed else "",
    )


def pick_slice_start(category: str, block_size: int) -> int:
    """A random start offset such that [start, start + block_size) is valid
    for `category`, within its locally-cached block (downloading that block
    first if this category hasn't been used yet - see ensure_local_cache).
    Unseeded - callers wanting a fresh slice each time (e.g. a Drawings-mode
    reroll) get one; this module has no opinion on when a fresh draw is
    warranted, that's the orchestrator's call."""
    cached_total = len(ensure_local_cache(category))
    if cached_total < block_size:
        raise ValueError(f"category {category!r} only has {cached_total} cached images, need {block_size}")
    return random.randrange(0, cached_total - block_size + 1)


def pick_slice_starts(categories: list[str]) -> dict[str, int]:
    """Convenience: one pick_slice_start() per category, sized for the
    combined train+val block used by get_quickdraw_data_loaders /
    get_quickdraw_raw_val_samples. Callers MUST reuse the same dict across
    both functions for a given (categories, "session") - that's what
    guarantees the debug 'load a real sample' button can never show an image
    that training also saw, without this module needing to track any state
    itself. See those functions' docstrings."""
    block = QUICKDRAW_TRAIN_PER_CLASS + QUICKDRAW_VAL_PER_CLASS
    return {category: pick_slice_start(category, block) for category in categories}


def fetch_category_slice(category: str, start: int, count: int) -> np.ndarray:
    """Returns (count, GRID_SIZE, GRID_SIZE) uint8 images for `category`,
    starting at image index `start` - sliced out of this category's locally-
    cached block (see ensure_local_cache), downloading that block first if
    this category hasn't been used yet."""
    cached = ensure_local_cache(category)
    if start + count > len(cached):
        raise ValueError(
            f"requested images [{start}:{start + count}) but only {len(cached)} are cached for {category!r}"
        )
    return cached[start : start + count]


def _to_float_tensor(images: np.ndarray) -> torch.Tensor:
    """images: (N, 28, 28) uint8. Returns (N, 28, 28) float32 in [0, 1],
    matching the convention used for live-drawn pixels and MNIST training
    data (server/data.py, server/inference.py): 0 = blank, 1 = fully inked."""
    arr = images.astype(np.float32) / 255.0
    if QUICKDRAW_INVERT_PIXELS:
        arr = 1.0 - arr
    return torch.from_numpy(arr)


class _QuickDrawTensorDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self._images = _to_float_tensor(images)
        self._labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int):
        return self._images[idx], self._labels[idx]


def _split_train_val(categories: list[str], slice_starts: dict[str, int]) -> tuple[
    list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]
]:
    """Shared fetch+split logic for get_quickdraw_data_loaders and
    get_quickdraw_raw_val_samples, so the two can never disagree about where
    the train/val boundary falls within a slice."""
    block = QUICKDRAW_TRAIN_PER_CLASS + QUICKDRAW_VAL_PER_CLASS
    train_images, train_labels, val_images, val_labels = [], [], [], []
    for label_idx, category in enumerate(categories):
        images = fetch_category_slice(category, slice_starts[category], block)
        train_images.append(images[:QUICKDRAW_TRAIN_PER_CLASS])
        val_images.append(images[QUICKDRAW_TRAIN_PER_CLASS:])
        train_labels.append(np.full(QUICKDRAW_TRAIN_PER_CLASS, label_idx, dtype=np.int64))
        val_labels.append(np.full(QUICKDRAW_VAL_PER_CLASS, label_idx, dtype=np.int64))
    return train_images, train_labels, val_images, val_labels


def get_quickdraw_data_loaders(categories: list[str], slice_starts: dict[str, int]) -> tuple[DataLoader, DataLoader]:
    """categories[i] is the class label for index i. slice_starts must have
    been produced by pick_slice_starts(categories) (or otherwise agree with
    its sizing) - the caller owns picking these, so that
    get_quickdraw_raw_val_samples can be called separately with the same
    slice_starts and see exactly the same held-out validation images this
    function used, with no possibility of the two drifting apart."""
    train_images, train_labels, val_images, val_labels = _split_train_val(categories, slice_starts)
    train_set = _QuickDrawTensorDataset(np.concatenate(train_images), np.concatenate(train_labels))
    val_set = _QuickDrawTensorDataset(np.concatenate(val_images), np.concatenate(val_labels))

    train_loader = DataLoader(train_set, batch_size=min(BATCH_SIZE, len(train_set)), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def get_quickdraw_raw_val_samples(categories: list[str], slice_starts: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Mirrors data.py's get_raw_test_samples(): returns the held-out val
    portion of the same slice get_quickdraw_data_loaders(categories,
    slice_starts) would use, as (images, labels) - uint8 (N, 28, 28) images
    in the raw 0=blank/255=inked convention (NOT inverted even if
    QUICKDRAW_INVERT_PIXELS is set - that flag only affects the float
    tensors fed to the model, same as how draw-canvas pixels are always sent
    raw) and int64 (N,) labels indexing into `categories`. Used by the
    'load a real sample' debug button's Quick Draw analog."""
    _, _, val_images, val_labels = _split_train_val(categories, slice_starts)
    return np.concatenate(val_images), np.concatenate(val_labels)


def _ascii_preview(image: np.ndarray) -> str:
    ramp = " .:-=+*#%@"
    lines = []
    for row in image:
        lines.append("".join(ramp[min(len(ramp) - 1, int(v) * len(ramp) // 256)] for v in row))
    return "\n".join(lines)


if __name__ == "__main__":
    # Standalone polarity-verification probe - see module docstring.
    category = "cat"
    print(f"fetching a small slice of {category!r} to check header parsing + pixel polarity...")
    total = get_category_total(category)
    print(f"{category!r} has {total} images (fetched via header range-request only)")
    images = fetch_category_slice(category, 0, 5)
    print(f"fetched slice shape={images.shape} dtype={images.dtype}")
    sample = images[0]
    print(f"corner (should be background) mean: {sample[:3, :3].mean():.1f}")
    print(f"center 8x8 (likely ink) mean: {sample[10:18, 10:18].mean():.1f}")
    print()
    print(_ascii_preview(sample))
