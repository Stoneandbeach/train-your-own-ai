"""MNIST dataset loading/caching."""

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from server.config import BATCH_SIZE, DATA_ROOT, GRID_SIZE, TRAIN_SUBSET_SEED, TRAIN_SUBSET_SIZE

logger = logging.getLogger(__name__)


def _downsample(image: np.ndarray, size: int) -> np.ndarray:
    """Block-average a square (H, W) array down to (size, size). H and W
    must be exact multiples of size - true for native 28x28 MNIST images
    with any GRID_SIZE that divides 28 (14, 7, ...)."""
    h, w = image.shape
    return image.reshape(size, h // size, size, w // size).mean(axis=(1, 3))


def _to_tensor(pic) -> torch.Tensor:
    """Convert a PIL MNIST image to a (GRID_SIZE, GRID_SIZE) float32 tensor
    in [0, 1], matching the normalization used for live-drawn pixels in
    inference.py."""
    arr = np.array(pic, dtype=np.float32) / 255.0
    arr = _downsample(arr, GRID_SIZE)
    return torch.from_numpy(arr)


def _mnist_already_present() -> bool:
    try:
        datasets.MNIST(root=DATA_ROOT, train=True, download=False)
        return True
    except RuntimeError:
        return False


def prewarm_mnist() -> None:
    """Ensures MNIST is downloaded and cached locally under DATA_ROOT - a
    no-op if it already is. Mirrors server/quickdraw_data.py's
    prewarm_categories() for the same reason (the kiosk runs offline on the
    event day - see server/main.py's startup, which calls this before it
    ever calls get_raw_test_samples()/get_data_loaders() below). One
    download=True call covers both splits: torchvision's MNIST.download()
    always fetches all four raw files (train + test, images + labels)
    together regardless of which split `train=` requests - the split only
    affects what gets loaded into memory afterward, not what gets
    downloaded."""
    already_present = _mnist_already_present()
    datasets.MNIST(root=DATA_ROOT, train=True, download=True)
    logger.info("MNIST prewarm done: %s", "already cached" if already_present else "downloaded")


def get_data_loaders() -> tuple[DataLoader, DataLoader]:
    transform = transforms.Lambda(_to_tensor)

    train_set = datasets.MNIST(
        root=DATA_ROOT, train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root=DATA_ROOT, train=False, download=True, transform=transform
    )

    generator = torch.Generator().manual_seed(TRAIN_SUBSET_SEED)
    subset_indices = torch.randperm(len(train_set), generator=generator)[:TRAIN_SUBSET_SIZE].tolist()
    train_set = Subset(train_set, subset_indices)

    train_loader = DataLoader(
        train_set, batch_size=min(BATCH_SIZE, TRAIN_SUBSET_SIZE), shuffle=True
    )
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


def get_raw_test_samples() -> tuple[np.ndarray, np.ndarray]:
    """Returns (images, labels) for the MNIST test split, unnormalized:
    images is (N, GRID_SIZE, GRID_SIZE) uint8 in [0, 255] (same convention as
    the pixels sent by the draw canvas), labels is (N,) int64. Used by
    debug-mode "load a real test image" tooling."""
    test_set = datasets.MNIST(root=DATA_ROOT, train=False, download=True)
    raw_images = test_set.data.numpy()  # uint8, (N, 28, 28)
    images = np.stack(
        [_downsample(img.astype(np.float32), GRID_SIZE) for img in raw_images]
    )
    images = images.round().astype(np.uint8)
    labels = test_set.targets.numpy()  # int64, (N,)
    return images, labels


def get_raw_train_samples() -> tuple[np.ndarray, np.ndarray]:
    """Returns (images, labels) for the exact TRAIN_SUBSET_SIZE-image subset
    get_data_loaders() actually trains on - same TRAIN_SUBSET_SEED, so always
    the same indices - unnormalized, same convention as get_raw_test_samples()
    above. Used by the title help dialog's sample grid: showing the real
    training images (not just "some MNIST images") is the point, matching
    what a visitor's Retrain click would actually use."""
    train_set = datasets.MNIST(root=DATA_ROOT, train=True, download=True)
    raw_images = train_set.data.numpy()  # uint8, (N, 28, 28)
    labels = train_set.targets.numpy()  # int64, (N,)

    generator = torch.Generator().manual_seed(TRAIN_SUBSET_SEED)
    subset_indices = torch.randperm(len(raw_images), generator=generator)[:TRAIN_SUBSET_SIZE].numpy()

    images = np.stack(
        [_downsample(img.astype(np.float32), GRID_SIZE) for img in raw_images[subset_indices]]
    )
    images = images.round().astype(np.uint8)
    return images, labels[subset_indices]
