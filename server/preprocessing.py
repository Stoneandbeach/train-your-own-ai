"""Preprocessing to match MNIST's own convention: digits in the dataset are
centered by center-of-mass within the 28x28 field. Live-drawn input isn't,
which was causing shape-sensitive digits (e.g. 8) to be misclassified even
though the model itself classifies real MNIST test images correctly."""

import numpy as np


def compute_center_shift(image: np.ndarray) -> tuple[int, int]:
    """image: (28, 28) float32, any range. Returns the whole-pixel (shift_y,
    shift_x) that would move image's center of mass to the center of the
    grid. (0, 0) for a blank image."""
    total = image.sum()
    if total <= 0:
        return 0, 0

    height, width = image.shape
    ys, xs = np.indices(image.shape)
    com_y = (ys * image).sum() / total
    com_x = (xs * image).sum() / total

    shift_y = int(round((height - 1) / 2 - com_y))
    shift_x = int(round((width - 1) / 2 - com_x))
    return shift_y, shift_x


def shift_image(image: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    """Shift image by (shift_y, shift_x) whole pixels, zero-padded, no
    wraparound. Used both to center a drawing before it reaches the model,
    and to shift a result (e.g. a saliency map) back into the drawing's own
    coordinate frame afterwards - shift_image(x, -shift_y, -shift_x) undoes
    shift_image(x, shift_y, shift_x)."""
    if shift_y == 0 and shift_x == 0:
        return image

    height, width = image.shape
    shifted = np.zeros_like(image)
    src_y0, src_y1 = max(0, -shift_y), min(height, height - shift_y)
    src_x0, src_x1 = max(0, -shift_x), min(width, width - shift_x)
    dst_y0, dst_y1 = max(0, shift_y), min(height, height + shift_y)
    dst_x0, dst_x1 = max(0, shift_x), min(width, width + shift_x)

    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return shifted


def center_by_mass(image: np.ndarray) -> np.ndarray:
    """image: (28, 28) float32, any range. Returns a copy shifted so its
    center of mass sits at the center of the grid."""
    shift_y, shift_x = compute_center_shift(image)
    return shift_image(image, shift_y, shift_x)


def downscale_by_fill_count(image: np.ndarray, block_size: int) -> np.ndarray:
    """image: (H, W) float32, H and W each divisible by block_size, where a
    pixel > 0 counts as "filled" (matches the draw canvas's own convention:
    0 = blank, up to 255 = inked). Returns an (H/block_size, W/block_size)
    array where each output pixel is 255 * (filled count in its block) /
    block_size**2 - a coverage-based downscale (how much of each block is
    inked) rather than an intensity average, so a corner-clipping stroke
    reads as a partial grey rather than either fully on or off."""
    height, width = image.shape
    out_h, out_w = height // block_size, width // block_size
    blocks = image.reshape(out_h, block_size, out_w, block_size)
    filled_counts = (blocks > 0).sum(axis=(1, 3))
    return (filled_counts / (block_size * block_size) * 255).astype(np.float32)
