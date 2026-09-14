"""Pluggable "what made this happen?" explanation methods.

Each SaliencyMethod attributes an importance score to every input pixel for
one scalar "score" of interest - an output class's probability, or a hidden
node's activation, or anything else expressible as image -> float. Callers
bake the target into a ScoreFn closure, so implementations only need
black-box access to that function, not to the classifier's internals - a
method works with any Classifier implementation (MLP today, potentially
CNN/BDT later) and against any score, without depending on internals.
Swapping in a different method (gradient-based, integrated gradients, LRP,
...) later is a one-line change in InferenceService.
"""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from server.config import OCCLUSION_PATCH_SIZE

# (28, 28) float32 image in [0, 1] -> a single scalar score.
ScoreFn = Callable[[np.ndarray], float]


class SaliencyMethod(ABC):
    @abstractmethod
    def compute(self, score_fn: ScoreFn, image: np.ndarray) -> np.ndarray:
        """image: (28, 28) float32 in [0, 1]. Returns a (28, 28) float32 array
        of raw (un-normalized, signed) importance scores, same shape as the
        input: how much each pixel contributed to score_fn(image)."""


class ContrastiveOcclusionSaliency(SaliencyMethod):
    """For each patch, contrasts "fully inked" against "fully blank" and
    measures the resulting swing in the score. Deliberately ignores whatever
    value the patch currently holds - the only thing about the current
    drawing that still matters is the rest of the image (the context each
    patch is judged against) and, upstream, wherever center-of-mass
    recentering places that context. Positive (red) = drawing ink there
    would raise the score; negative (blue) = it would lower it."""

    def __init__(self, patch_size: int = OCCLUSION_PATCH_SIZE):
        self.patch_size = patch_size

    def compute(self, score_fn: ScoreFn, image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        p = self.patch_size
        result = np.zeros_like(image)
        for y in range(0, h, p):
            for x in range(0, w, p):
                inked = image.copy()
                inked[y : y + p, x : x + p] = 1.0
                blank = image.copy()
                blank[y : y + p, x : x + p] = 0.0
                result[y : y + p, x : x + p] = score_fn(inked) - score_fn(blank)
        return result
