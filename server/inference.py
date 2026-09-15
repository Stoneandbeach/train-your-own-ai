"""Holds the current in-memory model used for live classification, reloaded
from the latest checkpoint whenever training signals a new one is ready.

Mode-agnostic: this module has no idea "Digits" or "Drawings" exist - it just
loads whatever checkpoint path it's given and checks its embedded class
names against whatever the caller currently expects. server/main.py owns the
mode concept and decides which path/class-names to pass in."""

import logging
import os
from typing import Optional

import numpy as np

from server.config import GRID_SIZE
from server.mlp_classifier import MLPClassifier
from server.model_interface import ArchitectureSpec
from server.preprocessing import center_by_mass, compute_center_shift, shift_image
from server.saliency import ContrastiveOcclusionSaliency, SaliencyMethod

logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(self):
        self._classifier = MLPClassifier()
        self._loaded = False
        # True only after a real checkpoint loads via reload() - distinct
        # from _loaded, which load_random() below also sets, so callers (see
        # is_trained/is_ready) can tell "there's a model I can run inference
        # against" apart from "that model actually learned anything."
        self._is_trained = False
        # Swap this for a different SaliencyMethod implementation (gradient x
        # input, integrated gradients, LRP, ...) to change the explanation
        # method without touching anything downstream.
        self._saliency_method: SaliencyMethod = ContrastiveOcclusionSaliency()

    def reset(self) -> None:
        """Marks no model as loaded, without touching anything on disk - used
        when switching modes/classes right before a reload() attempt, so a
        stale model from the previous mode/draw is never mistakenly served
        for a beat while the new checkpoint is being resolved."""
        self._loaded = False
        self._is_trained = False

    def reload(self, path: str, expected_class_names: list[str]) -> bool:
        """Attempts to load the checkpoint at `path`, requiring its embedded
        class_names to exactly match `expected_class_names`. Returns True iff
        a matching model is now loaded and ready.

        Never raises: a missing file, a tensor-shape mismatch (e.g. a stale
        checkpoint saved under a different architecture/dataset), or a
        class-name mismatch (e.g. a Drawings checkpoint left over from a
        previous random 8-class draw) are all just "not ready" outcomes, not
        crashes. This project previously had an unhandled RuntimeError from a
        shape mismatch take down the entire server at startup - this is the
        fix, generalized to also cover same-shape-different-classes."""
        self._loaded = False
        self._is_trained = False
        if not os.path.exists(path):
            return False
        try:
            self._classifier.load_checkpoint(path)
        except Exception:
            logger.warning("failed to load checkpoint %s", path, exc_info=True)
            return False

        spec = self._classifier.spec
        if spec is None or list(spec.class_names) != list(expected_class_names):
            logger.info(
                "checkpoint %s class_names %s don't match expected %s - not loading",
                path,
                spec.class_names if spec else None,
                expected_class_names,
            )
            return False

        self._loaded = True
        self._is_trained = True
        return True

    def load_random(self, spec: ArchitectureSpec) -> None:
        """Configures a freshly random-initialized (untrained) model matching
        spec directly in memory - no checkpoint file involved. Called right
        after a mode is selected whenever reload() didn't find a matching
        trained checkpoint, so visitors always have a live (if nonsensical)
        model to classify against and see the connection weights of
        immediately, instead of a blank pane until the first Train
        completes. is_trained stays False either way, so the UI still says a
        real training run hasn't happened yet - see is_trained below."""
        self._classifier.configure(spec)
        self._loaded = True
        self._is_trained = False

    def predict(self, pixels: list[int]) -> Optional[np.ndarray]:
        if not self._loaded:
            return None
        image = np.array(pixels, dtype=np.float32).reshape(GRID_SIZE, GRID_SIZE) / 255.0
        image = center_by_mass(image)
        return self._classifier.predict(image)

    def predict_with_activations(
        self, pixels: list[int]
    ) -> tuple[Optional[np.ndarray], Optional[list[np.ndarray]]]:
        if not self._loaded:
            return None, None
        image = np.array(pixels, dtype=np.float32).reshape(GRID_SIZE, GRID_SIZE) / 255.0
        image = center_by_mass(image)
        return self._classifier.predict_with_activations(image)

    def get_edge_weights(self) -> list:
        if not self._loaded:
            return []
        return self._classifier.get_edge_weights()

    def compute_saliency(self, pixels: list[int], target_class: int) -> Optional[np.ndarray]:
        if not self._loaded:
            return None
        raw_image = np.array(pixels, dtype=np.float32).reshape(GRID_SIZE, GRID_SIZE) / 255.0
        shift_y, shift_x = compute_center_shift(raw_image)
        image = shift_image(raw_image, shift_y, shift_x)
        score_fn = lambda img: float(self._classifier.predict(img)[target_class])
        saliency = self._saliency_method.compute(score_fn, image)
        # The map was computed against the centered image; shift it back so
        # it lines up with the drawing as the visitor actually drew it.
        return shift_image(saliency, -shift_y, -shift_x)

    def compute_node_saliency(
        self, pixels: list[int], layer_idx: int, node_idx: int
    ) -> Optional[np.ndarray]:
        if not self._loaded:
            return None
        raw_image = np.array(pixels, dtype=np.float32).reshape(GRID_SIZE, GRID_SIZE) / 255.0
        shift_y, shift_x = compute_center_shift(raw_image)
        image = shift_image(raw_image, shift_y, shift_x)
        _, activations = self._classifier.predict_with_activations(image)
        if activations is None:
            return None
        if not (0 <= layer_idx < len(activations)):
            return None
        if not (0 <= node_idx < len(activations[layer_idx])):
            return None

        def score_fn(img: np.ndarray) -> float:
            _, acts = self._classifier.predict_with_activations(img)
            return float(acts[layer_idx][node_idx])

        saliency = self._saliency_method.compute(score_fn, image)
        return shift_image(saliency, -shift_y, -shift_x)

    @property
    def is_ready(self) -> bool:
        """Is there a model in memory to run inference against at all -
        True for both a real trained checkpoint and a random init from
        load_random(). Gates predict()/get_edge_weights()/etc above."""
        return self._loaded

    @property
    def is_trained(self) -> bool:
        """Is the currently-loaded model an actual trained checkpoint, not
        just a random init - what server/main.py's checkpoint_ready field
        (and so the "no trained model yet" UI text) reflects."""
        return self._loaded and self._is_trained
