"""Architecture-agnostic interface between the orchestration layer and any
concrete classifier implementation (MLP today, potentially CNN/BDT later).

Nothing outside this module and a concrete implementation (e.g.
mlp_classifier.py) should need to know what kind of model is actually
running.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

import numpy as np

from server.config import NUM_CLASSES


@dataclass
class LayerSpec:
    """Hidden layer widths only. Input size is implied by GRID_SIZE and is
    not part of the user-configurable spec."""

    widths: list[int]


@dataclass
class ArchitectureSpec:
    kind: str  # "mlp" (future: "cnn", "bdt", ...)
    layers: LayerSpec
    # Which task this model was/will be trained for. class_names[i] is the
    # human-readable label for output index i (["0", ..., "9"] for Digits,
    # a Quick Draw category name per index for Drawings) - it rides inside
    # the saved checkpoint (see mlp_classifier.save_checkpoint) so a reload
    # carries its own labels without the caller having to remember them.
    num_classes: int = NUM_CLASSES
    class_names: list[str] = field(default_factory=list)
    dataset: str = "mnist"  # "mnist" | "quickdraw"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ArchitectureSpec":
        num_classes = d.get("num_classes", NUM_CLASSES)
        class_names = d.get("class_names") or [str(i) for i in range(num_classes)]
        return ArchitectureSpec(
            kind=d["kind"],
            layers=LayerSpec(widths=list(d["layers"]["widths"])),
            num_classes=num_classes,
            class_names=list(class_names),
            dataset=d.get("dataset", "mnist"),
            extra=dict(d.get("extra", {})),
        )


@dataclass
class Metrics:
    epoch: int
    batch: int
    loss: float
    accuracy: Optional[float] = None
    val_loss: Optional[float] = None
    done: bool = False


class Classifier(ABC):
    """Common interface any trainable digit classifier must implement."""

    @abstractmethod
    def configure(self, spec: ArchitectureSpec) -> None:
        """(Re)build the model fresh (random init) from the given spec."""

    @abstractmethod
    def train_epoch(
        self, epoch_idx: int, on_batch: Optional[Callable[[Metrics], None]] = None
    ) -> Metrics:
        """Run one full epoch over the training set. Calls on_batch(metrics)
        periodically for live progress reporting. Returns end-of-epoch metrics."""

    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """image: (28, 28) float32 array in [0, 1]. Returns: (num_classes,)
        probability vector summing to ~1."""

    @abstractmethod
    def predict_with_activations(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Like predict(), but also returns per-hidden-layer post-activation
        values (one 1D array per hidden layer, in order), for visualization."""

    @abstractmethod
    def save_checkpoint(self, path: str) -> None: ...

    @abstractmethod
    def load_checkpoint(self, path: str) -> None: ...

    @abstractmethod
    def get_metrics(self) -> Metrics: ...
