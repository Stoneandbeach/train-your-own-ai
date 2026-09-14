"""Subprocess entrypoint: trains a fresh model to completion (or until
stopped), reporting progress via a multiprocessing.Queue and checkpointing
to disk after every epoch.

Must be import-safe at module scope (no globals touching torch state) since
it is spawned via multiprocessing's "spawn" start method.
"""

import os
import tempfile

from server.config import METRICS_PUSH_EVERY_N_BATCHES, NUM_EPOCHS
from server.data import get_data_loaders
from server.mlp_classifier import MLPClassifier
from server.model_interface import ArchitectureSpec, Metrics
from server.quickdraw_data import get_quickdraw_data_loaders


def _atomic_save(classifier: MLPClassifier, path: str) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    os.close(fd)
    try:
        classifier.save_checkpoint(tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def run(arch_spec_dict: dict, metrics_queue, stop_event, checkpoint_path: str) -> None:
    """Entrypoint for the training subprocess. checkpoint_path is passed in
    explicitly (never read from a shared global) so this subprocess can only
    ever write to the exact file its caller named - the structural guarantee
    that a Drawings-mode run can't clobber the Digits-mode checkpoint or vice
    versa."""
    spec = ArchitectureSpec.from_dict(arch_spec_dict)

    try:
        if spec.dataset == "quickdraw":
            slice_starts = spec.extra["slice_starts"]
            train_loader, test_loader = get_quickdraw_data_loaders(spec.class_names, slice_starts)
        else:
            train_loader, test_loader = get_data_loaders()
        classifier = MLPClassifier(train_loader=train_loader, test_loader=test_loader)
        classifier.configure(spec)

        metrics_queue.put({"type": "training_status", "state": "running"})

        for epoch_idx in range(NUM_EPOCHS):
            if stop_event.is_set():
                break

            def on_batch(m: Metrics, epoch_idx=epoch_idx):
                if m.batch % METRICS_PUSH_EVERY_N_BATCHES == 0:
                    metrics_queue.put(
                        {
                            "type": "training_metrics",
                            "epoch": m.epoch,
                            "batch": m.batch,
                            "loss": m.loss,
                            "accuracy": m.accuracy,
                            "total_epochs": NUM_EPOCHS,
                        }
                    )
                if stop_event.is_set():
                    raise _StopTraining()

            try:
                final_metrics = classifier.train_epoch(epoch_idx, on_batch=on_batch)
            except _StopTraining:
                break

            _atomic_save(classifier, checkpoint_path)

            metrics_queue.put(
                {
                    "type": "training_metrics",
                    "epoch": final_metrics.epoch,
                    "batch": final_metrics.batch,
                    "loss": final_metrics.loss,
                    "val_loss": final_metrics.val_loss,
                    "accuracy": final_metrics.accuracy,
                    "total_epochs": NUM_EPOCHS,
                }
            )
            metrics_queue.put(
                {
                    "type": "checkpoint_loaded",
                    "epoch": final_metrics.epoch,
                    "checkpoint_path": checkpoint_path,
                }
            )

        metrics_queue.put(
            {
                "type": "training_status",
                "state": "stopped" if stop_event.is_set() else "idle",
            }
        )
    except Exception as exc:  # surface errors to the UI instead of a silent death
        metrics_queue.put({"type": "training_status", "state": "error", "message": str(exc)})
        raise


class _StopTraining(Exception):
    pass
