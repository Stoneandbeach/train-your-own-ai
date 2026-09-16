"""Subprocess entrypoint: trains a fresh model to completion (or until
stopped), reporting progress via a multiprocessing.Queue and checkpointing
to disk after every epoch.

Must be import-safe at module scope (no globals touching torch state) since
it is spawned via multiprocessing's "spawn" start method.
"""

import os
import tempfile

from server.config import EARLY_STOPPING_PATIENCE, METRICS_PUSH_EVERY_N_BATCHES, NUM_EPOCHS
from server.data import get_data_loaders
from server.mlp_classifier import MLPClassifier, default_device
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
    # Defined before the try block so the except handler below can always
    # report them, even if training never got past setup (e.g. a data-loading
    # error before epoch 0).
    epochs_trained = 0
    best_val_accuracy = None
    train_size = None
    val_size = None

    try:
        if spec.dataset == "quickdraw":
            slice_starts = spec.extra["slice_starts"]
            train_loader, test_loader = get_quickdraw_data_loaders(spec.class_names, slice_starts)
        else:
            train_loader, test_loader = get_data_loaders()
        train_size = len(train_loader.dataset)
        val_size = len(test_loader.dataset)
        device = default_device()
        print(f"[train_worker] training on device: {device}")
        classifier = MLPClassifier(train_loader=train_loader, test_loader=test_loader, device=device)
        classifier.configure(spec)

        metrics_queue.put({"type": "training_status", "state": "running"})

        # Early stopping on validation loss: the checkpoint on disk only ever
        # gets (over)written on a new best val_loss, so it always holds the
        # best epoch's weights, never a worse one from after that peak - the
        # whole point of early stopping. NUM_EPOCHS is just the upper bound;
        # in the common case EARLY_STOPPING_PATIENCE ends the run first.
        best_val_loss = float("inf")
        epochs_without_improvement = 0

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
            epochs_trained += 1

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

            # No val_loss to compare (test_loader not supplied) is not a real
            # code path today - train_worker always supplies one - but if it
            # ever happened, treat it as "no early-stopping signal" rather
            # than silently never checkpointing again.
            improved = final_metrics.val_loss is None or final_metrics.val_loss < best_val_loss
            if improved:
                if final_metrics.val_loss is not None:
                    best_val_loss = final_metrics.val_loss
                best_val_accuracy = final_metrics.accuracy
                epochs_without_improvement = 0
                _atomic_save(classifier, checkpoint_path)
                metrics_queue.put(
                    {
                        "type": "checkpoint_loaded",
                        "epoch": final_metrics.epoch,
                        "checkpoint_path": checkpoint_path,
                    }
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    break

        # Mirrors the three ways the loop above can end: stop_event short-
        # circuits every other break/exit check the moment it's set (both the
        # top-of-loop check and the _StopTraining exception from on_batch), so
        # checking it first here unambiguously identifies that case; otherwise
        # the patience counter tells early-stopping apart from simply running
        # out of epochs.
        if stop_event.is_set():
            stop_reason = "stopped_by_user"
        elif epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            stop_reason = "early_stopping"
        else:
            stop_reason = "max_epochs"

        metrics_queue.put(
            {
                "type": "training_status",
                "state": "stopped" if stop_event.is_set() else "idle",
                "epochs_trained": epochs_trained,
                "best_val_accuracy": best_val_accuracy,
                "stop_reason": stop_reason,
                "train_size": train_size,
                "val_size": val_size,
            }
        )
    except Exception as exc:  # surface errors to the UI instead of a silent death
        metrics_queue.put(
            {
                "type": "training_status",
                "state": "error",
                "message": str(exc),
                "epochs_trained": epochs_trained,
                "best_val_accuracy": best_val_accuracy,
                "train_size": train_size,
                "val_size": val_size,
            }
        )
        raise


class _StopTraining(Exception):
    pass
