"""MLP implementation of the Classifier interface."""

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from server.config import DRAWINGS_DROPOUT_RATE, INPUT_SIZE
from server.model_interface import ArchitectureSpec, Classifier, Metrics


def default_device() -> torch.device:
    """CUDA if available, else CPU. Used by the training subprocess (see
    server/train_worker.py) - deliberately NOT by live inference, which stays
    on CPU: it runs synchronously on the server's single asyncio event loop
    (server/main.py's classify_and_broadcast, called straight off each
    draw_update), so a first CUDA call there - context init, kernel launch -
    could stall every connected client, for a model this tiny CPU already
    runs instantly anyway."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dropout_rate_for_dataset(dataset: str) -> float:
    """0.0 (a true no-op for nn.Dropout) for everything but Quick Draw - see
    DRAWINGS_DROPOUT_RATE in server/config.py for why Drawings mode alone
    gets dropout."""
    return DRAWINGS_DROPOUT_RATE if dataset == "quickdraw" else 0.0


def build_mlp(widths: list[int], num_classes: int, dropout_rate: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Flatten()]
    in_features = INPUT_SIZE
    for w in widths:
        layers.append(nn.Linear(in_features, w))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        in_features = w
    layers.append(nn.Linear(in_features, num_classes))
    return nn.Sequential(*layers)


class MLPClassifier(Classifier):
    def __init__(self, train_loader=None, test_loader=None, device: Optional[torch.device] = None):
        self.device = device if device is not None else torch.device("cpu")
        self.model: Optional[nn.Sequential] = None
        self.spec: Optional[ArchitectureSpec] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._metrics = Metrics(epoch=0, batch=0, loss=float("nan"))
        self.train_loader = train_loader
        self.test_loader = test_loader

    def configure(self, spec: ArchitectureSpec) -> None:
        self.spec = spec
        dropout_rate = dropout_rate_for_dataset(spec.dataset)
        self.model = build_mlp(spec.layers.widths, spec.num_classes, dropout_rate=dropout_rate).to(self.device)
        self.model.eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self._metrics = Metrics(epoch=0, batch=0, loss=float("nan"))

    def train_epoch(
        self, epoch_idx: int, on_batch: Optional[Callable[[Metrics], None]] = None
    ) -> Metrics:
        assert self.model is not None and self.optimizer is not None
        assert self.train_loader is not None

        self.model.train()
        running_loss = 0.0
        n_batches = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            n_batches += 1

            metrics = Metrics(
                epoch=epoch_idx,
                batch=batch_idx,
                loss=loss.item(),
                done=False,
            )
            self._metrics = metrics
            if on_batch is not None:
                on_batch(metrics)

        self.model.eval()
        val_loss, accuracy = (
            self._evaluate() if self.test_loader is not None else (None, None)
        )
        final_metrics = Metrics(
            epoch=epoch_idx,
            batch=n_batches - 1 if n_batches else 0,
            loss=running_loss / max(n_batches, 1),
            accuracy=accuracy,
            val_loss=val_loss,
            done=True,
        )
        self._metrics = final_metrics
        return final_metrics

    def _evaluate(self) -> tuple[float, float]:
        """Runs a forward pass over the held-out test/validation set, returning
        (average validation loss, accuracy)."""
        assert self.model is not None
        correct = 0
        total = 0
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                total_loss += F.cross_entropy(logits, labels).item()
                n_batches += 1
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return total_loss / max(n_batches, 1), correct / max(total, 1)

    def predict(self, image: np.ndarray) -> np.ndarray:
        probs, _ = self.predict_with_activations(image)
        return probs

    def predict_with_activations(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(image).float().unsqueeze(0).to(self.device)  # (1, 28, 28)
            activations: list[np.ndarray] = []
            for layer in self.model:
                x = layer(x)
                if isinstance(layer, nn.ReLU):
                    activations.append(x.squeeze(0).cpu().numpy().copy())
            probs = F.softmax(x, dim=1).squeeze(0).cpu().numpy()
        return probs, activations

    def get_edge_weights(self) -> list[list[list[float]]]:
        """Weight matrices for every node-to-node transition, in order:
        input pixels -> first configured layer (or straight to output, if
        there are no hidden layers), each consecutive pair of hidden layers,
        and the final hidden-to-output layer. Each matrix is
        [dst_idx][src_idx], matching nn.Linear's own weight layout."""
        assert self.model is not None
        linears = [m for m in self.model if isinstance(m, nn.Linear)]
        return [lin.weight.detach().cpu().numpy().tolist() for lin in linears]

    def save_checkpoint(self, path: str) -> None:
        assert self.model is not None and self.spec is not None
        torch.save(
            {"arch": self.spec.to_dict(), "state_dict": self.model.state_dict()},
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        spec = ArchitectureSpec.from_dict(checkpoint["arch"])
        self.configure(spec)
        assert self.model is not None
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def get_metrics(self) -> Metrics:
        return self._metrics
