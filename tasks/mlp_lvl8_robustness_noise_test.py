
from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Standardizer:
    mean: torch.Tensor
    std: torch.Tensor

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def get_task_metadata() -> Dict[str, Any]:
    return {
        "task_id": "mlp_lvl8_robustness_noise_test",
        "task_name": "MLP (Robustness to Noisy Features)",
        "task_type": "multiclass_classification",
        "dataset": "sklearn.datasets.make_classification",
        "success_thresholds": {
            "clean_val_accuracy_min": 0.85,
            "robustness_drop_max": 0.35,
        },
        "noise_levels": [0.0, 0.1, 0.25, 0.5],
        "artifacts": [
            "robustness_curve.png",
            "metrics.json",
            "model.pt",
        ],
    }


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fit_standardizer(x_train: torch.Tensor) -> Standardizer:
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True, unbiased=False)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return Standardizer(mean=mean, std=std)


def make_dataloaders(
    batch_size: int = 64,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    x_np, y_np = make_classification(
        n_samples=2500,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=2.0,
        flip_y=0.01,
        random_state=seed,
    )

    x_train_np, x_val_np, y_train_np, y_val_np = train_test_split(
        x_np.astype(np.float32),
        y_np.astype(np.int64),
        test_size=val_ratio,
        random_state=seed,
        stratify=y_np,
    )

    x_train = torch.tensor(x_train_np, dtype=torch.float32)
    x_val = torch.tensor(x_val_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    y_val = torch.tensor(y_val_np, dtype=torch.long)

    scaler = _fit_standardizer(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

    return {
        "train_loader": train_loader,
        "train_eval_loader": train_eval_loader,
        "val_loader": val_loader,
        "x_val": x_val,
        "y_val": y_val,
        "input_dim": x_train.shape[1],
        "num_classes": len(np.unique(y_np)),
    }


class RobustMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Tuple[int, int] = (128, 64),
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(
    input_dim: int,
    num_classes: int,
    hidden_dims: Tuple[int, int] = (128, 64),
) -> nn.Module:
    return RobustMLP(input_dim=input_dim, num_classes=num_classes, hidden_dims=hidden_dims)


def _macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    f1_scores: List[float] = []

    for c in range(num_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum().item())
        fp = int(((y_true != c) & (y_pred == c)).sum().item())
        fn = int(((y_true == c) & (y_pred != c)).sum().item())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return float(sum(f1_scores) / num_classes)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    noise_std: float = 0.0,
) -> Dict[str, Any]:
    model.eval()

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)

            if noise_std > 0.0:
                noise = torch.randn_like(xb) * noise_std
                xb = xb + noise

            logits = model(xb)
            all_logits.append(logits.cpu())
            all_targets.append(yb.cpu())

    logits = torch.cat(all_logits, dim=0)
    y_true = torch.cat(all_targets, dim=0)
    y_pred = torch.argmax(logits, dim=1)

    loss = None
    if criterion is not None:
        loss = float(criterion(logits, y_true).item())

    accuracy = float((y_pred == y_true).float().mean().item())
    macro_f1 = _macro_f1(y_true, y_pred, num_classes=logits.shape[1])

    return {
        "noise_std": noise_std,
        "loss": loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def predict(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        preds = torch.argmax(logits, dim=1)
    return preds.cpu()


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 70,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 12,
) -> Dict[str, Any]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "loss_history": [],
        "val_loss_history": [],
        "val_accuracy_history": [],
    }

    best_state = None
    best_val_loss = math.inf
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            batch_size = xb.shape[0]
            total_loss += loss.item() * batch_size
            total_seen += batch_size

        train_loss = total_loss / max(total_seen, 1)
        val_metrics = evaluate(model, val_loader, device, criterion=criterion, noise_std=0.0)
        val_loss = float(val_metrics["loss"])

        history["loss_history"].append(train_loss)
        history["val_loss_history"].append(val_loss)
        history["val_accuracy_history"].append(val_metrics["accuracy"])

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_loss < (best_val_loss - 1e-6):
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        **history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


def _plot_robustness_curve(
    noise_levels: List[float],
    accuracies: List[float],
    output_dir: str,
) -> str:
    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels, accuracies, marker="o")
    plt.xlabel("Gaussian Noise Std")
    plt.ylabel("Validation Accuracy")
    plt.title("Robustness Curve: Accuracy vs Noise Level")
    plt.tight_layout()

    path = os.path.join(output_dir, "robustness_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_artifacts(
    model: nn.Module,
    outputs: Dict[str, Any],
    output_dir: str = "artifacts/mlp_lvl8_robustness_noise_test",
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), model_path)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)

    curve_path = _plot_robustness_curve(
        outputs["noise_levels"],
        outputs["accuracy_by_noise"],
        output_dir,
    )

    return {
        "model_path": model_path,
        "metrics_path": metrics_path,
        "robustness_curve_path": curve_path,
    }


if __name__ == "__main__":
    EXIT_SUCCESS = 0
    EXIT_FAILURE = 1

    try:
        metadata = get_task_metadata()
        set_seed(42)
        device = get_device()
        print(f"Using device: {device}")

        data = make_dataloaders(batch_size=64, val_ratio=0.2, seed=42)

        model = build_model(
            input_dim=data["input_dim"],
            num_classes=data["num_classes"],
            hidden_dims=(128, 64),
        ).to(device)

        train_outputs = train(
            model=model,
            train_loader=data["train_loader"],
            val_loader=data["val_loader"],
            device=device,
            epochs=70,
            lr=1e-3,
            weight_decay=1e-4,
            patience=12,
        )

        criterion = nn.CrossEntropyLoss()
        train_metrics = evaluate(
            model,
            data["train_eval_loader"],
            device,
            criterion=criterion,
            noise_std=0.0,
        )

        noise_levels = metadata["noise_levels"]
        val_metrics_by_noise: Dict[str, Any] = {}
        accuracy_by_noise: List[float] = []

        print("\n=== Robustness Evaluation ===")
        for sigma in noise_levels:
            metrics = evaluate(
                model,
                data["val_loader"],
                device,
                criterion=criterion,
                noise_std=float(sigma),
            )
            key = f"{sigma:.2f}"
            val_metrics_by_noise[key] = metrics
            accuracy_by_noise.append(metrics["accuracy"])

            print(
                f"noise_std={sigma:.2f} | "
                f"val_loss={metrics['loss']:.4f} | "
                f"val_acc={metrics['accuracy']:.4f} | "
                f"val_macro_f1={metrics['macro_f1']:.4f}"
            )

        clean_accuracy = val_metrics_by_noise["0.00"]["accuracy"]
        noisy_accuracy = val_metrics_by_noise["0.50"]["accuracy"]
        robustness_drop = float(clean_accuracy - noisy_accuracy)

        outputs = {
            "metadata": metadata,
            "loss_history": train_outputs["loss_history"],
            "val_loss_history": train_outputs["val_loss_history"],
            "best_epoch": train_outputs["best_epoch"],
            "best_val_loss": train_outputs["best_val_loss"],
            "train_metrics": train_metrics,
            "noise_levels": noise_levels,
            "val_metrics_by_noise": val_metrics_by_noise,
            "accuracy_by_noise": accuracy_by_noise,
            "robustness_drop": robustness_drop,
            "final_clean_metrics": val_metrics_by_noise["0.00"],
        }

        artifact_paths = save_artifacts(model, outputs)
        outputs["artifact_paths"] = artifact_paths

        print("\n=== Final Summary ===")
        print(f"Clean validation accuracy: {clean_accuracy:.4f}")
        print(f"Accuracy at sigma=0.50: {noisy_accuracy:.4f}")
        print(f"Robustness drop: {robustness_drop:.4f}")
        print(f"Artifacts saved to: {artifact_paths}")

        assert clean_accuracy > metadata["success_thresholds"]["clean_val_accuracy_min"], (
            f"Clean validation accuracy too low: {clean_accuracy:.4f}"
        )
        assert robustness_drop < metadata["success_thresholds"]["robustness_drop_max"], (
            f"Robustness drop too large: {robustness_drop:.4f}"
        )

        print("\nTask completed successfully.")
        sys.exit(EXIT_SUCCESS)

    except Exception as exc:
        print(f"\nTask failed: {exc}", file=sys.stderr)
        sys.exit(EXIT_FAILURE)
