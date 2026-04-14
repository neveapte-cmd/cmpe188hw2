
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
from sklearn.datasets import load_wine
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
        "task_id": "mlp_lvl5_wine_quality_tabular",
        "task_name": "MLP (Tabular Classification on Wine Dataset)",
        "task_type": "multiclass_classification",
        "dataset": "sklearn.datasets.load_wine",
        "input_dim": 13,
        "num_classes": 3,
        "success_thresholds": {
            "val_accuracy_min": 0.90,
            "val_macro_f1_min": 0.90,
        },
        "artifacts": [
            "loss_curve.png",
            "confusion_matrix.png",
            "metrics.json",
            "model.pt",
        ],
    }


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic behavior where reasonable
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _manual_standardize_fit(x_train: torch.Tensor) -> Standardizer:
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True, unbiased=False)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return Standardizer(mean=mean, std=std)


def make_dataloaders(
    batch_size: int = 32,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    wine = load_wine()
    x = wine.data.astype(np.float32)
    y = wine.target.astype(np.int64)

    x_train_np, x_val_np, y_train_np, y_val_np = train_test_split(
        x,
        y,
        test_size=val_ratio,
        random_state=seed,
        stratify=y,
    )

    x_train = torch.tensor(x_train_np, dtype=torch.float32)
    x_val = torch.tensor(x_val_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    y_val = torch.tensor(y_val_np, dtype=torch.long)

    scaler = _manual_standardize_fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # Full-batch eval loaders for deterministic evaluation
    train_eval_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

    return {
        "train_loader": train_loader,
        "train_eval_loader": train_eval_loader,
        "val_loader": val_loader,
        "scaler": scaler,
        "feature_names": wine.feature_names,
        "target_names": list(wine.target_names),
        "input_dim": x_train.shape[1],
        "num_classes": len(np.unique(y)),
    }


class WineMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Tuple[int, int] = (64, 32),
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(
    input_dim: int,
    num_classes: int,
    hidden_dims: Tuple[int, int] = (64, 32),
    dropout: float = 0.15,
) -> nn.Module:
    return WineMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )


def _compute_confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        cm[int(t), int(p)] += 1
    return cm


def _macro_f1_from_confusion_matrix(cm: torch.Tensor) -> float:
    f1_scores: List[float] = []
    num_classes = cm.shape[0]

    for c in range(num_classes):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)

        f1_scores.append(f1)

    return float(sum(f1_scores) / num_classes)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> Dict[str, Any]:
    model.eval()

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
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
    num_classes = logits.shape[1]
    cm = _compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    macro_f1 = _macro_f1_from_confusion_matrix(cm)

    return {
        "loss": loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
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
    epochs: int = 120,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
) -> Dict[str, Any]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "loss_history": [],
        "val_loss_history": [],
        "val_accuracy_history": [],
        "val_macro_f1_history": [],
    }

    best_state = None
    best_val_loss = math.inf
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            batch_size = xb.shape[0]
            running_loss += loss.item() * batch_size
            seen += batch_size

        train_loss = running_loss / max(seen, 1)
        val_metrics = evaluate(model, val_loader, device, criterion=criterion)
        val_loss = float(val_metrics["loss"])

        history["loss_history"].append(train_loss)
        history["val_loss_history"].append(val_loss)
        history["val_accuracy_history"].append(val_metrics["accuracy"])
        history["val_macro_f1_history"].append(val_metrics["macro_f1"])

        improved = val_loss < (best_val_loss - 1e-6)
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

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


def _plot_loss_curves(
    loss_history: List[float],
    val_loss_history: List[float],
    output_dir: str,
) -> str:
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _plot_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    output_dir: str,
) -> str:
    cm_np = np.array(cm, dtype=np.int64)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm_np, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm_np.max() / 2.0 if cm_np.size > 0 else 0.0
    for i in range(cm_np.shape[0]):
        for j in range(cm_np.shape[1]):
            plt.text(
                j,
                i,
                str(cm_np[i, j]),
                ha="center",
                va="center",
                color="white" if cm_np[i, j] > thresh else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_artifacts(
    model: nn.Module,
    outputs: Dict[str, Any],
    output_dir: str = "artifacts/mlp_lvl5_wine_quality_tabular",
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), model_path)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)

    loss_plot_path = _plot_loss_curves(
        outputs["loss_history"],
        outputs["val_loss_history"],
        output_dir,
    )

    cm_plot_path = _plot_confusion_matrix(
        outputs["final_metrics"]["val"]["confusion_matrix"],
        outputs["metadata"]["target_names"],
        output_dir,
    )

    return {
        "model_path": model_path,
        "metrics_path": metrics_path,
        "loss_curve_path": loss_plot_path,
        "confusion_matrix_path": cm_plot_path,
    }


if __name__ == "__main__":
    EXIT_SUCCESS = 0
    EXIT_FAILURE = 1

    try:
        metadata = get_task_metadata()
        set_seed(42)
        device = get_device()
        print(f"Using device: {device}")

        dl = make_dataloaders(batch_size=32, val_ratio=0.2, seed=42)

        model = build_model(
            input_dim=dl["input_dim"],
            num_classes=dl["num_classes"],
            hidden_dims=(64, 32),
            dropout=0.15,
        ).to(device)

        train_outputs = train(
            model=model,
            train_loader=dl["train_loader"],
            val_loader=dl["val_loader"],
            device=device,
            epochs=120,
            lr=1e-3,
            weight_decay=1e-4,
            patience=15,
        )

        criterion = nn.CrossEntropyLoss()
        train_metrics = evaluate(model, dl["train_eval_loader"], device, criterion=criterion)
        val_metrics = evaluate(model, dl["val_loader"], device, criterion=criterion)

        outputs = {
            "metadata": {
                **metadata,
                "target_names": dl["target_names"],
                "feature_names": dl["feature_names"],
            },
            "loss_history": train_outputs["loss_history"],
            "val_loss_history": train_outputs["val_loss_history"],
            "best_epoch": train_outputs["best_epoch"],
            "best_val_loss": train_outputs["best_val_loss"],
            "final_metrics": {
                "train": train_metrics,
                "val": val_metrics,
            },
        }

        artifact_paths = save_artifacts(model, outputs)
        outputs["artifact_paths"] = artifact_paths

        print("\n=== Final Metrics ===")
        print(
            f"Train | loss={train_metrics['loss']:.4f} | "
            f"acc={train_metrics['accuracy']:.4f} | "
            f"macro_f1={train_metrics['macro_f1']:.4f}"
        )
        print(
            f"Val   | loss={val_metrics['loss']:.4f} | "
            f"acc={val_metrics['accuracy']:.4f} | "
            f"macro_f1={val_metrics['macro_f1']:.4f}"
        )
        print(f"Best epoch: {train_outputs['best_epoch']}")
        print(f"Artifacts saved to: {artifact_paths}")

        assert val_metrics["accuracy"] > metadata["success_thresholds"]["val_accuracy_min"], (
            f"Validation accuracy too low: {val_metrics['accuracy']:.4f}"
        )
        assert val_metrics["macro_f1"] > metadata["success_thresholds"]["val_macro_f1_min"], (
            f"Validation macro-F1 too low: {val_metrics['macro_f1']:.4f}"
        )

        print("\nTask completed successfully.")
        sys.exit(EXIT_SUCCESS)

    except Exception as exc:
        print(f"\nTask failed: {exc}", file=sys.stderr)
        sys.exit(EXIT_FAILURE)
