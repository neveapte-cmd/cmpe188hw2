
from __future__ import annotations

import copy
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
from sklearn.datasets import make_moons
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
        "task_id": "mlp_lvl6_optimizer_comparison_moons",
        "task_name": "MLP (Optimizer Comparison on Two Moons)",
        "task_type": "binary_classification",
        "dataset": "sklearn.datasets.make_moons",
        "success_thresholds": {
            "best_val_accuracy_min": 0.90,
        },
        "optimizers_compared": ["sgd_momentum", "adam"],
        "artifacts": [
            "loss_comparison.png",
            "best_model_decision_boundary.png",
            "metrics.json",
            "best_model.pt",
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
    batch_size: int = 32,
    val_ratio: float = 0.2,
    seed: int = 42,
    n_samples: int = 1200,
    noise: float = 0.25,
) -> Dict[str, Any]:
    x_np, y_np = make_moons(n_samples=n_samples, noise=noise, random_state=seed)

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
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "scaler": scaler,
        "input_dim": x_train.shape[1],
        "num_classes": 2,
    }


class MoonMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, int] = (32, 32),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(
    input_dim: int,
    hidden_dims: Tuple[int, int] = (32, 32),
    dropout: float = 0.10,
) -> nn.Module:
    return MoonMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)


def _confusion_counts(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum().item())
    tn = int(((y_true == 0) & (y_pred == 0)).sum().item())
    fp = int(((y_true == 0) & (y_pred == 1)).sum().item())
    fn = int(((y_true == 1) & (y_pred == 0)).sum().item())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _precision_recall_f1(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    counts = _confusion_counts(y_true, y_pred)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        **counts,
    }


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
    prf = _precision_recall_f1(y_true, y_pred)

    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "confusion": {
            "tp": prf["tp"],
            "tn": prf["tn"],
            "fp": prf["fp"],
            "fn": prf["fn"],
        },
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def predict(model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        preds = torch.argmax(logits, dim=1)
    return preds.cpu()


def _make_optimizer(
    optimizer_name: str,
    model: nn.Module,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if optimizer_name == "sgd_momentum":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    optimizer_name: str,
    epochs: int = 100,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    patience: int = 15,
) -> Dict[str, Any]:
    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(optimizer_name, model, lr=lr, weight_decay=weight_decay)

    history = {
        "loss_history": [],
        "val_loss_history": [],
        "val_accuracy_history": [],
        "val_precision_history": [],
        "val_recall_history": [],
        "val_f1_history": [],
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
        val_metrics = evaluate(model, val_loader, device, criterion=criterion)
        val_loss = float(val_metrics["loss"])

        history["loss_history"].append(train_loss)
        history["val_loss_history"].append(val_loss)
        history["val_accuracy_history"].append(val_metrics["accuracy"])
        history["val_precision_history"].append(val_metrics["precision"])
        history["val_recall_history"].append(val_metrics["recall"])
        history["val_f1_history"].append(val_metrics["f1"])

        print(
            f"[{optimizer_name}] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
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
            print(f"[{optimizer_name}] Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        **history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


def _plot_loss_comparison(
    sgd_history: Dict[str, Any],
    adam_history: Dict[str, Any],
    output_dir: str,
) -> str:
    plt.figure(figsize=(8, 5))
    plt.plot(sgd_history["val_loss_history"], label="SGD+Momentum Val Loss")
    plt.plot(adam_history["val_loss_history"], label="Adam Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Optimizer Validation Loss Comparison")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, "loss_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _plot_decision_boundary(
    model: nn.Module,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    output_dir: str,
) -> str:
    model.eval()

    x_min, x_max = x_val[:, 0].min().item() - 1.0, x_val[:, 0].max().item() + 1.0
    y_min, y_max = x_val[:, 1].min().item() - 1.0, x_val[:, 1].max().item() + 1.0

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    grid_tensor = torch.tensor(grid, dtype=torch.float32)

    with torch.no_grad():
        logits = model(grid_tensor.to(device))
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    zz = preds.reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, zz, alpha=0.35)
    plt.scatter(
        x_val[:, 0].numpy(),
        x_val[:, 1].numpy(),
        c=y_val.numpy(),
        edgecolors="k",
        s=25,
    )
    plt.title("Decision Boundary of Best Optimizer Model")
    plt.xlabel("Feature 1 (standardized)")
    plt.ylabel("Feature 2 (standardized)")
    plt.tight_layout()

    path = os.path.join(output_dir, "best_model_decision_boundary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_artifacts(
    best_model: nn.Module,
    outputs: Dict[str, Any],
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    output_dir: str = "artifacts/mlp_lvl6_optimizer_comparison_moons",
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "best_model.pt")
    torch.save(best_model.state_dict(), model_path)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)

    loss_plot_path = _plot_loss_comparison(
        outputs["optimizer_results"]["sgd_momentum"]["training"],
        outputs["optimizer_results"]["adam"]["training"],
        output_dir,
    )

    boundary_path = _plot_decision_boundary(
        best_model,
        x_val=x_val,
        y_val=y_val,
        device=device,
        output_dir=output_dir,
    )

    return {
        "best_model_path": model_path,
        "metrics_path": metrics_path,
        "loss_comparison_path": loss_plot_path,
        "decision_boundary_path": boundary_path,
    }


if __name__ == "__main__":
    EXIT_SUCCESS = 0
    EXIT_FAILURE = 1

    try:
        metadata = get_task_metadata()
        set_seed(42)
        device = get_device()
        print(f"Using device: {device}")

        data = make_dataloaders(
            batch_size=32,
            val_ratio=0.2,
            seed=42,
            n_samples=1200,
            noise=0.25,
        )

        optimizer_configs = {
            "sgd_momentum": {"lr": 0.05, "weight_decay": 1e-4, "epochs": 100, "patience": 15},
            "adam": {"lr": 1e-2, "weight_decay": 1e-4, "epochs": 100, "patience": 15},
        }

        optimizer_results: Dict[str, Any] = {}
        trained_models: Dict[str, nn.Module] = {}

        for optimizer_name in ["sgd_momentum", "adam"]:
            print(f"\n=== Training with {optimizer_name} ===")
            set_seed(42)

            model = build_model(
                input_dim=data["input_dim"],
                hidden_dims=(32, 32),
                dropout=0.10,
            ).to(device)

            training_output = train(
                model=model,
                train_loader=data["train_loader"],
                val_loader=data["val_loader"],
                device=device,
                optimizer_name=optimizer_name,
                epochs=optimizer_configs[optimizer_name]["epochs"],
                lr=optimizer_configs[optimizer_name]["lr"],
                weight_decay=optimizer_configs[optimizer_name]["weight_decay"],
                patience=optimizer_configs[optimizer_name]["patience"],
            )

            criterion = nn.CrossEntropyLoss()
            train_metrics = evaluate(model, data["train_eval_loader"], device, criterion=criterion)
            val_metrics = evaluate(model, data["val_loader"], device, criterion=criterion)

            optimizer_results[optimizer_name] = {
                "training": training_output,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
            trained_models[optimizer_name] = copy.deepcopy(model).cpu()

            print(
                f"{optimizer_name} train | "
                f"loss={train_metrics['loss']:.4f} | "
                f"acc={train_metrics['accuracy']:.4f} | "
                f"precision={train_metrics['precision']:.4f} | "
                f"recall={train_metrics['recall']:.4f} | "
                f"f1={train_metrics['f1']:.4f}"
            )
            print(
                f"{optimizer_name} val   | "
                f"loss={val_metrics['loss']:.4f} | "
                f"acc={val_metrics['accuracy']:.4f} | "
                f"precision={val_metrics['precision']:.4f} | "
                f"recall={val_metrics['recall']:.4f} | "
                f"f1={val_metrics['f1']:.4f}"
            )

        # Select best optimizer by validation F1, break tie by lower validation loss
        optimizer_names = list(optimizer_results.keys())
        best_optimizer = max(
            optimizer_names,
            key=lambda name: (
                optimizer_results[name]["val_metrics"]["f1"],
                -optimizer_results[name]["val_metrics"]["loss"],
            ),
        )

        best_model = trained_models[best_optimizer].to(device)

        outputs = {
            "metadata": metadata,
            "optimizer_results": optimizer_results,
            "selected_best_optimizer": best_optimizer,
            "selection_rule": "highest validation F1, tie-break by lower validation loss",
        }

        artifact_paths = save_artifacts(
            best_model=best_model,
            outputs=outputs,
            x_val=data["x_val"],
            y_val=data["y_val"],
            device=device,
        )
        outputs["artifact_paths"] = artifact_paths

        best_val_metrics = optimizer_results[best_optimizer]["val_metrics"]

        print("\n=== Final Comparison ===")
        for name in optimizer_names:
            vm = optimizer_results[name]["val_metrics"]
            print(
                f"{name:13s} | "
                f"val_loss={vm['loss']:.4f} | "
                f"val_acc={vm['accuracy']:.4f} | "
                f"precision={vm['precision']:.4f} | "
                f"recall={vm['recall']:.4f} | "
                f"f1={vm['f1']:.4f}"
            )

        print(f"\nBest optimizer: {best_optimizer}")
        print(f"Artifacts saved to: {artifact_paths}")

        assert best_val_metrics["accuracy"] > metadata["success_thresholds"]["best_val_accuracy_min"], (
            f"Best validation accuracy too low: {best_val_metrics['accuracy']:.4f}"
        )

        print("\nTask completed successfully.")
        sys.exit(EXIT_SUCCESS)

    except Exception as exc:
        print(f"\nTask failed: {exc}", file=sys.stderr)
        sys.exit(EXIT_FAILURE)
