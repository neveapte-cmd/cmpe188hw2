

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
from sklearn.datasets import load_digits
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
        "task_id": "mlp_lvl7_regularization_ablation_digits",
        "task_name": "MLP (Regularization Ablation on Digits)",
        "task_type": "multiclass_classification",
        "dataset": "sklearn.datasets.load_digits",
        "num_classes": 10,
        "success_rule": "at least one regularized model must achieve validation accuracy >= baseline accuracy",
        "artifacts": [
            "combined_loss_curves.png",
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
    batch_size: int = 64,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    digits = load_digits()
    x = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)

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
        "input_dim": x_train.shape[1],
        "num_classes": len(np.unique(y)),
        "target_names": [str(i) for i in sorted(np.unique(y))],
    }


class DigitsMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.0,
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
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.0,
) -> nn.Module:
    return DigitsMLP(
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

        if precision + recall == 0.0:
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
    cm = _compute_confusion_matrix(y_true, y_pred, num_classes=logits.shape[1])
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
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int = 10,
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

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
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


def _plot_combined_loss_curves(
    experiment_outputs: Dict[str, Any],
    output_dir: str,
) -> str:
    plt.figure(figsize=(10, 6))

    for variant_name, result in experiment_outputs.items():
        plt.plot(
            result["training"]["loss_history"],
            linestyle="--",
            label=f"{variant_name} train",
        )
        plt.plot(
            result["training"]["val_loss_history"],
            label=f"{variant_name} val",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss Across Regularization Variants")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, "combined_loss_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_artifacts(
    best_model: nn.Module,
    outputs: Dict[str, Any],
    output_dir: str = "artifacts/mlp_lvl7_regularization_ablation_digits",
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "best_model.pt")
    torch.save(best_model.state_dict(), model_path)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)

    plot_path = _plot_combined_loss_curves(outputs["variant_results"], output_dir)

    return {
        "best_model_path": model_path,
        "metrics_path": metrics_path,
        "combined_loss_plot_path": plot_path,
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

        variant_configs = {
            "baseline": {
                "dropout": 0.0,
                "weight_decay": 0.0,
            },
            "dropout": {
                "dropout": 0.30,
                "weight_decay": 0.0,
            },
            "weight_decay": {
                "dropout": 0.0,
                "weight_decay": 1e-3,
            },
        }

        variant_results: Dict[str, Any] = {}
        trained_models: Dict[str, nn.Module] = {}

        for variant_name, cfg in variant_configs.items():
            print(f"\n=== Training variant: {variant_name} ===")
            set_seed(42)

            model = build_model(
                input_dim=data["input_dim"],
                num_classes=data["num_classes"],
                hidden_dims=(128, 64),
                dropout=cfg["dropout"],
            ).to(device)

            training_output = train(
                model=model,
                train_loader=data["train_loader"],
                val_loader=data["val_loader"],
                device=device,
                epochs=60,
                lr=1e-3,
                weight_decay=cfg["weight_decay"],
                patience=10,
            )

            criterion = nn.CrossEntropyLoss()
            train_metrics = evaluate(model, data["train_eval_loader"], device, criterion=criterion)
            val_metrics = evaluate(model, data["val_loader"], device, criterion=criterion)

            generalization_gap = float(train_metrics["accuracy"] - val_metrics["accuracy"])

            variant_results[variant_name] = {
                "config": cfg,
                "training": training_output,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "generalization_gap": generalization_gap,
            }

            trained_models[variant_name] = copy.deepcopy(model).cpu()

            print(
                f"{variant_name} train | "
                f"loss={train_metrics['loss']:.4f} | "
                f"acc={train_metrics['accuracy']:.4f} | "
                f"macro_f1={train_metrics['macro_f1']:.4f}"
            )
            print(
                f"{variant_name} val   | "
                f"loss={val_metrics['loss']:.4f} | "
                f"acc={val_metrics['accuracy']:.4f} | "
                f"macro_f1={val_metrics['macro_f1']:.4f} | "
                f"gap={generalization_gap:.4f}"
            )

        baseline_acc = variant_results["baseline"]["val_metrics"]["accuracy"]
        dropout_acc = variant_results["dropout"]["val_metrics"]["accuracy"]
        weight_decay_acc = variant_results["weight_decay"]["val_metrics"]["accuracy"]

        regularized_candidates = ["dropout", "weight_decay"]
        best_regularized_variant = max(
            regularized_candidates,
            key=lambda name: (
                variant_results[name]["val_metrics"]["accuracy"],
                variant_results[name]["val_metrics"]["macro_f1"],
            ),
        )

        best_overall_variant = max(
            variant_results.keys(),
            key=lambda name: (
                variant_results[name]["val_metrics"]["accuracy"],
                variant_results[name]["val_metrics"]["macro_f1"],
            ),
        )

        outputs = {
            "metadata": metadata,
            "variant_results": variant_results,
            "selected_best_regularized_model": best_regularized_variant,
            "selected_best_overall_model": best_overall_variant,
            "comparison_summary": {
                "baseline_val_accuracy": baseline_acc,
                "dropout_val_accuracy": dropout_acc,
                "weight_decay_val_accuracy": weight_decay_acc,
            },
        }

        best_model = trained_models[best_overall_variant].to(device)
        artifact_paths = save_artifacts(best_model, outputs)
        outputs["artifact_paths"] = artifact_paths

        print("\n=== Final Comparison ===")
        for name, result in variant_results.items():
            vm = result["val_metrics"]
            print(
                f"{name:12s} | "
                f"val_loss={vm['loss']:.4f} | "
                f"val_acc={vm['accuracy']:.4f} | "
                f"val_macro_f1={vm['macro_f1']:.4f} | "
                f"gap={result['generalization_gap']:.4f}"
            )

        print(f"\nBest regularized variant: {best_regularized_variant}")
        print(f"Best overall variant: {best_overall_variant}")
        print(f"Artifacts saved to: {artifact_paths}")

        assert (
            dropout_acc >= baseline_acc or weight_decay_acc >= baseline_acc
        ), "No regularized model matched or exceeded baseline validation accuracy."

        print("\nTask completed successfully.")
        sys.exit(EXIT_SUCCESS)

    except Exception as exc:
        print(f"\nTask failed: {exc}", file=sys.stderr)
        sys.exit(EXIT_FAILURE)
