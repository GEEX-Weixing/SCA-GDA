from __future__ import annotations

from typing import Dict

import torch


def classification_metrics(labels: torch.Tensor, predictions: torch.Tensor) -> Dict[str, float]:
    labels = labels.reshape(-1)
    predictions = predictions.reshape(-1)
    accuracy = (labels == predictions).float().mean()

    num_classes = int(max(labels.max().item(), predictions.max().item())) + 1
    f1_values = []
    for class_index in range(num_classes):
        true_positive = ((predictions == class_index) & (labels == class_index)).sum().float()
        false_positive = ((predictions == class_index) & (labels != class_index)).sum().float()
        false_negative = ((predictions != class_index) & (labels == class_index)).sum().float()
        denominator = 2.0 * true_positive + false_positive + false_negative
        f1 = torch.where(denominator > 0, 2.0 * true_positive / denominator, torch.zeros_like(denominator))
        f1_values.append(f1)

    macro_f1 = torch.stack(f1_values).mean()
    # For single-label multiclass classification, micro-F1 equals accuracy.
    return {
        "accuracy": 100.0 * float(accuracy.item()),
        "macro_f1": 100.0 * float(macro_f1.item()),
        "micro_f1": 100.0 * float(accuracy.item()),
    }
