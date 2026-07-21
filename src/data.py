from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch


@dataclass
class GraphData:
    """Small graph container used to avoid a mandatory PyG dependency."""

    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    name: str = "graph"

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))

    @property
    def num_features(self) -> int:
        return int(self.x.size(1))

    @property
    def num_classes(self) -> int:
        return int(self.y.max().item()) + 1

    def to(self, device: Union[str, torch.device]) -> "GraphData":
        return GraphData(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            y=self.y.to(device),
            name=self.name,
        )


def _dense_float32(value) -> np.ndarray:
    if sp.issparse(value):
        value = value.toarray()
    return np.asarray(value, dtype=np.float32)


def _labels_to_indices(value) -> np.ndarray:
    if sp.issparse(value):
        value = value.toarray()
    labels = np.asarray(value)
    if labels.ndim == 2 and labels.shape[1] > 1:
        labels = labels.argmax(axis=1)
    else:
        labels = labels.reshape(-1)
    _, labels = np.unique(labels, return_inverse=True)
    return labels.astype(np.int64, copy=False)


def _adjacency_to_edge_index(adjacency, make_undirected: bool) -> torch.Tensor:
    if sp.issparse(adjacency):
        coo = adjacency.tocoo()
        row = coo.row.astype(np.int64, copy=False)
        col = coo.col.astype(np.int64, copy=False)
    else:
        row, col = np.nonzero(np.asarray(adjacency))
        row = row.astype(np.int64, copy=False)
        col = col.astype(np.int64, copy=False)

    edge_index = torch.from_numpy(np.stack([row, col], axis=0)).long()
    if make_undirected and edge_index.numel() > 0:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_index = torch.unique(edge_index.t(), dim=0).t().contiguous()
    return edge_index


def _row_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.abs(x).sum(axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def load_mat_graph(
    path: Union[str, Path],
    *,
    feature_key: str = "attrb",
    adjacency_key: str = "network",
    label_key: str = "group",
    row_normalize_features: bool = False,
    make_undirected: bool = False,
) -> GraphData:
    """Load the MATLAB graph format used by the original project.

    Required keys:
      - ``attrb``: node feature matrix [N, F]
      - ``network``: sparse/dense adjacency matrix [N, N]
      - ``group``: class-index vector or one-hot matrix [N] / [N, C]
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    payload = sio.loadmat(path)
    missing = [key for key in (feature_key, adjacency_key, label_key) if key not in payload]
    if missing:
        raise KeyError(f"{path} is missing MATLAB keys: {missing}")

    x_np = _dense_float32(payload[feature_key])
    if row_normalize_features:
        x_np = _row_normalize(x_np)
    y_np = _labels_to_indices(payload[label_key])
    edge_index = _adjacency_to_edge_index(payload[adjacency_key], make_undirected)

    if x_np.ndim != 2:
        raise ValueError(f"Features must be 2D, got shape={x_np.shape}")
    if x_np.shape[0] != y_np.shape[0]:
        raise ValueError(f"Node/label mismatch: {x_np.shape[0]} vs {y_np.shape[0]}")
    if edge_index.numel() and int(edge_index.max()) >= x_np.shape[0]:
        raise ValueError("Adjacency contains a node index outside the feature matrix")

    return GraphData(
        x=torch.from_numpy(np.ascontiguousarray(x_np)),
        edge_index=edge_index,
        y=torch.from_numpy(np.ascontiguousarray(y_np)),
        name=path.stem,
    )
