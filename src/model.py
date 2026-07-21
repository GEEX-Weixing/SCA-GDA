from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    loops = torch.arange(num_nodes, device=edge_index.device, dtype=edge_index.dtype)
    loops = loops.unsqueeze(0).repeat(2, 1)
    return torch.cat([edge_index, loops], dim=1)


def _drop_edges(edge_index: torch.Tensor, probability: float) -> torch.Tensor:
    if probability <= 0.0 or edge_index.size(1) == 0:
        return edge_index
    keep = torch.rand(edge_index.size(1), device=edge_index.device) >= probability
    return edge_index[:, keep] if keep.any() else edge_index


class GraphConv(nn.Module):
    """Dependency-free GCN layer using COO ``edge_index``."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        edge_index = _add_self_loops(edge_index, num_nodes)
        row, col = edge_index
        transformed = self.linear(x)
        degree = torch.bincount(row, minlength=num_nodes).to(transformed.dtype).clamp_min(1.0)
        norm = degree[row].pow(-0.5) * degree[col].pow(-0.5)
        output = torch.zeros_like(transformed)
        output.index_add_(0, row, transformed[col] * norm.unsqueeze(1))
        return output


class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, layers: int = 2, dropout: float = 0.5):
        super().__init__()
        if layers < 2:
            raise ValueError("layers must be at least 2")
        dimensions = [in_channels] + [2 * hidden_dim] * (layers - 1) + [hidden_dim]
        self.convs = nn.ModuleList(
            GraphConv(dimensions[index], dimensions[index + 1])
            for index in range(len(dimensions) - 1)
        )
        self.dropout = float(dropout)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GDAClassifier(nn.Module):
    def __init__(self, encoder: Encoder, num_classes: int, temperature: float = 0.5):
        super().__init__()
        self.encoder = encoder
        self.temperature = float(temperature)
        self.classifier = nn.Linear(encoder.output_dim, num_classes, bias=False)

    def logits_from_embeddings(
        self, embeddings: torch.Tensor, classifier_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if classifier_weight is None:
            return self.classifier(embeddings)
        return embeddings @ classifier_weight.t()

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, classifier_weight: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = self.encoder(x, edge_index)
        logits = self.logits_from_embeddings(embeddings, classifier_weight)
        return F.softmax(logits, dim=1), logits, embeddings

    @staticmethod
    def _similarity(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return F.normalize(z1, dim=1) @ F.normalize(z2, dim=1).t()

    def _semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int) -> torch.Tensor:
        num_nodes = z1.size(0)
        indices = torch.arange(num_nodes, device=z1.device)
        losses = []
        for start in range(0, num_nodes, batch_size):
            mask = indices[start : start + batch_size]
            refl = torch.exp(self._similarity(z1[mask], z1) / self.temperature)
            cross = torch.exp(self._similarity(z1[mask], z2) / self.temperature)
            positive = cross[:, start : start + mask.numel()].diag()
            self_term = refl[:, start : start + mask.numel()].diag()
            denominator = refl.sum(1) + cross.sum(1) - self_term
            losses.append(-torch.log(positive / denominator.clamp_min(1e-8)))
        return torch.cat(losses)

    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int) -> torch.Tensor:
        loss_12 = self._semi_loss(z1, z2, batch_size)
        loss_21 = self._semi_loss(z2, z1, batch_size)
        return 0.5 * (loss_12 + loss_21).mean()


class DomainMILoss(nn.Module):
    """Estimate I(semantic prediction; domain)."""

    def __init__(self, num_domains: int = 2, eps: float = 1e-8):
        super().__init__()
        self.num_domains = num_domains
        self.eps = eps

    def forward(self, semantic_probs: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
        domain_onehot = F.one_hot(domain_labels, num_classes=self.num_domains).to(semantic_probs.dtype)
        joint = domain_onehot.t() @ semantic_probs
        joint = joint / joint.sum().clamp_min(self.eps)
        p_domain = joint.sum(dim=1, keepdim=True)
        p_semantic = joint.sum(dim=0, keepdim=True)
        return (joint * torch.log(joint / (p_domain * p_semantic + self.eps) + self.eps)).sum()


class ConditionalDomainMILoss(nn.Module):
    """Soft class-conditioned estimate of I(semantic prediction; domain | class)."""

    def __init__(self, num_domains: int = 2, eps: float = 1e-8):
        super().__init__()
        self.num_domains = num_domains
        self.eps = eps

    def _weighted_mi(
        self, semantic_probs: torch.Tensor, domain_labels: torch.Tensor, sample_weight: torch.Tensor
    ) -> torch.Tensor:
        weight = sample_weight.reshape(-1, 1).clamp_min(0.0)
        total = weight.sum().clamp_min(self.eps)
        domain_onehot = F.one_hot(domain_labels, num_classes=self.num_domains).to(semantic_probs.dtype)
        joint = domain_onehot.t() @ (semantic_probs * weight)
        joint = joint / total
        p_domain = joint.sum(dim=1, keepdim=True)
        p_semantic = joint.sum(dim=0, keepdim=True)
        return (joint * torch.log(joint / (p_domain * p_semantic + self.eps) + self.eps)).sum()

    def forward(
        self,
        probs_source: torch.Tensor,
        probs_target: torch.Tensor,
        source_assign: torch.Tensor,
        target_assign: torch.Tensor,
    ) -> torch.Tensor:
        semantic_probs = torch.cat([probs_source, probs_target], dim=0)
        domain_labels = torch.cat(
            [
                torch.zeros(probs_source.size(0), dtype=torch.long, device=semantic_probs.device),
                torch.ones(probs_target.size(0), dtype=torch.long, device=semantic_probs.device),
            ]
        )
        assignments = torch.cat([source_assign, target_assign], dim=0).to(semantic_probs.dtype)
        class_mass = assignments.sum(dim=0).clamp_min(self.eps)
        total_mass = class_mass.sum().clamp_min(self.eps)
        terms = [
            (class_mass[c] / total_mass)
            * self._weighted_mi(semantic_probs, domain_labels, assignments[:, c])
            for c in range(assignments.size(1))
        ]
        return torch.stack(terms).sum()


class BoundaryEvolutionMapper(nn.Module):
    """Map coupled node/edge evolution signals to class-wise boundary directions."""

    def __init__(self, embedding_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or max(embedding_dim, 64)
        self.network = nn.Sequential(
            nn.Linear(7 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(
        self,
        node_displacement: torch.Tensor,
        edge_tension: torch.Tensor,
        source_centers: torch.Tensor,
        target_centers: torch.Tensor,
        classifier_weight: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [
                node_displacement,
                edge_tension,
                node_displacement * edge_tension,
                torch.abs(node_displacement - edge_tension),
                source_centers,
                target_centers,
                classifier_weight,
            ],
            dim=1,
        )
        return self.network(features)


def _drop_features(x: torch.Tensor, probability: float) -> torch.Tensor:
    mask = torch.rand(x.size(1), device=x.device) < probability
    output = x.clone()
    output[:, mask] = 0.0
    return output


def graph_contrastive_loss(
    model: GDAClassifier,
    data,
    *,
    edge_drop_1: float = 0.3,
    edge_drop_2: float = 0.5,
    feature_drop_1: float = 0.3,
    feature_drop_2: float = 0.5,
    batch_size: int = 512,
    max_nodes: int = 1024,
) -> torch.Tensor:
    edge_1 = _drop_edges(data.edge_index, edge_drop_1)
    edge_2 = _drop_edges(data.edge_index, edge_drop_2)
    x_1 = _drop_features(data.x, feature_drop_1)
    x_2 = _drop_features(data.x, feature_drop_2)
    _, _, z1 = model(x_1, edge_1)
    _, _, z2 = model(x_2, edge_2)
    if max_nodes > 0 and z1.size(0) > max_nodes:
        indices = torch.randperm(z1.size(0), device=z1.device)[:max_nodes]
        z1, z2 = z1[indices], z2[indices]
    batch_size = max(1, min(batch_size, z1.size(0)))
    return model.contrastive_loss(z1, z2, batch_size=batch_size)
