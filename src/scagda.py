from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SDBEConfig:
    eps: float = 1e-8
    confidence_reweight: bool = True
    shrinkage_min: float = 0.05
    shrinkage_max: float = 0.50
    shrinkage_scale: float = 10.0
    target_shrinkage_boost: float = 1.25
    covariance_similarity_power: float = 1.0
    use_mapper: bool = True


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_rows(value: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(value, dim=1, eps=eps)


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels.reshape(-1), num_classes=num_classes).float()


def _normalized_adj_multiply(
    probabilities: torch.Tensor, edge_index: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    loops = torch.arange(num_nodes, device=edge_index.device, dtype=edge_index.dtype)
    loops = loops.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loops], dim=1)
    row, col = edge_index
    degree = torch.bincount(row, minlength=num_nodes).to(probabilities.dtype).clamp_min(1.0)
    norm = degree[row].pow(-0.5) * degree[col].pow(-0.5)
    output = torch.zeros_like(probabilities)
    output.index_add_(0, row, probabilities[col] * norm.unsqueeze(1))
    return output


def class_contact_matrix(
    probabilities: torch.Tensor, edge_index: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    propagated = _normalized_adj_multiply(probabilities, edge_index, probabilities.size(0))
    contact = probabilities.t() @ propagated
    class_mass = probabilities.sum(dim=0)
    scale = torch.diag((class_mass + eps).pow(-0.5))
    contact = scale @ contact @ scale
    contact = contact.masked_fill(
        torch.eye(contact.size(0), device=contact.device, dtype=torch.bool), 0.0
    )
    return contact / contact.norm(p="fro").clamp_min(eps)


def graph_smoothness(probabilities: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return probabilities.sum() * 0.0
    row, col = edge_index
    return (probabilities[row] - probabilities[col]).pow(2).sum(dim=1).mean()


def soft_margin_loss(
    logits: torch.Tensor, soft_assignments: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    num_classes = logits.size(1)
    if num_classes <= 1:
        return logits.sum() * 0.0
    diagonal = torch.eye(num_classes, device=logits.device, dtype=torch.bool)
    competitors = logits.unsqueeze(1).expand(-1, num_classes, -1)
    competitors = competitors.masked_fill(diagonal.unsqueeze(0), float("-inf"))
    negative_logsumexp = torch.logsumexp(competitors, dim=2)
    margins = logits - negative_logsumexp
    assignments = soft_assignments / soft_assignments.sum(dim=1, keepdim=True).clamp_min(eps)
    return (assignments * F.softplus(-margins)).sum(dim=1).mean()


def _effective_sample_size(assignments: torch.Tensor, eps: float) -> torch.Tensor:
    mass = assignments.sum(dim=0)
    squared_mass = assignments.pow(2).sum(dim=0).clamp_min(eps)
    return mass.pow(2) / squared_mass


def _weighted_statistics(
    embeddings: torch.Tensor,
    assignments: torch.Tensor,
    cfg: SDBEConfig,
    *,
    target: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings = normalize_rows(embeddings, cfg.eps)
    mass = assignments.sum(dim=0).clamp_min(cfg.eps)
    centers = normalize_rows((assignments.t() @ embeddings) / mass.unsqueeze(1), cfg.eps)

    effective_n = _effective_sample_size(assignments, cfg.eps)
    beta = cfg.shrinkage_min + cfg.shrinkage_scale / (effective_n + cfg.eps)
    if target:
        beta = beta * cfg.target_shrinkage_boost
    beta = beta.clamp(cfg.shrinkage_min, cfg.shrinkage_max)

    feature_dim = embeddings.size(1)
    identity = torch.eye(feature_dim, device=embeddings.device, dtype=embeddings.dtype)
    covariances = []
    for class_index in range(assignments.size(1)):
        weight = assignments[:, class_index].clamp_min(0.0)
        centered = embeddings - centers[class_index]
        raw_covariance = (centered * weight.unsqueeze(1)).t() @ centered
        raw_covariance = raw_covariance / mass[class_index].clamp_min(1.0)
        trace_scale = torch.trace(raw_covariance) / float(feature_dim)
        covariance = (1.0 - beta[class_index]) * raw_covariance
        covariance = covariance + beta[class_index] * trace_scale * identity
        covariances.append(covariance + cfg.eps * identity)
    return centers, torch.stack(covariances), beta


def _normalize_covariances(covariances: torch.Tensor, eps: float) -> torch.Tensor:
    denominator = covariances.norm(p="fro", dim=(-2, -1), keepdim=True).clamp_min(eps)
    return covariances / denominator


def _node_semantic_displacement(
    source_centers: torch.Tensor,
    target_centers: torch.Tensor,
    source_covariances: torch.Tensor,
    target_covariances: torch.Tensor,
    cfg: SDBEConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    raw_displacement = target_centers - source_centers
    source_cov_norm = _normalize_covariances(source_covariances, cfg.eps)
    target_cov_norm = _normalize_covariances(target_covariances, cfg.eps)
    similarity = (source_cov_norm * target_cov_norm).sum(dim=(-2, -1)).clamp(0.0, 1.0)
    direction_operator = source_cov_norm + target_cov_norm
    projected = torch.einsum("cde,ce->cd", direction_operator, raw_displacement)
    displacement = projected * similarity.pow(cfg.covariance_similarity_power).unsqueeze(1)
    return displacement, similarity


def _source_direction_basis(source_centers: torch.Tensor, eps: float) -> torch.Tensor:
    basis = source_centers.unsqueeze(0) - source_centers.unsqueeze(1)
    basis = F.normalize(basis, dim=-1, eps=eps)
    diagonal = torch.eye(source_centers.size(0), device=source_centers.device, dtype=torch.bool)
    return basis.masked_fill(diagonal.unsqueeze(-1), 0.0)


def _edge_transition_tension(delta_contact: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    diagonal = torch.eye(delta_contact.size(0), device=delta_contact.device, dtype=torch.bool)
    delta_contact = delta_contact.masked_fill(diagonal, 0.0)
    return (delta_contact.unsqueeze(-1) * basis).sum(dim=1)


def _deterministic_fusion(
    node_displacement: torch.Tensor, edge_tension: torch.Tensor, eps: float
) -> torch.Tensor:
    node_norm = node_displacement.norm(dim=1)
    edge_norm = edge_tension.norm(dim=1)
    alpha = node_norm / (node_norm + edge_norm + eps)
    trend = alpha.unsqueeze(1) * F.normalize(node_displacement, dim=1, eps=eps)
    trend = trend + (1.0 - alpha).unsqueeze(1) * F.normalize(edge_tension, dim=1, eps=eps)
    return F.normalize(trend, dim=1, eps=eps)


def curriculum_strength(epoch: int, warmup_epochs: int, ramp_epochs: int, maximum: float) -> float:
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return float(maximum)
    progress = min(max(epoch - warmup_epochs, 0), ramp_epochs)
    return float(maximum) * float(progress) / float(ramp_epochs)


def tracking_strength(
    epoch: int, warmup_epochs: int, ramp_epochs: int, minimum: float, maximum: float
) -> float:
    if epoch <= warmup_epochs:
        return float(minimum)
    if ramp_epochs <= 0:
        return float(maximum)
    progress = min(max(epoch - warmup_epochs, 0), ramp_epochs)
    return float(minimum) + (float(maximum) - float(minimum)) * progress / ramp_epochs


def build_boundary_state(
    embeddings_source: torch.Tensor,
    labels_source: torch.Tensor,
    embeddings_target: torch.Tensor,
    probabilities_target: torch.Tensor,
    classifier_weight: torch.Tensor,
    edge_index_source: torch.Tensor,
    edge_index_target: torch.Tensor,
    rho: float,
    cfg: SDBEConfig,
    mapper: Optional[torch.nn.Module],
) -> Dict[str, torch.Tensor]:
    """Construct the class-evolution state and the suggested classifier boundary."""

    embeddings_source = embeddings_source.detach()
    embeddings_target = embeddings_target.detach()
    probabilities_target = probabilities_target.detach()
    num_classes = classifier_weight.size(0)

    source_assignments = one_hot(labels_source, num_classes).to(
        embeddings_source.device, embeddings_source.dtype
    )
    target_assignments = probabilities_target
    if cfg.confidence_reweight:
        target_assignments = target_assignments * probabilities_target.max(dim=1, keepdim=True).values

    source_centers, source_covariances, source_beta = _weighted_statistics(
        embeddings_source, source_assignments, cfg, target=False
    )
    target_centers, target_covariances, target_beta = _weighted_statistics(
        embeddings_target, target_assignments, cfg, target=True
    )
    node_displacement, covariance_similarity = _node_semantic_displacement(
        source_centers, target_centers, source_covariances, target_covariances, cfg
    )

    source_contact = class_contact_matrix(source_assignments, edge_index_source, cfg.eps)
    target_contact = class_contact_matrix(probabilities_target, edge_index_target, cfg.eps)
    delta_contact = target_contact - source_contact
    edge_tension = _edge_transition_tension(
        delta_contact, _source_direction_basis(source_centers, cfg.eps)
    )

    fallback = _deterministic_fusion(node_displacement, edge_tension, cfg.eps)
    normalized_weight = normalize_rows(classifier_weight, cfg.eps)
    if cfg.use_mapper and mapper is not None:
        mapped = mapper(
            node_displacement,
            edge_tension,
            source_centers,
            target_centers,
            normalized_weight,
        )
        trend = F.normalize(mapped + 0.10 * fallback.detach(), dim=1, eps=cfg.eps)
        mapper_used = 1.0
    else:
        trend = fallback
        mapper_used = 0.0

    reference_weight = F.normalize(classifier_weight + rho * trend, dim=1, eps=cfg.eps)
    return {
        "reference_weight": reference_weight,
        "trend": trend,
        "node_displacement": node_displacement,
        "edge_tension": edge_tension,
        "covariance_similarity": covariance_similarity,
        "delta_contact": delta_contact,
        "source_shrinkage": source_beta,
        "target_shrinkage": target_beta,
        "mapper_used": torch.tensor(mapper_used, device=classifier_weight.device),
    }


def adaptive_ema_momentum(
    current_weight: torch.Tensor,
    previous_weight: Optional[torch.Tensor],
    *,
    base: float,
    sensitivity: float,
    minimum: float,
    maximum: float,
    eps: float = 1e-8,
) -> float:
    if previous_weight is None:
        return 0.0
    current = normalize_rows(current_weight.detach(), eps)
    previous = normalize_rows(previous_weight.detach(), eps)
    gap = (1.0 - (current * previous).sum(dim=1).clamp(-1.0, 1.0)).mean()
    return float(np.clip(base + sensitivity * float(gap.item()), minimum, maximum))


def update_ema(
    previous_weight: Optional[torch.Tensor], current_weight: torch.Tensor, momentum: float, eps: float
) -> torch.Tensor:
    current = normalize_rows(current_weight.detach(), eps)
    if previous_weight is None:
        return current.clone()
    return normalize_rows(momentum * previous_weight + (1.0 - momentum) * current, eps)


def update_tracked_weight(
    previous_weight: Optional[torch.Tensor], reference_weight: torch.Tensor, eta: float, eps: float
) -> torch.Tensor:
    reference = normalize_rows(reference_weight.detach(), eps)
    if previous_weight is None:
        return reference.clone()
    return normalize_rows((1.0 - eta) * previous_weight + eta * reference, eps)


def tracking_alignment_loss(
    learnable_weight: torch.Tensor, reference_weight: torch.Tensor, eps: float
) -> torch.Tensor:
    learnable = normalize_rows(learnable_weight, eps)
    reference = normalize_rows(reference_weight.detach(), eps)
    return (1.0 - (learnable * reference).sum(dim=1)).mean()
