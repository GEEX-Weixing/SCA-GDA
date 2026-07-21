from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.data import GraphData, load_mat_graph
from src.metrics import classification_metrics
from src.model import (
    BoundaryEvolutionMapper,
    ConditionalDomainMILoss,
    DomainMILoss,
    Encoder,
    GDAClassifier,
    graph_contrastive_loss,
)
from src.scagda import (
    SDBEConfig,
    adaptive_ema_momentum,
    build_boundary_state,
    curriculum_strength,
    graph_smoothness,
    normalize_rows,
    one_hot,
    set_seed,
    soft_margin_loss,
    tracking_alignment_loss,
    tracking_strength,
    update_ema,
    update_tracked_weight,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal SDBE-GDA training entry point")
    parser.add_argument("--source", required=True, help="Source-domain .mat file")
    parser.add_argument("--target", required=True, help="Target-domain .mat file")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--row-normalize-features", action="store_true")
    parser.add_argument("--make-undirected", action="store_true")

    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--warmup-epochs", type=int, default=100)
    parser.add_argument("--ramp-epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--encoder-dropout", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)

    parser.add_argument("--lambda-contrast", type=float, default=1.0)
    parser.add_argument("--lambda-source", type=float, default=1.0)
    parser.add_argument("--lambda-mi", type=float, default=1.5)
    parser.add_argument("--domain-info-mode", choices=["conditional", "global", "none"], default="conditional")
    parser.add_argument("--lambda-track", type=float, default=0.2)
    parser.add_argument("--lambda-margin", type=float, default=0.05)
    parser.add_argument("--lambda-smooth", type=float, default=0.02)
    parser.add_argument("--rho-max", type=float, default=0.75)

    parser.add_argument("--contrast-batch-size", type=int, default=512)
    parser.add_argument("--contrast-max-nodes", type=int, default=1024)
    parser.add_argument("--disable-mapper", action="store_true")
    parser.add_argument("--mapper-hidden-dim", type=int, default=0)
    parser.add_argument("--mapper-dropout", type=float, default=0.0)
    parser.add_argument("--disable-tracking", action="store_true")
    parser.add_argument("--ema-base", type=float, default=0.90)
    parser.add_argument("--ema-min", type=float, default=0.90)
    parser.add_argument("--ema-max", type=float, default=0.99)
    parser.add_argument("--ema-sensitivity", type=float, default=2.0)
    parser.add_argument("--tracking-eta-min", type=float, default=0.15)
    parser.add_argument("--tracking-eta-max", type=float, default=0.75)
    parser.add_argument("--log-interval", type=int, default=25)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def domain_information_loss(
    mode: str,
    global_loss: DomainMILoss,
    conditional_loss: ConditionalDomainMILoss,
    probs_source: torch.Tensor,
    probs_target: torch.Tensor,
    labels_source: torch.Tensor,
) -> torch.Tensor:
    if mode == "none":
        return probs_source.sum() * 0.0
    if mode == "global":
        probabilities = torch.cat([probs_source, probs_target], dim=0)
        domains = torch.cat(
            [
                torch.zeros(probs_source.size(0), dtype=torch.long, device=probabilities.device),
                torch.ones(probs_target.size(0), dtype=torch.long, device=probabilities.device),
            ]
        )
        return global_loss(probabilities, domains)
    source_assign = one_hot(labels_source, probs_source.size(1)).to(
        probs_source.device, probs_source.dtype
    )
    return conditional_loss(
        probs_source,
        probs_target,
        source_assign=source_assign,
        target_assign=probs_target.detach(),
    )


def build_model(args: argparse.Namespace, source: GraphData, device: torch.device):
    encoder = Encoder(
        source.num_features,
        hidden_dim=args.hidden_dim,
        layers=args.encoder_layers,
        dropout=args.encoder_dropout,
    )
    model = GDAClassifier(encoder, source.num_classes, temperature=args.temperature).to(device)
    mapper_hidden = None if args.mapper_hidden_dim <= 0 else args.mapper_hidden_dim
    mapper = BoundaryEvolutionMapper(
        args.hidden_dim, hidden_dim=mapper_hidden, dropout=args.mapper_dropout
    ).to(device)
    return model, mapper


@torch.no_grad()
def refresh_tracked_head(
    args: argparse.Namespace,
    cfg: SDBEConfig,
    model: GDAClassifier,
    mapper: BoundaryEvolutionMapper,
    source: GraphData,
    target: GraphData,
    lowpass_weight: torch.Tensor,
    tracked_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    model.eval()
    mapper.eval()
    _, _, embeddings_source = model(source.x, source.edge_index)
    probabilities_target, _, embeddings_target = model(target.x, target.edge_index)
    rho = curriculum_strength(args.epochs, args.warmup_epochs, args.ramp_epochs, args.rho_max)
    state = build_boundary_state(
        embeddings_source,
        source.y,
        embeddings_target,
        probabilities_target,
        model.classifier.weight,
        source.edge_index,
        target.edge_index,
        rho,
        cfg,
        None if args.disable_mapper else mapper,
    )
    if args.disable_tracking:
        lowpass_weight = normalize_rows(state["reference_weight"].detach(), cfg.eps)
        tracked_weight = lowpass_weight.clone()
    else:
        momentum = adaptive_ema_momentum(
            state["reference_weight"],
            lowpass_weight,
            base=args.ema_base,
            sensitivity=args.ema_sensitivity,
            minimum=args.ema_min,
            maximum=args.ema_max,
            eps=cfg.eps,
        )
        lowpass_weight = update_ema(lowpass_weight, state["reference_weight"], momentum, cfg.eps)
        eta = tracking_strength(
            args.epochs,
            args.warmup_epochs,
            args.ramp_epochs,
            args.tracking_eta_min,
            args.tracking_eta_max,
        )
        tracked_weight = update_tracked_weight(tracked_weight, lowpass_weight, eta, cfg.eps)
    return lowpass_weight, tracked_weight, state


@torch.no_grad()
def evaluate(
    model: GDAClassifier, graph: GraphData, classifier_weight: torch.Tensor
) -> Dict[str, float]:
    model.eval()
    _, _, embeddings = model(graph.x, graph.edge_index)
    predictions = model.logits_from_embeddings(embeddings, classifier_weight).argmax(dim=1)
    return classification_metrics(graph.y, predictions)


def train_run(
    args: argparse.Namespace,
    source_cpu: GraphData,
    target_cpu: GraphData,
    run_index: int,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, float]:
    run_seed = args.seed + run_index
    set_seed(run_seed)
    source = source_cpu.to(device)
    target = target_cpu.to(device)
    model, mapper = build_model(args, source, device)
    global_mi = DomainMILoss().to(device)
    conditional_mi = ConditionalDomainMILoss().to(device)

    parameters = list(model.parameters())
    if not args.disable_mapper:
        parameters += list(mapper.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    cfg = SDBEConfig(use_mapper=not args.disable_mapper)

    tracked_weight = normalize_rows(model.classifier.weight.detach(), cfg.eps)
    lowpass_weight = tracked_weight.clone()
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        mapper.train()
        optimizer.zero_grad(set_to_none=True)

        if args.lambda_contrast > 0.0:
            contrast_source = graph_contrastive_loss(
                model,
                source,
                batch_size=args.contrast_batch_size,
                max_nodes=args.contrast_max_nodes,
            )
            contrast_target = graph_contrastive_loss(
                model,
                target,
                batch_size=args.contrast_batch_size,
                max_nodes=args.contrast_max_nodes,
            )
        else:
            contrast_source = torch.zeros((), device=device)
            contrast_target = torch.zeros((), device=device)

        base_probs_source, _, embeddings_source = model(source.x, source.edge_index)
        base_probs_target, _, embeddings_target = model(target.x, target.edge_index)
        rho = curriculum_strength(epoch, args.warmup_epochs, args.ramp_epochs, args.rho_max)
        eta = tracking_strength(
            epoch,
            args.warmup_epochs,
            args.ramp_epochs,
            args.tracking_eta_min,
            args.tracking_eta_max,
        )

        if rho > 0.0:
            state = build_boundary_state(
                embeddings_source,
                source.y,
                embeddings_target,
                base_probs_target,
                model.classifier.weight,
                source.edge_index,
                target.edge_index,
                rho,
                cfg,
                None if args.disable_mapper else mapper,
            )
            reference_weight = state["reference_weight"]
        else:
            state = None
            reference_weight = normalize_rows(model.classifier.weight, cfg.eps)

        if args.disable_tracking:
            momentum = 0.0
            lowpass_weight = normalize_rows(reference_weight.detach(), cfg.eps)
            tracked_weight = lowpass_weight.clone()
            track_loss = torch.zeros((), device=device)
        else:
            momentum = adaptive_ema_momentum(
                reference_weight,
                lowpass_weight,
                base=args.ema_base,
                sensitivity=args.ema_sensitivity,
                minimum=args.ema_min,
                maximum=args.ema_max,
                eps=cfg.eps,
            )
            lowpass_weight = update_ema(lowpass_weight, reference_weight, momentum, cfg.eps)
            tracked_weight = update_tracked_weight(tracked_weight, lowpass_weight, eta, cfg.eps)
            track_loss = tracking_alignment_loss(model.classifier.weight, lowpass_weight, cfg.eps)

        logits_source = model.logits_from_embeddings(embeddings_source, reference_weight)
        probs_source = F.softmax(logits_source, dim=1)
        logits_target = model.logits_from_embeddings(embeddings_target, tracked_weight)
        probs_target = F.softmax(logits_target, dim=1)

        source_loss = F.cross_entropy(logits_source, source.y)
        mi_loss = domain_information_loss(
            args.domain_info_mode,
            global_mi,
            conditional_mi,
            probs_source,
            probs_target,
            source.y,
        )
        margin_loss = (
            soft_margin_loss(logits_target, probs_target.detach())
            if args.lambda_margin > 0.0
            else torch.zeros((), device=device)
        )
        smooth_loss = (
            graph_smoothness(probs_target, target.edge_index)
            if args.lambda_smooth > 0.0
            else torch.zeros((), device=device)
        )

        total_loss = (
            args.lambda_contrast * (contrast_source + contrast_target)
            + args.lambda_source * source_loss
            + args.lambda_mi * mi_loss
            + args.lambda_track * track_loss
            + args.lambda_margin * margin_loss
            + args.lambda_smooth * smooth_loss
        )
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {total_loss.item()}")
        total_loss.backward()
        optimizer.step()

        row = {
            "epoch": epoch,
            "loss_total": float(total_loss.detach().item()),
            "loss_contrast_source": float(contrast_source.detach().item()),
            "loss_contrast_target": float(contrast_target.detach().item()),
            "loss_source": float(source_loss.detach().item()),
            "loss_mi": float(mi_loss.detach().item()),
            "loss_track": float(track_loss.detach().item()),
            "loss_margin": float(margin_loss.detach().item()),
            "loss_smooth": float(smooth_loss.detach().item()),
            "rho": float(rho),
            "eta": float(eta),
            "ema_momentum": float(momentum),
            "node_displacement_mean": 0.0 if state is None else float(state["node_displacement"].norm(dim=1).mean().detach().item()),
            "edge_tension_mean": 0.0 if state is None else float(state["edge_tension"].norm(dim=1).mean().detach().item()),
            "covariance_similarity_mean": 0.0 if state is None else float(state["covariance_similarity"].mean().detach().item()),
        }
        log_rows.append(row)
        if epoch == 1 or epoch == args.epochs or epoch % args.log_interval == 0:
            print(
                f"run={run_index + 1} epoch={epoch:04d} loss={row['loss_total']:.4f} "
                f"src={row['loss_source']:.4f} mi={row['loss_mi']:.4f} "
                f"rho={row['rho']:.3f} node={row['node_displacement_mean']:.4f} "
                f"edge={row['edge_tension_mean']:.4f}"
            )

    lowpass_weight, tracked_weight, final_state = refresh_tracked_head(
        args, cfg, model, mapper, source, target, lowpass_weight, tracked_weight
    )
    source_metrics = evaluate(model, source, tracked_weight)
    target_metrics = evaluate(model, target, tracked_weight)

    run_dir = output_dir / f"run_{run_index + 1:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    checkpoint = {
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "mapper_state_dict": copy.deepcopy(mapper.state_dict()),
        "tracked_classifier_weight": tracked_weight.detach().cpu(),
        "lowpass_classifier_weight": lowpass_weight.detach().cpu(),
        "args": vars(args),
        "source_name": source.name,
        "target_name": target.name,
        "seed": run_seed,
    }
    torch.save(checkpoint, run_dir / "checkpoint.pt")

    summary = {
        "run": run_index + 1,
        "seed": run_seed,
        "source_accuracy": source_metrics["accuracy"],
        "source_macro_f1": source_metrics["macro_f1"],
        "target_accuracy": target_metrics["accuracy"],
        "target_macro_f1": target_metrics["macro_f1"],
        "target_micro_f1": target_metrics["micro_f1"],
        "final_node_displacement_mean": float(final_state["node_displacement"].norm(dim=1).mean().item()),
        "final_edge_tension_mean": float(final_state["edge_tension"].norm(dim=1).mean().item()),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"run={run_index + 1} target Acc={summary['target_accuracy']:.2f} "
        f"Macro-F1={summary['target_macro_f1']:.2f} Micro-F1={summary['target_micro_f1']:.2f}"
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.runs < 1:
        raise ValueError("epochs and runs must be positive")
    device = resolve_device(args.device)
    source = load_mat_graph(
        args.source,
        row_normalize_features=args.row_normalize_features,
        make_undirected=args.make_undirected,
    )
    target = load_mat_graph(
        args.target,
        row_normalize_features=args.row_normalize_features,
        make_undirected=args.make_undirected,
    )
    if source.num_features != target.num_features:
        raise ValueError(
            f"Feature dimensions differ: source={source.num_features}, target={target.num_features}"
        )
    if source.num_classes != target.num_classes:
        raise ValueError(
            f"Class counts differ: source={source.num_classes}, target={target.num_classes}"
        )

    output_dir = Path(args.output_dir) / f"{source.name}_to_{target.name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"source={source.name}({source.num_nodes} nodes) target={target.name}({target.num_nodes} nodes) "
        f"features={source.num_features} classes={source.num_classes} device={device}"
    )

    summaries = [
        train_run(args, source, target, run_index, output_dir, device)
        for run_index in range(args.runs)
    ]
    aggregate = {
        "runs": args.runs,
        "target_accuracy_mean": float(np.mean([item["target_accuracy"] for item in summaries])),
        "target_accuracy_std": float(np.std([item["target_accuracy"] for item in summaries])),
        "target_macro_f1_mean": float(np.mean([item["target_macro_f1"] for item in summaries])),
        "target_macro_f1_std": float(np.std([item["target_macro_f1"] for item in summaries])),
        "target_micro_f1_mean": float(np.mean([item["target_micro_f1"] for item in summaries])),
        "target_micro_f1_std": float(np.std([item["target_micro_f1"] for item in summaries])),
    }
    (output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
