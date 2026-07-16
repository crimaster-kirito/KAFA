"""One-round, tokenizer-agnostic federated distillation development runner.

Clients never exchange model parameters. Each client sends four canonical AG
News probabilities for each public anchor text; the server trains its own LoRA
adapter against the mean soft target. This is a transparent compatible baseline
for model-family heterogeneity, not a claim about privacy or production FL.
"""

import argparse
import hashlib
import io
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script.run_agnews_local_dev import (  # noqa: E402
    candidate_example, evaluate, select_client_train, select_manifest_rows)
from kafa_script.run_agnews_mixture_dev import (  # noqa: E402
    build_mixture, encode_mixture_example)
from kafa_script.diagnose_anchor_gradient_separation import (  # noqa: E402
    gradient_cosine, parameter_gradients, rank_percentiles, summary_quantiles)
from kafa_script.run_real_data_plumbing_smoke import (  # noqa: E402
    collate, encode_training_example, load_jsonl, sha256_file)


def candidate_loss_matrix(model, tokenizer, rows, device, max_length,
                          batch_size, requires_grad=False):
    """Canonical-label NLL matrix, shape [len(rows), 4]."""
    candidates = [candidate_example(tokenizer, {"text": row["text"]}, label,
                                    max_length)
                  for row in rows for label in range(4)]
    loader = DataLoader(candidates, batch_size=batch_size * 4, shuffle=False,
                        collate_fn=lambda batch: collate(tokenizer, batch))
    chunks = []
    context = torch.enable_grad() if requires_grad else torch.no_grad()
    with context:
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits[:, :-1]
            targets = batch["input_ids"][:, 1:]
            mask = batch["labels"][:, 1:] != -100
            token_losses = F.cross_entropy(logits.transpose(1, 2), targets,
                                           reduction="none")
            chunks.append((token_losses * mask).sum(1) /
                          mask.sum(1).clamp_min(1))
    return torch.cat(chunks).reshape(len(rows), 4)


def make_lora(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")).to(device)
    return model, tokenizer


def train_client(model, tokenizer, mixture, device, steps, batch_size,
                 max_length, learning_rate, seed):
    encoded = [encode_mixture_example(tokenizer, item, max_length)
               for item in mixture]
    loader = DataLoader(encoded, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed),
                        collate_fn=lambda batch: collate(tokenizer, batch))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate)
    iterator = iter(loader)
    model.train()
    losses = []
    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return sum(losses) / len(losses)


def route_client_mixture(model, tokenizer, mixture, anchor_rows, device,
                         max_length, keep_fraction, gate_threshold):
    """Relative anchor routing without inspecting hidden contamination labels."""
    parameters = [parameter for parameter in model.parameters()
                  if parameter.requires_grad]
    grouped = defaultdict(list)
    for row in anchor_rows:
        grouped[int(row["label"])].append(row)
    anchor_gradients = {}
    for label in range(4):
        encoded = [encode_training_example(tokenizer, row, max_length)
                   for row in grouped[label]]
        batch = collate(tokenizer, encoded)
        batch = {key: value.to(device) for key, value in batch.items()}
        model.zero_grad(set_to_none=True)
        model(**batch).loss.backward()
        anchor_gradients[label] = parameter_gradients(parameters)
    reference = []
    for label in range(4):
        for row in grouped[label]:
            batch = collate(tokenizer, [encode_training_example(
                tokenizer, row, max_length)])
            batch = {key: value.to(device) for key, value in batch.items()}
            model.zero_grad(set_to_none=True)
            model(**batch).loss.backward()
            reference.append(gradient_cosine(parameter_gradients(parameters),
                                              anchor_gradients[label]))
    records = []
    for item in mixture:
        encoded = encode_mixture_example(tokenizer, item, max_length)
        batch = collate(tokenizer, [encoded])
        batch = {key: value.to(device) for key, value in batch.items()}
        model.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        gradient = parameter_gradients(parameters)
        if item["kind"] == "off_task":
            cosine = max(gradient_cosine(gradient, anchor_gradients[label])
                         for label in range(4))
            claimed_label, semantic_margin = None, None
        else:
            claimed_label = int(item["training_label"])
            cosine = gradient_cosine(gradient,
                                     anchor_gradients[claimed_label])
            with torch.no_grad():
                candidate_losses = candidate_loss_matrix(
                    model, tokenizer, [{"text": item["text"]}], device,
                    max_length, 1).float().cpu()[0]
            semantic_margin = float(candidate_losses[claimed_label] -
                                    candidate_losses.min())
        records.append({
            "source_id": item["source_id"],
            "claimed_label": claimed_label,
            "loss": float(loss.detach()),
            "anchor_cosine": cosine,
            "semantic_margin": semantic_margin,
        })
    client_q25 = summary_quantiles(
        [record["anchor_cosine"] for record in records])["q25"]
    reference_q25 = summary_quantiles(reference)["q25"]
    ratio = client_q25 / max(reference_q25, 1e-12)
    if ratio >= gate_threshold:
        return mixture, {
            "gate_ratio": ratio, "client_q25": client_q25,
            "reference_q25": reference_q25, "gate_triggered": False,
            "selected": len(mixture),
        }
    semantic_observed = [record["semantic_margin"] for record in records
                         if record["semantic_margin"] is not None]
    off_task_value = max(semantic_observed, default=0.0) + 1.0
    signals = [
        rank_percentiles([-record["anchor_cosine"] for record in records]),
        rank_percentiles([record["loss"] for record in records]),
        rank_percentiles([off_task_value if record["semantic_margin"] is None
                          else record["semantic_margin"] for record in records]),
    ]
    scores = [sum(parts) / len(parts) for parts in zip(*signals)]
    quota = max(1, round(len(mixture) * keep_fraction / 4))
    selected_indices = set()
    for label in range(4):
        candidates = [index for index, record in enumerate(records)
                      if record["claimed_label"] == label]
        candidates.sort(key=lambda index: (scores[index],
                                           records[index]["source_id"]))
        selected_indices.update(candidates[:quota])
    selected = [item for index, item in enumerate(mixture)
                if index in selected_indices]
    return selected, {
        "gate_ratio": ratio, "client_q25": client_q25,
        "reference_q25": reference_q25, "gate_triggered": True,
        "selected": len(selected), "quota_per_claimed_label": quota,
        "score_record_order_sha256": hashlib.sha256("\n".join(
            record["source_id"] for record in records).encode()).hexdigest(),
    }


def select_disjoint_client_bases(rows, manifest, seed, clients, per_class=64):
    """Deterministically allocate non-overlapping target-task client data."""
    excluded = {
        record["id"] for split in ("anchor", "development")
        for record in manifest["splits"][split]["records"]
    }
    grouped = defaultdict(list)
    for row in rows:
        if row["_id"] not in excluded:
            rank = hashlib.sha256(
                f"hetero-distill-v1|{seed}|{row['_id']}".encode()).hexdigest()
            grouped[int(row["label"])].append((rank, row))
    bases = [[] for _ in range(clients)]
    for label in range(4):
        ranked = [row for _, row in sorted(grouped[label])]
        required = clients * per_class
        if len(ranked) < required:
            raise ValueError(f"insufficient class-{label} examples")
        for client_index in range(clients):
            start = client_index * per_class
            bases[client_index].extend(ranked[start:start + per_class])
    all_ids = [row["_id"] for base in bases for row in base]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("disjoint client allocation overlapped")
    return bases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-model", type=Path,
                        default=Path(".cache/models/Qwen2.5-1.5B"))
    parser.add_argument("--client-models", type=Path, nargs="+", required=True)
    parser.add_argument("--client-data-mode", choices=["common", "disjoint"],
                        default="disjoint")
    parser.add_argument("--client-routing",
                        choices=["none", "relative_balanced_rank"],
                        default="none")
    parser.add_argument("--routing-keep-fraction", type=float, default=0.5)
    parser.add_argument("--routing-gate-threshold", type=float, default=0.8)
    parser.add_argument("--manifest", type=Path,
                        default=Path("kafa_script/benchmark_manifest_v1.json"))
    parser.add_argument("--agnews-data", type=Path,
                        default=Path(".cache/datasets/agnews/train.jsonl"))
    parser.add_argument("--imdb-data", type=Path,
                        default=Path(".cache/datasets/imdb/train.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--client-steps", type=int, default=40)
    parser.add_argument("--server-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--server-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--anchor-score-batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--server-learning-rate", type=float, default=None)
    parser.add_argument("--anchor-supervision-weight", type=float, default=0.0,
                        help="Weight of public-anchor hard-label CE on server.")
    parser.add_argument("--distillation-weight", type=float, default=1.0,
                        help="Weight of averaged client probability loss on server.")
    parser.add_argument("--teacher-aggregation",
                        choices=["uniform", "anchor_accuracy"],
                        default="uniform")
    parser.add_argument("--teacher-temperature", type=float, default=0.1,
                        help="Softmax temperature for anchor-accuracy weighting.")
    parser.add_argument("--conflict-ratio", type=float, default=0.4)
    parser.add_argument("--off-task-ratio", type=float, default=0.2)
    parser.add_argument("--anchor-per-class", type=int, default=64)
    parser.add_argument("--eval-per-class", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.client_models:
        parser.error("at least one client model is required")
    if not 0 < args.routing_keep_fraction <= 1:
        parser.error("--routing-keep-fraction must be in (0, 1]")
    if args.server_batch_size is not None and args.server_batch_size <= 0:
        parser.error("--server-batch-size must be positive")
    if args.anchor_score_batch_size is not None and args.anchor_score_batch_size <= 0:
        parser.error("--anchor-score-batch-size must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    manifest = json.loads(args.manifest.read_text())
    agnews = load_jsonl(args.agnews_data)
    imdb = load_jsonl(args.imdb_data)
    anchor_rows = select_manifest_rows(agnews, manifest, "anchor",
                                       args.anchor_per_class)
    eval_rows = select_manifest_rows(agnews, manifest, "development",
                                     args.eval_per_class)
    client_probabilities, client_records = [], []
    anchor_score_batch_size = (args.eval_batch_size if args.anchor_score_batch_size
                               is None else args.anchor_score_batch_size)
    server_batch_size = (args.batch_size if args.server_batch_size is None
                         else args.server_batch_size)
    if args.client_data_mode == "disjoint":
        client_bases = select_disjoint_client_bases(
            agnews, manifest, args.seed, len(args.client_models))
    else:
        common = select_client_train(agnews, manifest, args.seed, 64)
        client_bases = [common] * len(args.client_models)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for client_index, path in enumerate(args.client_models):
        base = client_bases[client_index]
        mixture_seed = args.seed + client_index * 1009
        mixture = build_mixture(base, imdb, mixture_seed, args.conflict_ratio,
                                args.off_task_ratio)
        model, tokenizer = make_lora(path, device)
        if args.client_routing == "relative_balanced_rank":
            train_mixture, routing = route_client_mixture(
                model, tokenizer, mixture, anchor_rows, device, args.max_length,
                args.routing_keep_fraction, args.routing_gate_threshold)
        else:
            train_mixture, routing = mixture, {"gate_triggered": False,
                                                "selected": len(mixture)}
        loss = train_client(model, tokenizer, train_mixture, device,
                            args.client_steps, args.batch_size,
                            args.max_length, args.learning_rate,
                            args.seed * 100 + client_index)
        model.eval()
        losses = candidate_loss_matrix(model, tokenizer, anchor_rows, device,
                                       args.max_length, anchor_score_batch_size)
        probabilities = torch.softmax(-losses.float(), dim=1).cpu()
        client_probabilities.append(probabilities)
        buffer = io.BytesIO()
        torch.save(probabilities, buffer)
        client_records.append({
            "model": str(path),
            "model_weight_sha256": sha256_file(path / "model.safetensors"),
            "mean_client_train_loss": loss,
            "anchor_probability_payload_bytes": buffer.tell(),
            "train_examples": len(mixture),
            "selected_train_examples": len(train_mixture),
            "selected_realized_counts": dict(sorted(Counter(
                item["kind"] for item in train_mixture).items())),
            "routing": routing,
            "anchor_label_accuracy": float((probabilities.argmax(1) ==
                torch.tensor([int(row["label"]) for row in anchor_rows])).float().mean()),
            "anchor_label_nll": float(-probabilities[torch.arange(
                len(anchor_rows)), torch.tensor([int(row["label"]) for row
                                                  in anchor_rows])].clamp_min(1e-12).log().mean()),
            "base_target_ids_sha256": hashlib.sha256("\n".join(sorted(
                row["_id"] for row in base)).encode()).hexdigest(),
            "mixture_assignment_sha256": hashlib.sha256("\n".join(
                f"{item['kind']}:{item['source_id']}:{item['answer']}"
                for item in mixture).encode()).hexdigest(),
        })
        del model
        torch.cuda.empty_cache()

    if args.teacher_aggregation == "uniform":
        teacher_weights = torch.full((len(client_probabilities),),
                                     1 / len(client_probabilities))
    else:
        if args.teacher_temperature <= 0:
            parser.error("--teacher-temperature must be positive")
        quality = torch.tensor([record["anchor_label_accuracy"]
                                for record in client_records])
        teacher_weights = torch.softmax(quality / args.teacher_temperature,
                                        dim=0)
    teacher = sum(weight * probabilities for weight, probabilities in
                  zip(teacher_weights, client_probabilities))
    server, server_tokenizer = make_lora(args.server_model, device)
    initial = evaluate(server, server_tokenizer, eval_rows, device,
                       args.max_length, args.eval_batch_size)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in server.parameters() if parameter.requires_grad],
        lr=(args.learning_rate if args.server_learning_rate is None
            else args.server_learning_rate))
    order = torch.randperm(len(anchor_rows), generator=torch.Generator().manual_seed(
        args.seed))
    cursor = 0
    server_losses = []
    server.train()
    for _ in range(args.server_steps):
        indices = order[cursor:cursor + server_batch_size]
        if len(indices) < server_batch_size:
            order = torch.randperm(len(anchor_rows), generator=torch.Generator().manual_seed(
                args.seed + len(server_losses) + 1))
            cursor = 0
            indices = order[:server_batch_size]
        cursor += server_batch_size
        row_batch = [anchor_rows[int(index)] for index in indices]
        target = teacher[indices].to(device)
        hard_target = torch.tensor([int(row["label"]) for row in row_batch],
                                   device=device)
        optimizer.zero_grad(set_to_none=True)
        losses = candidate_loss_matrix(server, server_tokenizer, row_batch,
                                       device, args.max_length,
                                       server_batch_size, requires_grad=True)
        log_probs = F.log_softmax(-losses, dim=1)
        distillation_loss = -(target * log_probs).sum(1).mean()
        anchor_loss = F.cross_entropy(-losses, hard_target)
        loss = (args.distillation_weight * distillation_loss +
                args.anchor_supervision_weight * anchor_loss)
        loss.backward()
        optimizer.step()
        server_losses.append(float(loss.detach()))
    final = evaluate(server, server_tokenizer, eval_rows, device,
                     args.max_length, args.eval_batch_size)
    elapsed = time.perf_counter() - started
    communication = sum(record["anchor_probability_payload_bytes"]
                        for record in client_records)
    result = {
        "evidence_label": ("benchmark_dev" if args.client_data_mode == "disjoint"
                           else "smoke_real"),
        "paper_result": False,
        "method": ("public_anchor_only" if args.distillation_weight == 0
                   else "canonical_probability_distillation"),
        "scope": ("one-round heterogeneous development baseline with disjoint client target data"
                  if args.client_data_mode == "disjoint" else
                  "one-round heterogeneous compatibility smoke; clients share the same selected mixture"),
        "seed": args.seed,
        "manifest_sha256": sha256_file(args.manifest),
        "server_model": str(args.server_model),
        "server_weight_sha256": sha256_file(args.server_model / "model.safetensors"),
        "client_records": client_records,
        "client_count": len(client_records),
        "client_data_mode": args.client_data_mode,
        "client_routing": args.client_routing,
        "routing_keep_fraction": args.routing_keep_fraction,
        "routing_gate_threshold": args.routing_gate_threshold,
        "anchor_examples": len(anchor_rows),
        "eval_examples": len(eval_rows),
        "client_steps": args.client_steps,
        "server_steps": args.server_steps,
        "client_batch_size": args.batch_size,
        "server_batch_size": server_batch_size,
        "anchor_score_batch_size": anchor_score_batch_size,
        "client_learning_rate": args.learning_rate,
        "server_learning_rate": (args.learning_rate if args.server_learning_rate
                                  is None else args.server_learning_rate),
        "anchor_supervision_weight": args.anchor_supervision_weight,
        "distillation_weight": args.distillation_weight,
        "teacher_aggregation": args.teacher_aggregation,
        "teacher_temperature": args.teacher_temperature,
        "teacher_weights": teacher_weights.tolist(),
        "mean_server_distillation_loss": sum(server_losses) / len(server_losses),
        "initial": initial,
        "final": final,
        "measured_client_to_server_payload_bytes": communication,
        "total_wall_seconds": elapsed,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        "limitations": [
            "One communication round only; this is not a convergence study.",
            "Anchor texts are public; hard anchor labels are used only when anchor_supervision_weight is positive.",
            "This measures compatibility, not privacy, convergence, or a final federated result."
        ]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
