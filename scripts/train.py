"""
DDP training script for the ResGRU heavy-hexagon decoder.

Single GPU
----------
python scripts/train.py \\
    --data_dir /path/to/data \\
    --distance 5 \\
    --rounds 3 5 7 9 11 13 15 \\
    --hidden_size 256 --num_layers 2 \\
    --train_mode stim --stim_noise 1e-3

Multi-GPU (torchrun)
--------------------
torchrun --nproc_per_node=4 scripts/train.py \\
    --data_dir /path/to/data \\
    --distance 5 \\
    --rounds 3 5 7 9 11 13 15 \\
    --hidden_size 256 --num_layers 2 \\
    --train_mode stim --stim_noise 1e-3 \\
    --stim_train_samples 10000000
"""

import argparse
import copy
import os
import time
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from hhd.data.datasets import HHLazyDataset, HHStimDataset, collate_fn, load_hh_packages
from hhd.models.gru_decoder import Branched_GRU


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def ddp_print(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)


def setup_distributed(backend: str = "nccl") -> int:
    if is_dist_avail_and_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return local_rank
    dist.init_process_group(backend=backend, init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed() -> None:
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    if is_dist_avail_and_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


# ---------------------------------------------------------------------------
# Misc utils
# ---------------------------------------------------------------------------

def format_seconds(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0].get("lr", 0.0))


def save_checkpoint(path, model, optimizer, scheduler, epoch, args, best_val_acc=None):
    state = {
        "model": (model.module if isinstance(model, DDP) else model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "args": vars(args),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
        },
    }
    torch.save(state, path)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    scheduler,
    epochs: int,
    train_sampler=None,
    val_loader=None,
    args=None,
    save_every: int = 0,
    checkpoint_dir: str = None,
    warmup_epochs: int = 0,
    save_name: str = "gru_decoder",
):
    """Main training loop with early stopping and optional periodic checkpointing.

    The ``best_model_wts`` deepcopy is rank-gated to the main process to
    prevent non-main DDP ranks from restoring stale (random) weights after
    allreduce.  The best weights are broadcast from rank 0 before the final
    ``load_state_dict`` so all ranks end up with identical parameters.
    """
    best_model_wts = None
    if is_main_process():
        best_model_wts = copy.deepcopy(
            model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        )

    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    total_start = time.time()

    for epoch in range(epochs):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_start = time.time()
        running_loss = torch.tensor(0.0, device=device)
        running_correct = torch.tensor(0.0, device=device)
        running_total = torch.tensor(0.0, device=device)

        iterator = (
            tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
            if is_main_process()
            else dataloader
        )

        for detection, final_det, initial_det, label, lengths in iterator:
            detection = detection.to(device, non_blocking=True)
            final_det = final_det.to(device, non_blocking=True)
            initial_det = initial_det.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True).unsqueeze(1)
            lengths = lengths.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(detection, final_det, initial_det, lengths)
            loss = criterion(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                batch = label.size(0)
                running_loss += loss.detach() * batch
                preds = (torch.sigmoid(logits) > 0.5).float()
                running_correct += (preds == label).sum()
                running_total += torch.as_tensor(batch, device=device, dtype=torch.float32)

        running_loss = all_reduce_sum(running_loss)
        running_correct = all_reduce_sum(running_correct)
        running_total = all_reduce_sum(running_total)

        epoch_time = time.time() - epoch_start
        epoch_loss = (running_loss / running_total).item()
        epoch_acc = (running_correct / running_total).item()
        samples_per_sec = float(running_total.item() / max(epoch_time, 1e-6))

        if scheduler is not None and epoch >= warmup_epochs:
            scheduler.step()

        ddp_print(
            f"[Epoch {epoch + 1}/{epochs}] "
            f"time={format_seconds(epoch_time)} | "
            f"train_loss={epoch_loss:.4f} | train_acc={epoch_acc * 100:.2f}% | "
            f"throughput={samples_per_sec:.1f} samples/s | lr={current_lr(optimizer):.6f}"
        )

        if val_loader is not None:
            val_time = time.time()
            val_acc = evaluate(
                model, val_loader, device, compute_f1=False, reduce_only=True, desc="Validation"
            )
            ddp_print(
                f"           val_acc={val_acc * 100:.2f}%, "
                f"val_time={format_seconds(time.time() - val_time)}"
            )

            improved = val_acc > best_val_acc
            flag = torch.tensor(1 if improved else 0, device=device)
            all_reduce_sum(flag)
            improved = flag.item() > 0

            if improved:
                best_val_acc = val_acc
                # Only rank 0 deepcopies to avoid stale weights on other ranks
                if is_main_process():
                    best_model_wts = copy.deepcopy(
                        model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                    )
                patience_counter = 0
            else:
                patience_counter += 1

            if is_main_process() and save_every and ((epoch + 1) % save_every == 0):
                tag = f"{save_name}_d{args.distance}_ep{epoch + 1:03d}.pt"
                path = os.path.join(checkpoint_dir or ".", tag)
                save_checkpoint(path, model, optimizer, scheduler, epoch + 1, args, best_val_acc)
                ddp_print(f"           checkpoint_saved={path}")

            if patience_counter >= patience:
                ddp_print(f"Early stopping at epoch {epoch + 1}.")
                break

    # Broadcast best weights from rank 0 so all ranks restore consistently
    raw = model.module if isinstance(model, DDP) else model
    for param_tensor in raw.state_dict():
        if is_main_process():
            t = best_model_wts[param_tensor].to(device)
        else:
            t = raw.state_dict()[param_tensor].clone()
        if is_dist_avail_and_initialized():
            dist.broadcast(t, src=0)
        raw.state_dict()[param_tensor].copy_(t)

    total_time = time.time() - total_start
    ddp_print(f"Total training time: {format_seconds(total_time)}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    verbose: bool = False,
    compute_f1: bool = False,
    reduce_only: bool = False,
    desc: str = "Evaluation",
) -> float:
    eval_time = time.time()
    model.eval()
    local_correct = torch.tensor(0.0, device=device)
    local_total = torch.tensor(0.0, device=device)

    preds_list, labels_list = [], []
    keep_lists = compute_f1 and is_main_process()

    iterator = tqdm(dataloader, desc=desc)
    for detection, final_det, initial_det, label, lengths in iterator:
        detection = detection.to(device, non_blocking=True)
        final_det = final_det.to(device, non_blocking=True)
        initial_det = initial_det.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True).unsqueeze(1)
        lengths = lengths.to(device, non_blocking=True)

        logits = model(detection, final_det, initial_det, lengths)
        preds = (torch.sigmoid(logits) > 0.5).float()
        local_correct += (preds == label).sum()
        local_total += label.numel()

        if keep_lists:
            preds_list.extend(preds.squeeze(1).cpu().numpy())
            labels_list.extend(label.squeeze(1).cpu().numpy())

    global_correct = all_reduce_sum(local_correct)
    global_total = all_reduce_sum(local_total)
    acc = (global_correct / global_total).item()

    if reduce_only:
        return acc

    if is_main_process():
        print(
            f"Accuracy: {acc * 100:.2f}% "
            f"({int(global_correct.item())}/{int(global_total.item())}), "
            f"eval_time={format_seconds(time.time() - eval_time)}"
        )
    return acc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DDP ResGRU decoder training")
    parser.add_argument("--data_dir",    type=str, required=True,
                        help="Directory with .npy validation/test files")
    parser.add_argument("--rounds",      type=int, nargs="+",
                        default=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
                        help="Syndrome round values to include")
    parser.add_argument("--distance",    type=int, required=True,
                        help="Code distance")
    parser.add_argument("--epochs",      type=int, default=100)
    parser.add_argument("--batch_size",  type=int, default=512,
                        help="Per-GPU batch size")
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers",  type=int, default=2)
    parser.add_argument("--workers",     type=int, default=8,
                        help="DataLoader workers per process")
    parser.add_argument("--save_name",   type=str, default="decoder")
    parser.add_argument("--save_every",  type=int, default=5,
                        help="Save checkpoint every N epochs (0 to disable)")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Checkpoint directory (defaults to data_dir)")
    parser.add_argument("--warmup_epochs",  type=int, default=5)

    # Training-data mode
    parser.add_argument("--train_mode", type=str, default="stim",
                        choices=["npy", "stim"],
                        help="'stim' for on-the-fly simulation, 'npy' for pre-generated files")
    parser.add_argument("--stim_train_samples", type=int, default=100_000_000,
                        help="Virtual epoch size when train_mode='stim'")
    parser.add_argument("--stim_noise",  type=float, default=1e-3,
                        help="Physical error rate p for on-the-fly Stim training")
    parser.add_argument("--stim_seed",   type=int, default=0,
                        help="Base RNG seed for on-the-fly dataset")
    parser.add_argument("--memory_basis", type=str, default="Z",
                        choices=["X", "Z"],
                        help="Memory experiment basis")

    args = parser.parse_args()

    local_rank = setup_distributed(backend="nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    detector_per_round = int(((args.distance - 1) * (args.distance + 3)) / 2)
    num_z = int((args.distance - 1) + ((args.distance - 1) ** 2) / 2)

    ddp_print(f"Distance            : {args.distance}")
    ddp_print(f"Detectors per round : {detector_per_round}")
    ddp_print(f"Z stabilizers       : {num_z}")
    ddp_print(f"Train mode          : {args.train_mode}")

    # ---- Disk dataset (val / test always use .npy) ----
    packages = load_hh_packages(args.data_dir, args.rounds, args.distance)
    disk_dataset = HHLazyDataset(packages, args.distance)

    g = torch.Generator().manual_seed(42)
    total = len(disk_dataset)
    val_size = 500_000
    test_size = int(0.05 * total)
    unused_size = total - val_size - test_size

    if unused_size < 0:
        raise ValueError(
            f"Dataset too small for requested val_size={val_size} and "
            f"test_size={test_size} (total={total})."
        )

    _, val_set, test_set = torch.utils.data.random_split(
        disk_dataset, [unused_size, val_size, test_size], generator=g
    )

    # ---- Training dataset ----
    if args.train_mode == "stim":
        ddp_print("Using on-the-fly Stim data for TRAIN, .npy data for VAL/TEST.")
        train_set = HHStimDataset(
            distance=args.distance,
            rounds=args.rounds,
            samples_per_epoch=args.stim_train_samples,
            noise_p=args.stim_noise,
            seed=args.stim_seed,
            memory_basis=args.memory_basis,
        )
    else:
        ddp_print("Using .npy data for TRAIN (held-out portion of disk dataset).")
        train_size_disk = int(0.7 * total)
        val_size_npy = int(0.20 * total)
        test_size_npy = total - train_size_disk - val_size_npy
        train_set, val_set, test_set = torch.utils.data.random_split(
            disk_dataset,
            [train_size_disk, val_size_npy, test_size_npy],
            generator=g,
        )

    # ---- Samplers ----
    train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=False)
    val_sampler = DistributedSampler(val_set, shuffle=False, drop_last=False)
    test_sampler = DistributedSampler(test_set, shuffle=False, drop_last=False)

    # ---- DataLoaders ----
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=(args.workers > 0),
        prefetch_factor=2 if args.workers > 0 else None,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=(args.workers > 0),
        prefetch_factor=2 if args.workers > 0 else None,
    )

    # ---- Model ----
    model = Branched_GRU(
        input_size=detector_per_round,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        z_stab=num_z,
    ).to(device)

    model = DDP(
        model,
        device_ids=[local_rank] if torch.cuda.is_available() else None,
        output_device=local_rank if torch.cuda.is_available() else None,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    cosine_epochs = max(1, args.epochs - args.warmup_epochs)
    scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-6)

    ckpt_dir = args.checkpoint_dir or args.data_dir
    if is_main_process():
        os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Train ----
    train(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        epochs=args.epochs,
        train_sampler=train_sampler,
        val_loader=val_loader,
        args=args,
        warmup_epochs=args.warmup_epochs,
        save_every=args.save_every,
        checkpoint_dir=ckpt_dir,
        save_name=args.save_name,
    )

    # ---- Test ----
    evaluate(model=model, dataloader=test_loader, device=device, verbose=True,
             compute_f1=True, desc="Testing")

    # ---- Save best model ----
    if is_main_process():
        final_path = os.path.join(
            args.data_dir, f"{args.save_name}_d{args.distance}_best.pt"
        )
        torch.save(
            (model.module if isinstance(model, DDP) else model).state_dict(),
            final_path,
        )
        print(f"Best model saved: {final_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
