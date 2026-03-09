#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Example: Using MSCCL++ and NCCL backends together in a PyTorch training loop.
#
# This demonstrates the expected deployment pattern where MSCCL++ handles
# gradient allreduce (its strongest collective) while NCCL handles everything
# else (broadcast for parameter sync, barrier, etc.).
#
# The example simulates a simple data-parallel training loop:
#   1. Forward pass: each rank computes a local loss
#   2. Backward pass: compute local gradients
#   3. Gradient sync: MSCCL++ allreduce (high-performance path)
#   4. Parameter broadcast: NCCL broadcast (MSCCL++ doesn't support this)
#   5. Repeat for N steps
#
# Run with:
#   torchrun --nproc_per_node=N \
#     comms/torchcomms/mscclpp/tests/integration/py/test_mscclpp_training_loop.py

import os
import sys

import torch
import torch.nn as nn
import torchcomms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mscclpp_plan_gen import generate_plans


def _cc_backend() -> str:
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        return "rccl"
    return "nccl"


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # ── Plan generation ────────────────────────────────────────────
    tmp_plan_dir = None
    if "TORCHCOMM_MSCCLPP_PLAN_DIR" not in os.environ:
        tmp_plan_dir = generate_plans(world_size)
        os.environ["TORCHCOMM_MSCCLPP_PLAN_DIR"] = tmp_plan_dir.name

    # ── Create communicators ───────────────────────────────────────
    # MSCCL++ for gradient allreduce (high-performance path)
    # NCCL for everything else (broadcast, barrier, etc.)
    cc_backend = _cc_backend()

    mscclpp_comm = torchcomms.new_comm("mscclpp", device, name="grad_sync")
    nccl_comm = torchcomms.new_comm(cc_backend, device, name="param_sync")

    # ── Simple model ───────────────────────────────────────────────
    # A small MLP that each rank will train on different data.
    # In real training this would be a large model with data parallelism.
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
    ).to(device)

    min_mscclpp_bytes = 4096  # 4KB minimum for MSCCL++ alignment safety

    if rank == 0:
        print(
            f"Training with {world_size} GPUs\n"
            f"  Gradient allreduce: MSCCL++ (large tensors) + {cc_backend.upper()} (small tensors)\n"
            f"  Parameter broadcast: {cc_backend.upper()}\n"
            f"  Tensor routing ({min_mscclpp_bytes} byte threshold):",
            flush=True,
        )
        for name, p in model.named_parameters():
            backend = "MSCCL++" if p.nbytes >= min_mscclpp_bytes else cc_backend.upper()
            print(
                f"    {name}: {list(p.shape)} ({p.nbytes} bytes) -> {backend}",
                flush=True,
            )
        print(flush=True)

    # Broadcast initial parameters from rank 0 so all ranks start identical.
    if rank == 0:
        print("  Initial param broadcast (NCCL):", flush=True)
    for name, param in model.named_parameters():
        nccl_comm.broadcast(param.data, root=0, async_op=False).wait()
        if rank == 0:
            print(
                f"    [NCCL]    broadcast {name} ({param.data.nbytes} bytes)",
                flush=True,
            )

    loss_fn = nn.MSELoss()
    lr = 0.01
    num_steps = 5

    # ── Training loop ──────────────────────────────────────────────
    for step in range(num_steps):
        # --- Forward pass (each rank gets different data) ---
        torch.manual_seed(step * world_size + rank)
        x = torch.randn(32, 256, device=device)
        target = torch.randn(32, 128, device=device)

        output = model(x)
        loss = loss_fn(output, target)

        # --- Backward pass ---
        model.zero_grad()
        loss.backward()

        # --- Gradient allreduce via MSCCL++ ---
        # This is the hot path in data-parallel training.
        if rank == 0 and step == 0:
            print("  Step 0 gradient allreduce:", flush=True)
        # faster than NCCL for allreduce on certain topologies because its
        # execution plans are tuned to the specific GPU interconnect.
        #
        # MSCCL++ executor plans require tensor sizes to be aligned with the
        # chunk count (num_gpus * num_threadblocks). Small tensors (like bias
        # vectors) may not meet this requirement, so we fall back to NCCL for
        # those. In production, a CommRouter would handle this automatically.
        grad_works = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                if param.grad.data.nbytes >= min_mscclpp_bytes:
                    # Large tensor → MSCCL++ (high-performance path)
                    work = mscclpp_comm.all_reduce(
                        param.grad.data,
                        torchcomms.ReduceOp.SUM,
                        True,  # async_op
                    )
                    if rank == 0 and step == 0:
                        print(
                            f"    [MSCCL++] allreduce {name} "
                            f"({param.grad.data.nbytes} bytes)",
                            flush=True,
                        )
                else:
                    # Small tensor → NCCL fallback
                    work = nccl_comm.all_reduce(
                        param.grad.data,
                        torchcomms.ReduceOp.SUM,
                        True,  # async_op
                    )
                    if rank == 0 and step == 0:
                        print(
                            f"    [NCCL]    allreduce {name} "
                            f"({param.grad.data.nbytes} bytes)",
                            flush=True,
                        )
                grad_works.append(work)

        # Wait for all gradient syncs to complete
        for work in grad_works:
            work.wait()

        # Average gradients (allreduce computes sum, divide by world_size)
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data /= world_size

        # --- SGD step ---
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param.data -= lr * param.grad.data

        # --- Verify all ranks have the same parameters ---
        # Broadcast rank 0's params and compare (using NCCL since MSCCL++
        # doesn't support broadcast).
        all_match = True
        for param in model.parameters():
            ref = param.data.clone()
            nccl_comm.broadcast(ref, root=0, async_op=False).wait()
            if not torch.allclose(param.data, ref, atol=1e-6):
                all_match = False

        local_loss = loss.item()
        if rank == 0:
            print(
                f"  Step {step}: loss={local_loss:.4f}  "
                f"params_synced={'YES' if all_match else 'NO'}",
                flush=True,
            )

        if not all_match:
            print(
                f"[rank {rank}] ERROR: Parameters diverged at step {step}!",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)

    # ── Cleanup ────────────────────────────────────────────────────
    mscclpp_comm.finalize()
    nccl_comm.finalize()

    if tmp_plan_dir is not None:
        tmp_plan_dir.cleanup()

    if rank == 0:
        print(
            f"\nTraining complete: {num_steps} steps, "
            f"all ranks synchronized.  PASS",
            flush=True,
        )


if __name__ == "__main__":
    main()
