#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# End-to-end all_reduce test using the MSCCL++ backend.
#
# Generates execution plans at test time via the MSCCL++ language DSL so the
# test works with any GPU count (no hardcoded 4-GPU plan dependency).
#
# Run with:
#   torchrun --nproc_per_node=N \
#     comms/torchcomms/mscclpp/tests/integration/py/test_mscclpp_allreduce.py
#
# Override the plan directory via TORCHCOMM_MSCCLPP_PLAN_DIR:
#   TORCHCOMM_MSCCLPP_PLAN_DIR=/path/to/plans torchrun --nproc_per_node=N \
#     comms/torchcomms/mscclpp/tests/integration/py/test_mscclpp_allreduce.py

import os
import sys

import torch
import torchcomms

# Plan generation helper lives alongside this test.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mscclpp_plan_gen import generate_plans


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------
def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Generate plans for the actual world size (unless already provided).
    tmp_plan_dir = None
    if "TORCHCOMM_MSCCLPP_PLAN_DIR" not in os.environ:
        tmp_plan_dir = generate_plans(world_size)
        os.environ["TORCHCOMM_MSCCLPP_PLAN_DIR"] = tmp_plan_dir.name

    # ------------------------------------------------------------------
    # Create communicator
    # ------------------------------------------------------------------
    if rank == 0:
        print(
            f"[rank {rank}] Creating MSCCL++ communicator  "
            f"world_size={world_size}  plan_dir={os.environ['TORCHCOMM_MSCCLPP_PLAN_DIR']}",
            flush=True,
        )

    comm = torchcomms.new_comm("mscclpp", device, name="allreduce_test")

    if rank == 0:
        print(
            f"[rank {rank}] Communicator created: "
            f"backend={comm.get_backend()} world_size={world_size}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # All-reduce: each rank contributes its rank+1 as a float
    # Expected result: sum(1..world_size) = world_size*(world_size+1)/2
    # ------------------------------------------------------------------
    n_elems = 256 * 1024  # 1 MB
    fill_value = float(rank + 1)
    tensor = torch.full((n_elems,), fill_value, dtype=torch.float32, device=device)

    expected_sum = float(world_size * (world_size + 1) // 2)

    work = comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, False)
    work.wait()

    actual = tensor[0].item()
    ok = abs(actual - expected_sum) < 1e-3

    print(
        f"[rank {rank}] all_reduce result: {actual:.1f}  expected: {expected_sum:.1f}  {'PASS' if ok else 'FAIL'}",
        flush=True,
    )

    if not ok:
        print(
            f"[rank {rank}] MISMATCH: got {actual} expected {expected_sum}",
            file=sys.stderr,
            flush=True,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    comm.finalize()

    if tmp_plan_dir is not None:
        tmp_plan_dir.cleanup()

    if rank == 0:
        print("[rank 0] All done.", flush=True)

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
