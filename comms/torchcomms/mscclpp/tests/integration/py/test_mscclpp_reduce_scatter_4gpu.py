#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# End-to-end reduce_scatter_single test using the MSCCL++ backend.
#
# Implemented internally as all_reduce + cudaMemcpyAsync of rank's chunk,
# so it reuses the bundled allreduce_4gpu.json plan.
#
# Each rank i fills its input with float(i + 1) for all elements.
# After reduce_scatter_single (SUM), each rank's output chunk equals
# sum(1, 2, ..., world_size) = world_size * (world_size + 1) / 2 = 10.0.
#
# Run with:
#   torchrun --nproc_per_node=4 \
#     comms/torchcomms/mscclpp/tests/integration/py/test_mscclpp_reduce_scatter_4gpu.py

import os
import shutil
import sys
import tempfile

import torch
import torchcomms

# ---------------------------------------------------------------------------
# Plan setup
# ---------------------------------------------------------------------------
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_BUNDLED_PLAN = os.path.join(_TEST_DIR, "allreduce_4gpu.json")


def setup_plan_dir() -> tempfile.TemporaryDirectory:
    """
    Copy the bundled 4-GPU allreduce plan as 'allreduce.json' in a temp dir
    so selectPlan() picks it up by the bare collective name.
    Returns the TemporaryDirectory object; caller must keep it alive.
    """
    if not os.path.exists(_BUNDLED_PLAN):
        raise FileNotFoundError(
            f"Bundled 4-GPU allreduce plan not found: {_BUNDLED_PLAN}"
        )
    tmp = tempfile.TemporaryDirectory(prefix="mscclpp_rs_plans_")
    shutil.copy(_BUNDLED_PLAN, os.path.join(tmp.name, "allreduce.json"))
    return tmp


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------
def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    tmp_plan_dir: tempfile.TemporaryDirectory | None = None
    if "MSCCLPP_PLAN_DIR" not in os.environ:
        tmp_plan_dir = setup_plan_dir()
        os.environ["MSCCLPP_PLAN_DIR"] = tmp_plan_dir.name

    if rank == 0:
        print(
            f"[rank {rank}] Creating MSCCL++ communicator "
            f" world_size={world_size}  plan_dir={os.environ['MSCCLPP_PLAN_DIR']}",
            flush=True,
        )

    comm = torchcomms.new_comm("mscclpp", device, name="rs_4gpu_test")

    if rank == 0:
        print(
            f"[rank {rank}] Communicator created: backend=mscclpp world_size={world_size}",
            flush=True,
        )

    # chunk size per rank (number of float32 elements)
    chunk = 64
    expected = float(world_size * (world_size + 1) // 2)  # 10.0 for 4 ranks

    # --- sync test ---
    inp = torch.full(
        (world_size * chunk,), float(rank + 1), dtype=torch.float32, device=device
    )
    out = torch.zeros(chunk, dtype=torch.float32, device=device)
    work = comm.reduce_scatter_single(out, inp, torchcomms.ReduceOp.SUM, False)
    work.wait()
    assert torch.allclose(out, torch.full_like(out, expected)), (
        f"rank {rank}: reduce_scatter sync FAIL  got={out.tolist()[:4]}  expected={expected}"
    )
    print(f"[rank {rank}] reduce_scatter_single sync: PASS", flush=True)

    # --- async test ---
    inp2 = torch.full(
        (world_size * chunk,), float(rank + 1), dtype=torch.float32, device=device
    )
    out2 = torch.zeros(chunk, dtype=torch.float32, device=device)
    work2 = comm.reduce_scatter_single(out2, inp2, torchcomms.ReduceOp.SUM, True)
    work2.wait()
    assert torch.allclose(out2, torch.full_like(out2, expected)), (
        f"rank {rank}: reduce_scatter async FAIL  got={out2.tolist()[:4]}  expected={expected}"
    )
    print(f"[rank {rank}] reduce_scatter_single async: PASS", flush=True)

    # --- non-SUM raises ---
    inp3 = torch.full(
        (world_size * chunk,), float(rank + 1), dtype=torch.float32, device=device
    )
    out3 = torch.zeros(chunk, dtype=torch.float32, device=device)
    raised = False
    try:
        comm.reduce_scatter_single(out3, inp3, torchcomms.ReduceOp.MAX, False)
    except RuntimeError:
        raised = True
    assert raised, f"rank {rank}: reduce_scatter MAX should have raised"
    print(f"[rank {rank}] reduce_scatter_single MAX raises: PASS", flush=True)

    comm.finalize()

    if tmp_plan_dir is not None:
        tmp_plan_dir.cleanup()

    if rank == 0:
        print(f"[rank {rank}] All done.", flush=True)


if __name__ == "__main__":
    main()
