#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Integration tests for TorchCommMSCCLPP reduce().
#
# Requires a plan directory with an "allreduce" plan (reduce is implemented
# as all_reduce under the hood). Uses the bundled allreduce_4gpu.json plan.
#
# Run with:
#   torchrun --nproc_per_node=4 \
#     comms/torchcomms/mscclpp/tests/integration/py/test_mscclpp_reduce.py

import os
import shutil
import sys
import tempfile

import torch
import torchcomms

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_BUNDLED_PLAN = os.path.join(_TEST_DIR, "allreduce_4gpu.json")


def setup_plan_dir() -> tempfile.TemporaryDirectory:
    if not os.path.exists(_BUNDLED_PLAN):
        raise FileNotFoundError(
            f"Bundled 4-GPU allreduce plan not found: {_BUNDLED_PLAN}"
        )
    tmp = tempfile.TemporaryDirectory(prefix="mscclpp_plans_")
    shutil.copy(_BUNDLED_PLAN, os.path.join(tmp.name, "allreduce.json"))
    return tmp


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    tmp_plan_dir: tempfile.TemporaryDirectory | None = None
    if "TORCHCOMM_MSCCLPP_PLAN_DIR" not in os.environ:
        tmp_plan_dir = setup_plan_dir()
        os.environ["TORCHCOMM_MSCCLPP_PLAN_DIR"] = tmp_plan_dir.name

    comm = torchcomms.new_comm("mscclpp", device, name="reduce_test")

    failed = False

    # ------------------------------------------------------------------
    # Test 1: SUM reduce, root=0
    # Each rank contributes rank+1; expected sum = world_size*(world_size+1)/2
    # ------------------------------------------------------------------
    n_elems = 256 * 1024
    fill_value = float(rank + 1)
    tensor = torch.full((n_elems,), fill_value, dtype=torch.float32, device=device)
    expected_sum = float(world_size * (world_size + 1) // 2)

    work = comm.reduce(tensor, 0, torchcomms.ReduceOp.SUM, False)
    work.wait()

    if rank == 0:
        actual = tensor[0].item()
        ok = abs(actual - expected_sum) < 1e-3
        print(
            f"[rank {rank}] reduce SUM root=0: result={actual:.1f} "
            f"expected={expected_sum:.1f} {'PASS' if ok else 'FAIL'}",
            flush=True,
        )
        if not ok:
            failed = True

    # ------------------------------------------------------------------
    # Test 2: async reduce
    # ------------------------------------------------------------------
    tensor2 = torch.full(
        (n_elems,), float(rank + 1), dtype=torch.float32, device=device
    )
    work2 = comm.reduce(tensor2, 0, torchcomms.ReduceOp.SUM, True)
    work2.wait()

    if rank == 0:
        actual2 = tensor2[0].item()
        ok2 = abs(actual2 - expected_sum) < 1e-3
        print(
            f"[rank {rank}] reduce SUM async: result={actual2:.1f} "
            f"expected={expected_sum:.1f} {'PASS' if ok2 else 'FAIL'}",
            flush=True,
        )
        if not ok2:
            failed = True

    # ------------------------------------------------------------------
    # Test 3: non-SUM op must raise
    # ------------------------------------------------------------------
    tensor3 = torch.ones(64, dtype=torch.float32, device=device)
    raised = False
    try:
        comm.reduce(tensor3, 0, torchcomms.ReduceOp.MAX, False)
    except RuntimeError:
        raised = True

    if rank == 0:
        print(
            f"[rank {rank}] reduce MAX raises: {'PASS' if raised else 'FAIL'}",
            flush=True,
        )
        if not raised:
            failed = True

    comm.finalize()

    if tmp_plan_dir is not None:
        tmp_plan_dir.cleanup()

    if rank == 0:
        print("[rank 0] All done.", flush=True)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
