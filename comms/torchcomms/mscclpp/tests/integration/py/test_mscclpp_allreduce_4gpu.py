#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# End-to-end all_reduce test using the MSCCL++ backend.
#
# Uses allreduce_4gpu.json bundled alongside this test (generated with
# MSCCL++'s allreduce.py plan generator for 4 GPUs, memory channels only).
#
# Run with:
#   torchrun --nproc_per_node=4 \
#     comms/torchcomms/mscclpp/tests/integration/py/test_mscclpp_allreduce_4gpu.py
#
# Override the plan directory via TORCHCOMM_MSCCLPP_PLAN_DIR:
#   TORCHCOMM_MSCCLPP_PLAN_DIR=/path/to/plans torchrun --nproc_per_node=4 \
#     comms/torchcomms/mscclpp/tests/integration/py/test_mscclpp_allreduce_4gpu.py

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
    tmp = tempfile.TemporaryDirectory(prefix="mscclpp_plans_")
    shutil.copy(_BUNDLED_PLAN, os.path.join(tmp.name, "allreduce.json"))
    return tmp


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------
def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if world_size != 4:
        if rank == 0:
            print(
                f"[WARNING] This test is designed for exactly 4 ranks "
                f"(got {world_size}). The 4-GPU plan may not match.",
                flush=True,
            )

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Stage plans on rank 0 then let all ranks inherit via env var.
    # Because each rank is an independent process under torchrun we must
    # set up the plan dir before calling new_comm().
    tmp_plan_dir: tempfile.TemporaryDirectory | None = None
    if "TORCHCOMM_MSCCLPP_PLAN_DIR" not in os.environ:
        tmp_plan_dir = setup_plan_dir()
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

    # Barrier-style: wait for all ranks to print before the process group
    # is torn down (torchrun's own sync handles this).
    if rank == 0:
        print("[rank 0] All done.", flush=True)

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
