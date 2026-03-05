#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Integration tests for TorchCommMSCCLPP barrier().
#
# barrier() is implemented as a dummy all_reduce on a 1-element buffer,
# so it requires an "allreduce" plan. Uses the bundled allreduce_4gpu.json.
#
# Run with:
#   torchrun --nproc_per_node=4 \
#     comms/torchcomms/mscclpp/tests/integration/py/test_mscclpp_barrier.py

import os
import shutil
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
    if "MSCCLPP_PLAN_DIR" not in os.environ:
        tmp_plan_dir = setup_plan_dir()
        os.environ["MSCCLPP_PLAN_DIR"] = tmp_plan_dir.name

    comm = torchcomms.new_comm("mscclpp", device, name="barrier_test")

    # ------------------------------------------------------------------
    # Test 1: sync barrier does not hang
    # ------------------------------------------------------------------
    comm.barrier(False)
    if rank == 0:
        print(f"[rank {rank}] barrier sync: PASS", flush=True)

    # ------------------------------------------------------------------
    # Test 2: async barrier — wait() completes
    # ------------------------------------------------------------------
    work = comm.barrier(True)
    work.wait()
    if rank == 0:
        print(f"[rank {rank}] barrier async: PASS", flush=True)

    # ------------------------------------------------------------------
    # Test 3: multiple barriers in sequence
    # ------------------------------------------------------------------
    for i in range(5):
        comm.barrier(False)
    if rank == 0:
        print(f"[rank {rank}] barrier x5 sequence: PASS", flush=True)

    # Drain all pending GPU work on every rank before teardown.
    # Without this, fast ranks can enter finalize() while others are still
    # completing their last barrier, causing MSCCLPP bootstrap to hang.
    torch.cuda.synchronize(device)

    comm.finalize()

    if tmp_plan_dir is not None:
        tmp_plan_dir.cleanup()

    if rank == 0:
        print("[rank 0] All done.", flush=True)


if __name__ == "__main__":
    main()
