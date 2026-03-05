#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# End-to-end all_gather_single test using the MSCCL++ backend.
#
# Uses allgather_4gpu.json bundled alongside this test (generated with
# MSCCL++'s CollectiveProgram / AllGather plan generator for 4 GPUs,
# memory channels only).
#
# Each rank i contributes (i+1) * ones as its input.  After all_gather_single
# the output tensor must satisfy:
#   output[i * n_elems : (i+1) * n_elems] == float(i + 1)  for i in 0..world_size-1
#
# Run with:
#   torchrun --nproc_per_node=4 \
#     comms/torchcomms/mscclpp/tests/integration/py/test_mscclpp_allgather_4gpu.py
#
# Override the plan directory via TORCHCOMM_MSCCLPP_PLAN_DIR:
#   TORCHCOMM_MSCCLPP_PLAN_DIR=/path/to/plans torchrun --nproc_per_node=4 \
#     comms/torchcomms/mscclpp/tests/integration/py/test_mscclpp_allgather_4gpu.py

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
_BUNDLED_PLAN = os.path.join(_TEST_DIR, "allgather_4gpu.json")


def setup_plan_dir() -> tempfile.TemporaryDirectory:
    """
    Copy the bundled 4-GPU allgather plan as 'allgather.json' in a temp dir
    so selectPlan() picks it up by the bare collective name.
    Returns the TemporaryDirectory object; caller must keep it alive.
    """
    if not os.path.exists(_BUNDLED_PLAN):
        raise FileNotFoundError(
            f"Bundled 4-GPU allgather plan not found: {_BUNDLED_PLAN}"
        )
    tmp = tempfile.TemporaryDirectory(prefix="mscclpp_plans_")
    shutil.copy(_BUNDLED_PLAN, os.path.join(tmp.name, "allgather.json"))
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

    comm = torchcomms.new_comm("mscclpp", device, name="allgather_test")

    if rank == 0:
        print(
            f"[rank {rank}] Communicator created: "
            f"backend={comm.get_backend()} world_size={world_size}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # All-gather: rank i contributes (i+1) * ones.
    # Expected: output[i * n_elems : (i+1) * n_elems] == float(i + 1)
    # ------------------------------------------------------------------
    n_elems = 1024  # keep small for CI; allgather output is world_size × input
    fill_value = float(rank + 1)
    input_tensor = torch.full(
        (n_elems,), fill_value, dtype=torch.float32, device=device
    )
    output_tensor = torch.zeros(
        world_size * n_elems, dtype=torch.float32, device=device
    )

    work = comm.all_gather_single(output_tensor, input_tensor, False)
    work.wait()

    # Validate each rank's chunk independently.
    ok = True
    for i in range(world_size):
        expected = float(i + 1)
        chunk = output_tensor[i * n_elems : (i + 1) * n_elems]
        actual = chunk[0].item()
        chunk_ok = abs(actual - expected) < 1e-3 and torch.allclose(
            chunk, torch.full_like(chunk, expected)
        )
        if not chunk_ok:
            ok = False
            print(
                f"[rank {rank}] slot {i}: got {actual:.1f}  expected {expected:.1f}  FAIL",
                file=sys.stderr,
                flush=True,
            )

    status = "PASS" if ok else "FAIL"
    print(
        f"[rank {rank}] all_gather_single: {status}",
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
