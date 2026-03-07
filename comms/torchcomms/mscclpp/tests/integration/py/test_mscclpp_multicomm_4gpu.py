#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Multi-comm coexistence test: MSCCL++ + NCCL communicators in the same process.
#
# Validates that an MSCCL++ communicator and an NCCL (or RCCL on AMD)
# communicator can operate simultaneously without interference. This is
# critical for deployments where MSCCL++ handles allreduce/allgather while
# NCCL handles operations MSCCL++ doesn't support (send/recv, all_to_all, etc.).
#
# Uses allreduce_4gpu.json bundled alongside this test.
#
# Run with:
#   torchrun --nproc_per_node=4 \
#     comms/torchcomms/mscclpp/tests/integration/py/test_mscclpp_multicomm_4gpu.py
#
# Override the plan directory via TORCHCOMM_MSCCLPP_PLAN_DIR:
#   TORCHCOMM_MSCCLPP_PLAN_DIR=/path/to/plans torchrun --nproc_per_node=4 \
#     comms/torchcomms/mscclpp/tests/integration/py/test_mscclpp_multicomm_4gpu.py

import os
import shutil
import sys
import tempfile

import torch
import torchcomms

# ---------------------------------------------------------------------------
# Plan setup — reuse the allreduce plan bundled with the allreduce test.
# ---------------------------------------------------------------------------
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_BUNDLED_PLAN = os.path.join(_TEST_DIR, "allreduce_4gpu.json")


def setup_plan_dir() -> tempfile.TemporaryDirectory:
    """
    Copy the bundled 4-GPU allreduce plan as 'allreduce.json' in a temp dir
    so selectPlan() picks it up by the bare collective name.
    """
    if not os.path.exists(_BUNDLED_PLAN):
        raise FileNotFoundError(
            f"Bundled 4-GPU allreduce plan not found: {_BUNDLED_PLAN}"
        )
    tmp = tempfile.TemporaryDirectory(prefix="mscclpp_multicomm_plans_")
    shutil.copy(_BUNDLED_PLAN, os.path.join(tmp.name, "allreduce.json"))
    return tmp


def _cc_backend() -> str:
    """Return the platform-appropriate collective comm backend."""
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        return "rccl"
    return "nccl"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_concurrent_allreduce(
    mscclpp_comm: torchcomms.TorchComm,
    nccl_comm: torchcomms.TorchComm,
    device: torch.device,
    rank: int,
) -> bool:
    """MSCCL++ all_reduce and NCCL all_reduce on different tensors concurrently."""
    t_mscclpp = torch.ones(512, device=device) * 3.0
    t_nccl = torch.ones(512, device=device) * 5.0

    # Launch both async
    w1 = mscclpp_comm.all_reduce(t_mscclpp, torchcomms.ReduceOp.SUM, True)
    w2 = nccl_comm.all_reduce(t_nccl, torchcomms.ReduceOp.SUM, True)

    w1.wait()
    w2.wait()

    expected_mscclpp = 3.0 * mscclpp_comm.get_size()
    expected_nccl = 5.0 * nccl_comm.get_size()

    ok_mscclpp = torch.allclose(
        t_mscclpp, torch.full_like(t_mscclpp, expected_mscclpp)
    )
    ok_nccl = torch.allclose(t_nccl, torch.full_like(t_nccl, expected_nccl))

    ok = ok_mscclpp and ok_nccl
    if rank == 0:
        print(
            f"  concurrent_allreduce: mscclpp={t_mscclpp[0].item():.1f} "
            f"(expect {expected_mscclpp:.1f}) "
            f"nccl={t_nccl[0].item():.1f} (expect {expected_nccl:.1f})  "
            f"{'PASS' if ok else 'FAIL'}",
            flush=True,
        )
    return ok


def test_nccl_broadcast_after_mscclpp_allreduce(
    mscclpp_comm: torchcomms.TorchComm,
    nccl_comm: torchcomms.TorchComm,
    device: torch.device,
    rank: int,
) -> bool:
    """MSCCL++ handles allreduce, then NCCL handles broadcast (MSCCL++ can't)."""
    # Step 1: all_reduce via MSCCL++
    tensor = torch.ones(256, device=device) * float(rank + 1)
    mscclpp_comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, False).wait()

    world_size = mscclpp_comm.get_size()
    expected_sum = float(world_size * (world_size + 1) // 2)
    ok_allreduce = abs(tensor[0].item() - expected_sum) < 1e-3

    # Step 2: broadcast via NCCL (MSCCL++ doesn't support broadcast)
    bcast_tensor = torch.full((128,), 42.0, device=device) if rank == 0 else torch.zeros(128, device=device)
    nccl_comm.broadcast(bcast_tensor, 0, False).wait()

    ok_broadcast = torch.allclose(bcast_tensor, torch.full_like(bcast_tensor, 42.0))

    ok = ok_allreduce and ok_broadcast
    if rank == 0:
        print(
            f"  nccl_broadcast_after_mscclpp_allreduce: "
            f"allreduce={tensor[0].item():.1f} (expect {expected_sum:.1f})  "
            f"broadcast={bcast_tensor[0].item():.1f} (expect 42.0)  "
            f"{'PASS' if ok else 'FAIL'}",
            flush=True,
        )
    return ok


def test_sequential_alternating(
    mscclpp_comm: torchcomms.TorchComm,
    nccl_comm: torchcomms.TorchComm,
    device: torch.device,
    rank: int,
) -> bool:
    """Alternate all_reduce between backends to check state isolation."""
    world_size = mscclpp_comm.get_size()
    ok = True
    for i in range(3):
        # MSCCL++ all_reduce
        t1 = torch.ones(128, device=device) * float(i + 1)
        mscclpp_comm.all_reduce(t1, torchcomms.ReduceOp.SUM, False).wait()
        expected1 = float((i + 1) * world_size)
        if abs(t1[0].item() - expected1) > 1e-3:
            ok = False

        # NCCL all_reduce
        t2 = torch.ones(128, device=device) * float(i + 10)
        nccl_comm.all_reduce(t2, torchcomms.ReduceOp.SUM, False).wait()
        expected2 = float((i + 10) * world_size)
        if abs(t2[0].item() - expected2) > 1e-3:
            ok = False

    if rank == 0:
        print(
            f"  sequential_alternating (3 iterations):  "
            f"{'PASS' if ok else 'FAIL'}",
            flush=True,
        )
    return ok


# ---------------------------------------------------------------------------
# Main
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

    # Set up plan directory for MSCCL++
    tmp_plan_dir: tempfile.TemporaryDirectory | None = None
    if "TORCHCOMM_MSCCLPP_PLAN_DIR" not in os.environ:
        tmp_plan_dir = setup_plan_dir()
        os.environ["TORCHCOMM_MSCCLPP_PLAN_DIR"] = tmp_plan_dir.name

    # ------------------------------------------------------------------
    # Create both communicators
    # ------------------------------------------------------------------
    cc_backend = _cc_backend()

    if rank == 0:
        print(
            f"[rank {rank}] Creating MSCCL++ + {cc_backend.upper()} communicators  "
            f"world_size={world_size}  "
            f"plan_dir={os.environ.get('TORCHCOMM_MSCCLPP_PLAN_DIR', '<not set>')}",
            flush=True,
        )

    mscclpp_comm = torchcomms.new_comm("mscclpp", device, name="mscclpp_comm")
    nccl_comm = torchcomms.new_comm(cc_backend, device, name="nccl_comm")

    if rank == 0:
        print(
            f"[rank {rank}] Both communicators created: "
            f"mscclpp (rank={mscclpp_comm.get_rank()}, size={mscclpp_comm.get_size()})  "
            f"{cc_backend} (rank={nccl_comm.get_rank()}, size={nccl_comm.get_size()})",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Run tests
    # ------------------------------------------------------------------
    all_ok = True

    all_ok &= test_concurrent_allreduce(mscclpp_comm, nccl_comm, device, rank)
    all_ok &= test_nccl_broadcast_after_mscclpp_allreduce(
        mscclpp_comm, nccl_comm, device, rank
    )
    all_ok &= test_sequential_alternating(mscclpp_comm, nccl_comm, device, rank)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    mscclpp_comm.finalize()
    nccl_comm.finalize()

    if tmp_plan_dir is not None:
        tmp_plan_dir.cleanup()

    if rank == 0:
        print(f"\n[rank 0] Multi-comm coexistence: {'ALL PASS' if all_ok else 'FAIL'}", flush=True)

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
