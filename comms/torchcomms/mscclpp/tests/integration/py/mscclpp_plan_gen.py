#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Generate MSCCL++ execution plans for any GPU count at test time.
#
# MSCCL++ execution plans encode the exact GPU topology (rank-to-rank
# communication patterns, channel counts, threadblock assignments), so a
# plan generated for 4 GPUs cannot run on 2 or 8.  This module generates
# plans programmatically using the MSCCL++ language DSL so integration tests
# work with whatever GPU count is available.
#
# Usage from tests:
#   from mscclpp_plan_gen import generate_plans
#   plan_dir = generate_plans(world_size=N)
#   os.environ["TORCHCOMM_MSCCLPP_PLAN_DIR"] = plan_dir.name
#   # ... create communicator and run tests ...
#   plan_dir.cleanup()

import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout


def _generate_allreduce_json(num_gpus: int) -> str:
    """Generate an allreduce execution plan JSON for num_gpus GPUs.

    Uses the MSCCL++ language DSL (memory-channel based allreduce with
    8 threadblocks per GPU, Simple protocol, in-place).
    """
    from mscclpp.language.channel import MemoryChannel, SyncType
    from mscclpp.language.collectives import AllReduce
    from mscclpp.language.general import JSON
    from mscclpp.language.program import CollectiveProgram
    from mscclpp.language.rank import Rank

    num_tb = 8
    collective = AllReduce(num_gpus, num_tb, True)
    with CollectiveProgram(
        "allreduce",
        collective,
        num_gpus,
        protocol="Simple",
        instr_fusion=True,
        num_threads_per_block=1024,
        use_double_scratch_buffer=False,
    ):
        channels = {}
        for gpu in range(num_gpus):
            for tb in range(num_tb):
                for peer in range(num_gpus):
                    if peer != gpu:
                        channels[(peer, gpu, tb)] = MemoryChannel(peer, gpu)

        for gpu in range(num_gpus):
            for tb in range(num_tb):
                for peer in range(num_gpus):
                    if gpu != peer:
                        channels[(peer, gpu, tb)].signal(tb, relaxed=True)

        for gpu in range(num_gpus):
            for tb in range(num_tb):
                for peer in range(num_gpus):
                    if gpu != peer:
                        channels[(peer, gpu, tb)].wait(
                            tb, data_sync=SyncType.after, relaxed=True
                        )

        for gpu in range(num_gpus):
            rank = Rank(gpu)
            input_buffer = rank.get_input_buffer()
            for tb in range(num_tb):
                index = gpu * num_tb + tb
                src_chunk = input_buffer[index : index + 1]
                for peer in range(num_gpus):
                    if gpu != peer:
                        peer_rank = Rank(peer)
                        peer_input_buffer = peer_rank.get_input_buffer()
                        channels[(peer, gpu, tb)].reduce(
                            src_chunk,
                            [peer_input_buffer[index : index + 1]],
                            tb,
                        )
                for peer in range(num_gpus):
                    if gpu != peer:
                        peer_rank = Rank(peer)
                        peer_input_buffer = peer_rank.get_input_buffer()
                        channels[(peer, gpu, tb)].put(
                            peer_input_buffer[index : index + 1], src_chunk, tb
                        )

        for gpu in range(num_gpus):
            for tb in range(num_tb):
                for peer in range(num_gpus):
                    if gpu != peer:
                        channels[(peer, gpu, tb)].signal(
                            tb, data_sync=SyncType.before
                        )

        for gpu in range(num_gpus):
            for tb in range(num_tb):
                for peer in range(num_gpus):
                    if gpu != peer:
                        channels[(peer, gpu, tb)].wait(tb)

        buf = io.StringIO()
        with redirect_stdout(buf):
            print(JSON())
        return buf.getvalue()


def _generate_allgather_json(num_gpus: int) -> str:
    """Generate an allgather execution plan JSON for num_gpus GPUs.

    Uses the MSCCL++ language DSL with a full-mesh allgather pattern:
    every rank puts its chunk to every other rank. Separates sync
    phases from data transfer for correctness with N>2 GPUs.
    """
    from mscclpp.language.channel import MemoryChannel, SyncType
    from mscclpp.language.collectives import AllGather
    from mscclpp.language.general import JSON
    from mscclpp.language.program import CollectiveProgram
    from mscclpp.language.rank import Rank

    chunksperloop = 1
    collective = AllGather(num_gpus, chunksperloop, True)
    with CollectiveProgram(
        "allgather",
        collective,
        num_gpus,
        protocol="Simple",
        instr_fusion=True,
        num_threads_per_block=1024,
        use_double_scratch_buffer=False,
    ):
        # Create all channels up front
        channels = {}
        for src in range(num_gpus):
            for dst in range(num_gpus):
                if src != dst:
                    channels[(dst, src)] = MemoryChannel(dst, src)

        # Phase 1: Initial sync — all ranks signal readiness
        for src in range(num_gpus):
            for dst in range(num_gpus):
                if src != dst:
                    channels[(dst, src)].signal(tb=0, relaxed=True)
        for src in range(num_gpus):
            for dst in range(num_gpus):
                if src != dst:
                    channels[(dst, src)].wait(
                        tb=0, data_sync=SyncType.after, relaxed=True
                    )

        # Phase 2: Each rank puts its data to every other rank
        for src in range(num_gpus):
            rank = Rank(src)
            src_buffer = rank.get_output_buffer()
            src_chunk = src_buffer[src : src + 1]
            for dst in range(num_gpus):
                if src != dst:
                    dst_rank = Rank(dst)
                    dst_buffer = dst_rank.get_output_buffer()
                    dst_chunk = dst_buffer[src : src + 1]
                    channels[(dst, src)].put(dst_chunk, src_chunk, tb=0)

        # Phase 3: Final sync — ensure all puts are visible
        for src in range(num_gpus):
            for dst in range(num_gpus):
                if src != dst:
                    channels[(dst, src)].signal(
                        tb=0, data_sync=SyncType.before
                    )
        for src in range(num_gpus):
            for dst in range(num_gpus):
                if src != dst:
                    channels[(dst, src)].wait(tb=0, relaxed=True)

        buf = io.StringIO()
        with redirect_stdout(buf):
            print(JSON())
        return buf.getvalue()


def generate_plans(world_size: int) -> tempfile.TemporaryDirectory:
    """Generate MSCCL++ execution plans for the given world size.

    Creates a temporary directory containing:
      - allreduce.json  (for all_reduce)
      - allgather.json  (for all_gather_single)

    The caller must keep the returned TemporaryDirectory alive for the
    duration of the test and call .cleanup() when done.

    Args:
        world_size: Number of GPUs / ranks.

    Returns:
        A TemporaryDirectory whose .name is the path to use as
        TORCHCOMM_MSCCLPP_PLAN_DIR.
    """
    tmp = tempfile.TemporaryDirectory(prefix=f"mscclpp_plans_{world_size}gpu_")

    allreduce_json = _generate_allreduce_json(world_size)
    with open(os.path.join(tmp.name, "allreduce.json"), "w") as f:
        f.write(allreduce_json)

    allgather_json = _generate_allgather_json(world_size)
    with open(os.path.join(tmp.name, "allgather.json"), "w") as f:
        f.write(allgather_json)

    return tmp


if __name__ == "__main__":
    # Quick self-test: generate plans for the given GPU count and print paths.
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    plan_dir = generate_plans(n)
    print(f"Generated {n}-GPU plans in: {plan_dir.name}")
    for fname in sorted(os.listdir(plan_dir.name)):
        fpath = os.path.join(plan_dir.name, fname)
        with open(fpath) as f:
            d = json.load(f)
        print(f"  {fname}: {len(d['gpus'])} gpus, collective={d['collective']}")
    plan_dir.cleanup()
