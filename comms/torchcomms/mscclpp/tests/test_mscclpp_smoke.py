#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Smoke tests for the MSCCL++ backend.
#
# Tests three things independently of any real distributed setup:
#   1. The extension module (_comms_mscclpp) can be imported.
#   2. The "mscclpp" entry point is registered in torchcomms.backends.
#   3. TorchCommFactory can create an "mscclpp" comm and its metadata is
#      correct (uses the stub init(), so no CUDA kernel or network needed).
#
# Run with:   pytest comms/torchcomms/tests/unit/py/test_mscclpp_smoke.py -v

import importlib.metadata
import os
import unittest

import torch
import torchcomms


# ---------------------------------------------------------------------------
# Single-process env required by the factory's store setup.
# ---------------------------------------------------------------------------
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")


class TestMscclppModuleImport(unittest.TestCase):
    """The extension module must be importable without a CUDA device or network."""

    def test_import_extension_module(self) -> None:
        # This will raise ImportError if the module failed to build or install.
        import torchcomms._comms_mscclpp  # noqa: F401

    def test_entry_point_registered(self) -> None:
        eps = list(
            importlib.metadata.entry_points(
                group="torchcomms.backends", name="mscclpp"
            )
        )
        self.assertGreater(
            len(eps),
            0,
            "No 'mscclpp' entry point found in group 'torchcomms.backends'.\n"
            "Reinstall torchcomms with: pip install --no-build-isolation .",
        )


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.device_count() > 0,
    "No CUDA device available",
)
class TestMscclppBackendLifecycle(unittest.TestCase):
    """Factory, metadata, and teardown — uses the stub init(), no real communicator."""

    def _make_comm(self, name: str = "smoke_test_comm") -> torchcomms.TorchComm:
        return torchcomms.new_comm("mscclpp", torch.device("cuda:0"), name)

    # ------------------------------------------------------------------

    def test_new_comm_returns_object(self) -> None:
        comm = self._make_comm()
        self.assertIsNotNone(comm)
        comm.finalize()

    def test_backend_name_is_mscclpp(self) -> None:
        comm = self._make_comm()
        try:
            self.assertEqual(comm.get_backend(), "mscclpp")
        finally:
            comm.finalize()

    def test_comm_name_round_trips(self) -> None:
        comm = self._make_comm("my_mscclpp_comm")
        try:
            self.assertEqual(comm.get_name(), "my_mscclpp_comm")
        finally:
            comm.finalize()

    def test_backend_impl_is_correct_type(self) -> None:
        from torchcomms._comms_mscclpp import TorchCommMSCCLPP

        comm = self._make_comm()
        try:
            self.assertIsInstance(comm.get_backend_impl(), TorchCommMSCCLPP)
        finally:
            comm.finalize()

    def test_finalize_twice_is_safe(self) -> None:
        """finalize() sets initialized_=false; calling it again must not raise."""
        comm = self._make_comm()
        comm.finalize()
        try:
            comm.finalize()  # second call is a no-op
        except Exception as exc:
            self.fail(f"Second finalize() raised unexpectedly: {exc}")

    def test_two_comms_are_independent(self) -> None:
        """Two separate comms for the same device must coexist without interference."""
        comm_a = torchcomms.new_comm("mscclpp", torch.device("cuda:0"), "comm_a")
        comm_b = torchcomms.new_comm("mscclpp", torch.device("cuda:0"), "comm_b")
        try:
            self.assertEqual(comm_a.get_name(), "comm_a")
            self.assertEqual(comm_b.get_name(), "comm_b")
        finally:
            comm_a.finalize()
            comm_b.finalize()


if __name__ == "__main__":
    unittest.main()
