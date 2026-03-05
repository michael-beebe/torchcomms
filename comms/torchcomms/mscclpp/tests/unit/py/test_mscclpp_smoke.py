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
os.environ.setdefault("LOCAL_RANK", "0")


class TestMscclppModuleImport(unittest.TestCase):
    """The extension module must be importable without a CUDA device or network."""

    def test_import_extension_module(self) -> None:
        # This will raise ImportError if the module failed to build or install.
        import torchcomms._comms_mscclpp  # noqa: F401

    def test_entry_point_registered(self) -> None:
        eps = list(
            importlib.metadata.entry_points(group="torchcomms.backends", name="mscclpp")
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
        return torchcomms.new_comm("mscclpp", torch.device("cuda:0"), name=name)

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
        comm_a = torchcomms.new_comm("mscclpp", torch.device("cuda:0"), name="comm_a")
        comm_b = torchcomms.new_comm("mscclpp", torch.device("cuda:0"), name="comm_b")
        try:
            self.assertEqual(comm_a.get_name(), "comm_a")
            self.assertEqual(comm_b.get_name(), "comm_b")
        finally:
            comm_a.finalize()
            comm_b.finalize()


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.device_count() > 0,
    "No CUDA device available",
)
class TestMscclppPlanLoading(unittest.TestCase):
    """Tests that loadPlans() handles edge-case plan directories gracefully.

    These tests exercise the TORCHCOMM_MSCCLPP_PLAN_DIR env-var path of init() without
    real plan JSON files — verifying that bad or empty directories produce a
    warning (not a crash).
    """

    def _make_comm_with_plan_dir(self, plan_dir: str) -> torchcomms.TorchComm:
        """Create a comm with TORCHCOMM_MSCCLPP_PLAN_DIR temporarily set to plan_dir."""
        orig = os.environ.get("TORCHCOMM_MSCCLPP_PLAN_DIR")
        try:
            os.environ["TORCHCOMM_MSCCLPP_PLAN_DIR"] = plan_dir
            return torchcomms.new_comm(
                "mscclpp", torch.device("cuda:0"), name="plan_dir_test"
            )
        finally:
            if orig is None:
                os.environ.pop("TORCHCOMM_MSCCLPP_PLAN_DIR", None)
            else:
                os.environ["TORCHCOMM_MSCCLPP_PLAN_DIR"] = orig

    def test_nonexistent_plan_dir_does_not_raise(self) -> None:
        """A non-existent TORCHCOMM_MSCCLPP_PLAN_DIR should log a warning, not crash."""
        comm = self._make_comm_with_plan_dir("/tmp/no_such_mscclpp_plan_dir_xyz_12345")
        comm.finalize()

    def test_empty_plan_dir_does_not_raise(self) -> None:
        """An empty TORCHCOMM_MSCCLPP_PLAN_DIR directory should not crash init()."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            comm = self._make_comm_with_plan_dir(tmpdir)
            comm.finalize()

    def test_plan_dir_with_non_json_files_does_not_raise(self) -> None:
        """Plan dir containing only non-.json files: init() succeeds, 0 plans."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write dummy non-plan files — loadPlans() must skip these.
            for name in ("README.md", "manifest.txt"):
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write("not a plan")
            comm = self._make_comm_with_plan_dir(tmpdir)
            comm.finalize()


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.device_count() > 0,
    "No CUDA device available",
)
class TestMscclppAllReduceValidation(unittest.TestCase):
    """Validates all_reduce() error-handling paths that don't require a plan.

    These tests run on a single GPU without a plan directory.  They verify:
    - Non-SUM ops are rejected immediately (before plan lookup).
    - SUM op with no plans loaded produces a helpful 'no plan found' error.
    Both paths are exercised without any real distributed communication.
    """

    def setUp(self) -> None:
        self.comm = torchcomms.new_comm(
            "mscclpp", torch.device("cuda:0"), name="allreduce_val_test"
        )
        self.tensor = torch.ones(64, device="cuda:0")

    def tearDown(self) -> None:
        self.comm.finalize()

    def test_non_sum_allreduce_raises(self) -> None:
        """PRODUCT is rejected before any plan lookup — error names the op constraint."""
        with self.assertRaises(RuntimeError) as ctx:
            self.comm.all_reduce(self.tensor, torchcomms.ReduceOp.PRODUCT, False)
        self.assertIn("SUM", str(ctx.exception))

    def test_sum_allreduce_no_plans_raises_with_helpful_message(self) -> None:
        """SUM all_reduce with no plans loaded raises and names the collective."""
        with self.assertRaises(RuntimeError) as ctx:
            self.comm.all_reduce(self.tensor, torchcomms.ReduceOp.SUM, False)
        self.assertIn("allreduce", str(ctx.exception).lower())


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.device_count() > 0,
    "No CUDA device available",
)
class TestMscclppAllGatherSingleValidation(unittest.TestCase):
    """Validates all_gather_single() error-handling paths that don't require a plan."""

    def setUp(self) -> None:
        self.comm = torchcomms.new_comm(
            "mscclpp", torch.device("cuda:0"), name="allgather_val_test"
        )
        self.input = torch.ones(64, device="cuda:0")
        self.output = torch.zeros(256, device="cuda:0")

    def tearDown(self) -> None:
        self.comm.finalize()

    def test_allgather_no_plans_raises_with_helpful_message(self) -> None:
        """all_gather_single() with no plans loaded raises and names the collective."""
        with self.assertRaises(RuntimeError) as ctx:
            self.comm.all_gather_single(self.output, self.input, False)
        self.assertIn("allgather", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
