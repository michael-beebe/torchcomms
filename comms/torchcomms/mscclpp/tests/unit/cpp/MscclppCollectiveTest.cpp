// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Unit tests for TorchCommMSCCLPP collective-operation validation helpers.
//
// Tests validateReduceOp() and the checkInitialized() guard that protects
// all collectives.  No CUDA device, network, or real MSCCL++ communicator is
// required — all paths under test throw before touching any GPU state.
//
// Run standalone:
//   ctest --test-dir build -R MscclppCollectiveTest --output-on-failure

#include <stdexcept>
#include <string>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "comms/torchcomms/mscclpp/MscclppUtils.hpp"
#include "comms/torchcomms/mscclpp/TorchCommMSCCLPP.hpp"

using namespace torch::comms;

#ifdef HAS_MSCCLPP

// ---------------------------------------------------------------------------
// validateReduceOp() — pure utility, no GPU or communicator needed.
// ---------------------------------------------------------------------------

TEST(ValidateReduceOpTest, SumPassesWithoutThrow) {
  EXPECT_NO_THROW(
      mscclpp_utils::validateReduceOp(
          ReduceOp(ReduceOp::RedOpType::SUM), "allreduce"));
}

TEST(ValidateReduceOpTest, ProductThrows) {
  EXPECT_THROW(
      mscclpp_utils::validateReduceOp(
          ReduceOp(ReduceOp::RedOpType::PRODUCT), "all_reduce"),
      std::runtime_error);
}

TEST(ValidateReduceOpTest, MaxThrows) {
  EXPECT_THROW(
      mscclpp_utils::validateReduceOp(
          ReduceOp(ReduceOp::RedOpType::MAX), "all_reduce"),
      std::runtime_error);
}

TEST(ValidateReduceOpTest, MinThrows) {
  EXPECT_THROW(
      mscclpp_utils::validateReduceOp(
          ReduceOp(ReduceOp::RedOpType::MIN), "all_reduce"),
      std::runtime_error);
}

TEST(ValidateReduceOpTest, AvgThrows) {
  EXPECT_THROW(
      mscclpp_utils::validateReduceOp(
          ReduceOp(ReduceOp::RedOpType::AVG), "all_reduce"),
      std::runtime_error);
}

TEST(ValidateReduceOpTest, ErrorMessageNamesCollective) {
  // The exception message must identify which collective received the invalid
  // op so the user knows where the error originated.
  const std::string kCollective = "my_test_collective";
  try {
    mscclpp_utils::validateReduceOp(
        ReduceOp(ReduceOp::RedOpType::PRODUCT), kCollective);
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find(kCollective), std::string::npos)
        << "Error message must name the collective '" << kCollective << "'";
  }
}

TEST(ValidateReduceOpTest, ErrorMessageMentionsSUM) {
  // The message should guide the user toward the supported op.
  try {
    mscclpp_utils::validateReduceOp(
        ReduceOp(ReduceOp::RedOpType::MAX), "all_reduce");
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("SUM"), std::string::npos)
        << "Error message should mention SUM as the supported op";
  }
}

// ---------------------------------------------------------------------------
// checkInitialized() guard — calling any collective before init() must throw.
//
// TorchCommMSCCLPP::checkInitialized() is protected by the not-initialized
// guard; the tensor/op values are irrelevant since the throw happens before
// they are touched.  We only need to verify the throw occurs.
// ---------------------------------------------------------------------------

TEST(MscclppCheckInitializedTest, AllReduceThrowsBeforeInit) {
  // Calling all_reduce() on a freshly constructed (not initialized) comm
  // should throw std::runtime_error immediately via checkInitialized().
  // A CPU tensor is fine here: the throw happens before any GPU access.
  TorchCommMSCCLPP comm;
  auto tensor = at::ones({64});
  EXPECT_THROW(
      comm.all_reduce(tensor, ReduceOp(ReduceOp::RedOpType::SUM), false, {}),
      std::runtime_error);
}

TEST(MscclppCheckInitializedTest, AllGatherSingleThrowsBeforeInit) {
  // Calling all_gather_single() on an uninitialized comm must throw
  // immediately via checkInitialized(), before any GPU or plan access.
  TorchCommMSCCLPP comm;
  auto input = at::ones({64});
  auto output = at::zeros({256});
  EXPECT_THROW(
      comm.all_gather_single(output, input, false, {}), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Unsupported ops — verify throws with actionable guidance.
//
// Each test verifies:
//   1. The method throws std::runtime_error.
//   2. The message identifies the backend ("[TorchCommMSCCLPP]").
//   3. The message contains guidance (mentions "NCCL" or "RCCL" or
//      "sub-communicator" so the user knows where to go).
//
// These ops throw unconditionally — no checkInitialized() guard needed,
// no GPU or network required.
// ---------------------------------------------------------------------------

namespace {
void assertMessageContainsGuidance(const std::runtime_error& e) {
  std::string msg(e.what());
  EXPECT_NE(msg.find("TorchCommMSCCLPP"), std::string::npos)
      << "Message must identify the backend";
  bool has_guidance = msg.find("NCCL") != std::string::npos ||
      msg.find("RCCL") != std::string::npos ||
      msg.find("sub-communicator") != std::string::npos;
  EXPECT_TRUE(has_guidance)
      << "Message must contain actionable guidance (NCCL/RCCL/sub-communicator). "
         "Got: "
      << msg;
}
} // namespace

TEST(MscclppUnsupportedOpsTest, SendThrowsWithGuidance) {
  TorchCommMSCCLPP comm;
  auto t = at::ones({64});
  try {
    comm.send(t, 0, false, {});
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    assertMessageContainsGuidance(e);
    EXPECT_NE(std::string(e.what()).find("send"), std::string::npos);
  }
}

TEST(MscclppUnsupportedOpsTest, RecvThrowsWithGuidance) {
  TorchCommMSCCLPP comm;
  auto t = at::ones({64});
  try {
    comm.recv(t, 0, false, {});
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    assertMessageContainsGuidance(e);
    EXPECT_NE(std::string(e.what()).find("recv"), std::string::npos);
  }
}

TEST(MscclppUnsupportedOpsTest, BatchOpIssueThrowsWithGuidance) {
  TorchCommMSCCLPP comm;
  try {
    comm.batch_op_issue({}, false, {});
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    assertMessageContainsGuidance(e);
    EXPECT_NE(std::string(e.what()).find("batch_op_issue"), std::string::npos);
  }
}

TEST(MscclppUnsupportedOpsTest, BroadcastThrowsWithGuidance) {
  TorchCommMSCCLPP comm;
  auto t = at::ones({64});
  try {
    comm.broadcast(t, 0, false, {});
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    assertMessageContainsGuidance(e);
    EXPECT_NE(std::string(e.what()).find("broadcast"), std::string::npos);
  }
}

TEST(MscclppUnsupportedOpsTest, ReduceThrowsWithGuidance) {
  TorchCommMSCCLPP comm;
  auto t = at::ones({64});
  try {
    comm.reduce(t, 0, ReduceOp(ReduceOp::RedOpType::SUM), false, {});
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    assertMessageContainsGuidance(e);
    EXPECT_NE(std::string(e.what()).find("reduce"), std::string::npos);
  }
}

TEST(MscclppUnsupportedOpsTest, AllGatherThrowsWithGuidance) {
  TorchCommMSCCLPP comm;
  auto t = at::ones({64});
  std::vector<at::Tensor> list = {at::zeros({64})};
  try {
    comm.all_gather(list, t, false, {});
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    assertMessageContainsGuidance(e);
    EXPECT_NE(std::string(e.what()).find("all_gather"), std::string::npos);
  }
}

TEST(MscclppUnsupportedOpsTest, AllGatherVThrowsWithGuidance) {
  TorchCommMSCCLPP comm;
  auto t = at::ones({64});
  std::vector<at::Tensor> list = {at::zeros({64})};
  try {
    comm.all_gather_v(list, t, false, {});
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    assertMessageContainsGuidance(e);
  }
}

TEST(MscclppUnsupportedOpsTest, ReduceScatterThrowsWithGuidance) {
  TorchCommMSCCLPP comm;
  auto output = at::zeros({64});
  std::vector<at::Tensor> inputs = {at::ones({64})};
  try {
    comm.reduce_scatter(
        output, inputs, ReduceOp(ReduceOp::RedOpType::SUM), false, {});
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    assertMessageContainsGuidance(e);
    EXPECT_NE(std::string(e.what()).find("reduce_scatter"), std::string::npos);
  }
}

TEST(MscclppUnsupportedOpsTest, ReduceScatterVThrowsWithGuidance) {
  TorchCommMSCCLPP comm;
  auto output = at::zeros({64});
  std::vector<at::Tensor> inputs = {at::ones({64})};
  try {
    comm.reduce_scatter_v(
        output, inputs, ReduceOp(ReduceOp::RedOpType::SUM), false, {});
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    assertMessageContainsGuidance(e);
  }
}

TEST(MscclppUnsupportedOpsTest, ReduceScatterSingleThrowsWithGuidance) {
  TorchCommMSCCLPP comm;
  auto output = at::zeros({64});
  auto input = at::ones({256});
  try {
    comm.reduce_scatter_single(
        output, input, ReduceOp(ReduceOp::RedOpType::SUM), false, {});
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    assertMessageContainsGuidance(e);
    EXPECT_NE(
        std::string(e.what()).find("reduce_scatter_single"), std::string::npos);
  }
}

TEST(MscclppUnsupportedOpsTest, AllToAllSingleThrowsWithGuidance) {
  TorchCommMSCCLPP comm;
  auto output = at::zeros({64});
  auto input = at::ones({64});
  try {
    comm.all_to_all_single(output, input, false, {});
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    assertMessageContainsGuidance(e);
    EXPECT_NE(
        std::string(e.what()).find("all_to_all_single"), std::string::npos);
  }
}

TEST(MscclppUnsupportedOpsTest, AllToAllVSingleThrowsWithGuidance) {
  TorchCommMSCCLPP comm;
  auto output = at::zeros({64});
  auto input = at::ones({64});
  try {
    comm.all_to_all_v_single(output, input, {}, {}, false, {});
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    assertMessageContainsGuidance(e);
  }
}

TEST(MscclppUnsupportedOpsTest, AllToAllThrowsWithGuidance) {
  TorchCommMSCCLPP comm;
  std::vector<at::Tensor> out_list = {at::zeros({64})};
  std::vector<at::Tensor> in_list = {at::ones({64})};
  try {
    comm.all_to_all(out_list, in_list, false, {});
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    assertMessageContainsGuidance(e);
    EXPECT_NE(std::string(e.what()).find("all_to_all"), std::string::npos);
  }
}

TEST(MscclppUnsupportedOpsTest, BarrierThrowsWithGuidance) {
  TorchCommMSCCLPP comm;
  try {
    comm.barrier(false, {});
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    assertMessageContainsGuidance(e);
    EXPECT_NE(std::string(e.what()).find("barrier"), std::string::npos);
  }
}

TEST(MscclppUnsupportedOpsTest, ScatterThrowsWithGuidance) {
  TorchCommMSCCLPP comm;
  auto output = at::zeros({64});
  std::vector<at::Tensor> inputs = {at::ones({64})};
  try {
    comm.scatter(output, inputs, 0, false, {});
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    assertMessageContainsGuidance(e);
    EXPECT_NE(std::string(e.what()).find("scatter"), std::string::npos);
  }
}

TEST(MscclppUnsupportedOpsTest, GatherThrowsWithGuidance) {
  TorchCommMSCCLPP comm;
  auto input = at::ones({64});
  std::vector<at::Tensor> outputs = {at::zeros({64})};
  try {
    comm.gather(outputs, input, 0, false, {});
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    assertMessageContainsGuidance(e);
    EXPECT_NE(std::string(e.what()).find("gather"), std::string::npos);
  }
}

TEST(MscclppUnsupportedOpsTest, SplitThrowsWithGuidance) {
  TorchCommMSCCLPP comm;
  try {
    comm.split({0, 1}, "sub", {});
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    std::string msg(e.what());
    EXPECT_NE(msg.find("TorchCommMSCCLPP"), std::string::npos);
    EXPECT_NE(msg.find("split"), std::string::npos);
    // split() mentions sub-communicator instead of NCCL/RCCL directly
    bool has_guidance = msg.find("sub-communicator") != std::string::npos ||
        msg.find("NCCL") != std::string::npos ||
        msg.find("RCCL") != std::string::npos;
    EXPECT_TRUE(has_guidance) << "split() message must mention guidance";
  }
}

#else // !HAS_MSCCLPP

TEST(MscclppCollectiveTestStub, SkippedWithoutMscclpp) {
  GTEST_SKIP() << "Built without HAS_MSCCLPP — collective tests skipped";
}

#endif // HAS_MSCCLPP

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
