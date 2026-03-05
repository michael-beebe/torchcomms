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
