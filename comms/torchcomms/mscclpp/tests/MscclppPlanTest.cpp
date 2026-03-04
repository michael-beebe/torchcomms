// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Unit tests for TorchCommMSCCLPP plan selection and loading.
//
// Uses a friend fixture (MscclppPlanTest declared as friend in
// TorchCommMSCCLPP.hpp) to call the private selectPlan() / loadPlans()
// methods directly.  No CUDA device, network, or real MSCCL++ communicator
// is required — the tests exercise edge cases that never reach the API calls.
//
// Run standalone:
//   ctest --test-dir build -R MscclppPlanTest --output-on-failure

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "comms/torchcomms/mscclpp/TorchCommMSCCLPP.hpp"

using namespace torch::comms;

#ifdef HAS_MSCCLPP

// ---------------------------------------------------------------------------
// Test fixture — declared as friend in TorchCommMSCCLPP.hpp.
// Provides thin wrappers around the private methods so test bodies stay
// readable, and manages a list of temp directories for automatic cleanup.
// ---------------------------------------------------------------------------
class MscclppPlanTest : public ::testing::Test {
 protected:
  // Call the private selectPlan() method.
  const mscclpp::ExecutionPlan& selectPlan(
      const TorchCommMSCCLPP& comm,
      const std::string& collective,
      size_t bytes,
      const std::unordered_map<std::string, std::string>& hints = {}) {
    return comm.selectPlan(collective, bytes, hints);
  }

  // Call the private loadPlans() method.
  void loadPlans(TorchCommMSCCLPP& comm, const std::string& dir) {
    comm.loadPlans(dir);
  }

  // Returns the number of plans currently cached in the comm instance.
  size_t planCount(const TorchCommMSCCLPP& comm) {
    return comm.plans_.size();
  }

  // Creates a uniquely-named temp directory and registers it for removal on
  // TearDown.
  std::string makeTempDir(const std::string& label) {
    std::string dir = "/tmp/mscclpp_plan_test_" + label + "_" +
        std::to_string(
            ::testing::UnitTest::GetInstance()->random_seed());
    std::filesystem::create_directories(dir);
    temp_dirs_.push_back(dir);
    return dir;
  }

  void TearDown() override {
    for (const auto& d : temp_dirs_) {
      // Use shell rm -rf instead of std::filesystem::remove_all to avoid a
      // libstdc++ version-specific crash when recursively deleting non-empty
      // directories on this platform.
      std::system(("rm -rf '" + d + "'").c_str());
    }
  }

 private:
  std::vector<std::string> temp_dirs_;
};

// ---------------------------------------------------------------------------
// selectPlan() — throw cases
//
// All tests use an uninitialized (empty plans_) TorchCommMSCCLPP so no API
// calls are made.  We only verify the error-handling paths.
// ---------------------------------------------------------------------------

TEST_F(MscclppPlanTest, SelectPlanNoPlansThrows) {
  // plans_ is empty → every lookup fails → must throw std::runtime_error
  TorchCommMSCCLPP comm;
  EXPECT_THROW(selectPlan(comm, "allreduce", 1024), std::runtime_error);
}

TEST_F(MscclppPlanTest, SelectPlanErrorMentionsCollective) {
  // The error message must identify which collective was requested so the
  // user knows what plan file to provide.
  TorchCommMSCCLPP comm;
  try {
    selectPlan(comm, "allgather", 512 * 1024);
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("allgather"), std::string::npos)
        << "Error message should name the missing collective";
  }
}

TEST_F(MscclppPlanTest, SelectPlanExplicitHintNotFoundThrows) {
  // When the caller provides torchcomm::mscclpp::plan=<name> and that name
  // is not in plans_, we should throw immediately (no fallback).
  TorchCommMSCCLPP comm;
  std::unordered_map<std::string, std::string> hints = {
      {"torchcomm::mscclpp::plan", "custom_plan_v2"}};
  EXPECT_THROW(
      selectPlan(comm, "allreduce", 1024, hints), std::runtime_error);
}

TEST_F(MscclppPlanTest, SelectPlanExplicitHintErrorMentionsPlanName) {
  // The error thrown when an explicit hint plan is missing must include the
  // requested plan name — makes debugging easier.
  TorchCommMSCCLPP comm;
  const std::string kMissing = "my_custom_allreduce";
  std::unordered_map<std::string, std::string> hints = {
      {"torchcomm::mscclpp::plan", kMissing}};
  try {
    selectPlan(comm, "allreduce", 1024, hints);
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find(kMissing), std::string::npos)
        << "Error should name the missing plan '" << kMissing << "'";
  }
}

TEST_F(MscclppPlanTest, SelectPlanSmallMessageFallsThrough) {
  // ≤ 1 MB path: tries <collective>_sm_packet, then <collective>, then throws.
  // With empty plans_ both lookups fail → throw.
  TorchCommMSCCLPP comm;
  EXPECT_THROW(
      selectPlan(comm, "allreduce", 512 * 1024 /* 512 KB */),
      std::runtime_error);
}

TEST_F(MscclppPlanTest, SelectPlanLargeMessageFallsThrough) {
  // > 1 MB path: tries <collective>_sm, then <collective>, then throws.
  TorchCommMSCCLPP comm;
  EXPECT_THROW(
      selectPlan(comm, "allreduce", 2 * 1024 * 1024 /* 2 MB */),
      std::runtime_error);
}

// ---------------------------------------------------------------------------
// loadPlans() — directory edge cases
//
// These cases never reach mscclpp_api_->loadExecutionPlan() because:
//   - non-existent dir: fs::exists() returns false → early return
//   - empty dir: directory_iterator finds nothing → loop body never runs
//   - non-JSON files: extension != ".json" → each entry is skipped
// So mscclpp_api_ being null (uninitialized comm) is safe.
// ---------------------------------------------------------------------------

TEST_F(MscclppPlanTest, LoadPlansNonexistentDirLogsWarningAndDoesNotThrow) {
  TorchCommMSCCLPP comm;
  EXPECT_NO_THROW(
      loadPlans(comm, "/tmp/no_such_mscclpp_plan_dir_xyz_never_exists"));
  EXPECT_EQ(planCount(comm), 0u);
}

TEST_F(MscclppPlanTest, LoadPlansEmptyDirLoadsNothingAndDoesNotThrow) {
  TorchCommMSCCLPP comm;
  EXPECT_NO_THROW(loadPlans(comm, makeTempDir("empty")));
  EXPECT_EQ(planCount(comm), 0u);
}

TEST_F(MscclppPlanTest, LoadPlansIgnoresNonJsonFiles) {
  std::string dir = makeTempDir("nonjson");
  // Write files with non-.json extensions — all should be skipped.
  for (const char* name : {"README.md", "manifest.txt", "notes.log"}) {
    std::ofstream f(dir + "/" + name);
    f << "not a plan";
    f.close();
  }
  TorchCommMSCCLPP comm;
  EXPECT_NO_THROW(loadPlans(comm, dir));
  EXPECT_EQ(planCount(comm), 0u)
      << "Non-.json files must not be loaded as plans";
}

#else // !HAS_MSCCLPP

TEST(MscclppPlanTestStub, SkippedWithoutMscclpp) {
  GTEST_SKIP() << "Built without HAS_MSCCLPP — plan tests skipped";
}

#endif // HAS_MSCCLPP

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
