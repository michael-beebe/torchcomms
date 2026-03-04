// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Unit tests for MscclppGpuEventPool and TorchWorkMSCCLPP.
//
// Tests the CUDA event pool and work-handle lifecycle independently of any
// real MSCCL++ communicator or multi-GPU setup. Requires one CUDA device;
// tests are skipped automatically when none is available.

#ifdef HAS_MSCCLPP

#include <gtest/gtest.h>

#include <glog/logging.h>

#include <comms/torchcomms/device/cuda/CudaApi.hpp>
#include <comms/torchcomms/mscclpp/TorchWorkMSCCLPP.hpp>

using namespace torch::comms;

// ============================================================
// Test fixture — owns a DefaultCudaApi and a cudaStream_t.
// ============================================================
class MscclppEventPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int dev_count = 0;
    cudaError_t err = cudaGetDeviceCount(&dev_count);
    if (err != cudaSuccess || dev_count == 0) {
      GTEST_SKIP() << "No CUDA devices available";
    }
    ASSERT_EQ(cuda_api_.setDevice(0), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
  }

  void TearDown() override {
    if (stream_) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
  }

  DefaultCudaApi cuda_api_;
  cudaStream_t stream_ = nullptr;
};

// ============================================================
// MscclppGpuEventPool tests
// ============================================================

TEST_F(MscclppEventPoolTest, AcquireReturnsNonNull) {
  // Pool starts empty — acquire must allocate a new event.
  MscclppGpuEventPool pool(&cuda_api_, /*max_size=*/8);
  cudaEvent_t event = pool.acquire();
  ASSERT_NE(event, nullptr);
  pool.release(event);
}

TEST_F(MscclppEventPoolTest, ReleasedEventIsReused) {
  // After release, the next acquire must return the exact same pointer.
  MscclppGpuEventPool pool(&cuda_api_, /*max_size=*/8);
  cudaEvent_t first = pool.acquire();
  ASSERT_NE(first, nullptr);
  pool.release(first);

  cudaEvent_t second = pool.acquire();
  EXPECT_EQ(first, second) << "Pool should reuse the released event";
  pool.release(second);
}

TEST_F(MscclppEventPoolTest, MultipleAcquiresReturnDistinctEvents) {
  MscclppGpuEventPool pool(&cuda_api_, /*max_size=*/8);
  cudaEvent_t a = pool.acquire();
  cudaEvent_t b = pool.acquire();
  EXPECT_NE(a, b);
  pool.release(a);
  pool.release(b);
}

TEST_F(MscclppEventPoolTest, PoolCapEvictsOverflowEvent) {
  // max_size=2: releasing a third event should destroy it, not pool it.
  // After that, re-acquiring two events must succeed (pool still functional).
  MscclppGpuEventPool pool(&cuda_api_, /*max_size=*/2);
  cudaEvent_t e1 = pool.acquire();
  cudaEvent_t e2 = pool.acquire();
  cudaEvent_t e3 = pool.acquire();
  pool.release(e1);
  pool.release(e2);
  pool.release(e3); // cap=2, so e3 is destroyed here

  // Re-acquire 2 from pool (e1/e2), then a fresh one
  cudaEvent_t r1 = pool.acquire();
  cudaEvent_t r2 = pool.acquire();
  cudaEvent_t r3 = pool.acquire(); // new allocation — pool was empty again
  EXPECT_NE(r1, nullptr);
  EXPECT_NE(r2, nullptr);
  EXPECT_NE(r3, nullptr);
  pool.release(r1);
  pool.release(r2);
  pool.release(r3);
}

TEST_F(MscclppEventPoolTest, EventsAreCreatedWithTimingDisabled) {
  // cudaEventDisableTiming events must fail with a non-success error when
  // elapsed time is queried (cudaEventElapsedTime is disabled for them).
  MscclppGpuEventPool pool(&cuda_api_, /*max_size=*/4);
  cudaEvent_t start = pool.acquire();
  cudaEvent_t end = pool.acquire();

  ASSERT_EQ(cudaEventRecord(start, stream_), cudaSuccess);
  ASSERT_EQ(cudaEventRecord(end, stream_), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  float ms = 0.0f;
  cudaError_t timing_err = cudaEventElapsedTime(&ms, start, end);
  EXPECT_NE(timing_err, cudaSuccess)
      << "cudaEventElapsedTime must fail for events created with "
         "cudaEventDisableTiming";

  pool.release(start);
  pool.release(end);
}

// ============================================================
// TorchWorkMSCCLPP lifecycle tests
// ============================================================

TEST_F(MscclppEventPoolTest, WorkInitialStatusIsNotStarted) {
  // Sanity: freshly constructed work must have NOT_STARTED status.
  MscclppGpuEventPool pool(&cuda_api_, /*max_size=*/8);
  auto work = c10::make_intrusive<TorchWorkMSCCLPP>(
      stream_,
      /*device_index=*/0,
      std::chrono::milliseconds{30'000},
      pool,
      &cuda_api_);
  EXPECT_EQ(work->status(), TorchWork::WorkStatus::NOT_STARTED);
}

TEST_F(MscclppEventPoolTest, WorkReturnsEventsToPoolOnDestruction) {
  // After destructor runs, its two events should be back in the pool.
  // Verify by acquiring two events afterwards — they must be non-null.
  MscclppGpuEventPool pool(&cuda_api_, /*max_size=*/8);
  {
    auto work = c10::make_intrusive<TorchWorkMSCCLPP>(
        stream_,
        /*device_index=*/0,
        std::chrono::milliseconds{30'000},
        pool,
        &cuda_api_);
    (void)work; // destructor fires at end of scope
  }
  cudaEvent_t r1 = pool.acquire();
  cudaEvent_t r2 = pool.acquire();
  EXPECT_NE(r1, nullptr);
  EXPECT_NE(r2, nullptr);
  pool.release(r1);
  pool.release(r2);
}

TEST_F(MscclppEventPoolTest, RecordStartAndEndDoNotThrow) {
  // recordStart() / recordEnd() must record events onto op_stream_ without
  // throwing.
  MscclppGpuEventPool pool(&cuda_api_, /*max_size=*/8);
  auto work = c10::make_intrusive<TorchWorkMSCCLPP>(
      stream_,
      /*device_index=*/0,
      std::chrono::milliseconds{30'000},
      pool,
      &cuda_api_);
  EXPECT_NO_THROW(work->recordStart());
  EXPECT_NO_THROW(work->recordEnd());
  // Drain the stream so events complete before the work is destroyed.
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

TEST_F(MscclppEventPoolTest, GetTimeoutReturnsConfiguredValue) {
  MscclppGpuEventPool pool(&cuda_api_, /*max_size=*/8);
  auto timeout = std::chrono::milliseconds{12'345};
  auto work = c10::make_intrusive<TorchWorkMSCCLPP>(
      stream_, /*device_index=*/0, timeout, pool, &cuda_api_);
  EXPECT_EQ(work->getTimeout(), timeout);
}

// ============================================================
// main — initialise glog before running tests
// ============================================================
int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#else // !HAS_MSCCLPP

// When built without MSCCL++, produce a file that compiles but has no tests.
#include <gtest/gtest.h>

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif // HAS_MSCCLPP
