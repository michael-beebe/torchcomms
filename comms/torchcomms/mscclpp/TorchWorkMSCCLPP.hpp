// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#ifdef HAS_MSCCLPP

#include <chrono>
#include <mutex>
#include <optional>
#include <vector>

#include <comms/torchcomms/TorchWork.hpp>
#include <comms/torchcomms/mscclpp/GpuTypes.hpp>

namespace torch::comms {

// Forward declaration
class TorchCommMSCCLPP;

/**
 * GPU event pool — reuses CUDA events to avoid alloc/free overhead.
 *
 * Thread-safe. Intended to be owned by TorchCommMSCCLPP (Commit 8) and
 * borrowed by TorchWorkMSCCLPP via reference. Follows the same pattern as
 * TorchCommNCCL's getEvent()/returnEvent() methods, but extracted into a
 * class so it can be declared here and instantiated before TorchCommMSCCLPP
 * has its full initialization.
 *
 * Events are created with gpuEventDisableTiming (no timing overhead) since
 * we only use them for stream synchronization.
 */
class MscclppGpuEventPool {
 public:
  explicit MscclppGpuEventPool(
      std::shared_ptr<mscclpp_detail::GpuApi> gpu_api,
      size_t max_size = 256);
  ~MscclppGpuEventPool();

  // Non-copyable, non-movable
  MscclppGpuEventPool(const MscclppGpuEventPool&) = delete;
  MscclppGpuEventPool& operator=(const MscclppGpuEventPool&) = delete;
  MscclppGpuEventPool(MscclppGpuEventPool&&) = delete;
  MscclppGpuEventPool& operator=(MscclppGpuEventPool&&) = delete;

  /// Acquire an event from the pool (or allocate a new one if empty).
  mscclpp_detail::gpuEvent_t acquire();

  /// Return an event to the pool. If the pool is full, destroys the event.
  void release(mscclpp_detail::gpuEvent_t event);

 private:
  std::shared_ptr<mscclpp_detail::GpuApi> gpu_api_;
  std::vector<mscclpp_detail::gpuEvent_t> available_;
  std::mutex mutex_;
  size_t max_size_;
};

/**
 * GPU event-based async work handle for MSCCL++ operations.
 *
 * Follows TorchWorkNCCL pattern exactly:
 *   - recordStart() / recordEnd() bracket the MSCCL++ executor call
 *   - wait() issues streamWaitEvent on the caller's current stream (GPU-side,
 *     no CPU blocking)
 *   - checkStatus() polls events and enforces timeout
 *
 * Simplified vs TorchWorkNCCL: no RecordFunction / profiling integration
 * and no input tensor lifetime tracking (the MSCCL++ executor operates on
 * raw data_ptr(); the caller is responsible for tensor lifetime during the
 * collective). Both can be added later if needed.
 */
class TorchWorkMSCCLPP : public TorchWork {
 public:
  TorchWorkMSCCLPP(
      mscclpp_detail::gpuStream_t op_stream,
      int device_index,
      std::chrono::milliseconds timeout_ms,
      std::shared_ptr<MscclppGpuEventPool> event_pool,
      std::shared_ptr<mscclpp_detail::GpuApi> gpu_api);
  ~TorchWorkMSCCLPP() override;

  // Non-copyable, non-movable
  TorchWorkMSCCLPP(const TorchWorkMSCCLPP&) = delete;
  TorchWorkMSCCLPP(TorchWorkMSCCLPP&&) = delete;
  TorchWorkMSCCLPP& operator=(const TorchWorkMSCCLPP&) = delete;
  TorchWorkMSCCLPP& operator=(TorchWorkMSCCLPP&&) = delete;

  // TorchWork interface — only wait() is pure virtual on the base
  void wait() override;
  std::chrono::milliseconds getTimeout() const override {
    return timeout_ms_;
  }

  // Called by TorchCommMSCCLPP around the executor call.
  // recordStart() must be called before launching the collective;
  // recordEnd() immediately after.
  void recordStart();
  void recordEnd();

 private:
  // Poll GPU events and advance status. Returns current WorkStatus.
  WorkStatus checkStatus();

  mscclpp_detail::gpuEvent_t start_event_;
  mscclpp_detail::gpuEvent_t end_event_;
  mscclpp_detail::gpuStream_t op_stream_; // not owned
  int device_index_;
  std::chrono::milliseconds timeout_ms_;
  std::shared_ptr<MscclppGpuEventPool> event_pool_;
  std::shared_ptr<mscclpp_detail::GpuApi> gpu_api_;
  std::optional<std::chrono::steady_clock::time_point> start_completed_time_;
};

} // namespace torch::comms

#endif // HAS_MSCCLPP
