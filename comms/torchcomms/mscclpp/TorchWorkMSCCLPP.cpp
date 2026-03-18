// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef HAS_MSCCLPP

#include <comms/torchcomms/mscclpp/TorchWorkMSCCLPP.hpp>

#include <glog/logging.h>

namespace torch::comms {

using namespace mscclpp_gpu;

// --- MscclppGpuEventPool ---

MscclppGpuEventPool::MscclppGpuEventPool(
    std::shared_ptr<GpuApi> gpu_api,
    size_t max_size)
    : gpu_api_(std::move(gpu_api)), max_size_(max_size) {}

MscclppGpuEventPool::~MscclppGpuEventPool() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto event : available_) {
    gpu_api_->eventDestroy(event);
  }
  available_.clear();
}

gpuEvent_t MscclppGpuEventPool::acquire() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!available_.empty()) {
    gpuEvent_t event = available_.back();
    available_.pop_back();
    return event;
  }
  gpuEvent_t event;
  GPU_CHECK(
      gpu_api_,
      gpu_api_->eventCreateWithFlags(&event, gpuEventDisableTiming),
      "Failed to create GPU event for MscclppGpuEventPool");
  return event;
}

void MscclppGpuEventPool::release(gpuEvent_t event) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (available_.size() < max_size_) {
    available_.push_back(event);
  } else {
    gpu_api_->eventDestroy(event);
  }
}

// --- TorchWorkMSCCLPP ---

TorchWorkMSCCLPP::TorchWorkMSCCLPP(
    gpuStream_t op_stream,
    int device_index,
    std::chrono::milliseconds timeout_ms,
    std::shared_ptr<MscclppGpuEventPool> event_pool,
    std::shared_ptr<GpuApi> gpu_api)
    : op_stream_(op_stream),
      device_index_(device_index),
      timeout_ms_(timeout_ms),
      event_pool_(std::move(event_pool)),
      gpu_api_(std::move(gpu_api)) {
  // Acquire two events from the pool: one to mark when the collective
  // starts executing on the GPU, one to mark when it finishes.
  // Pool events are reused across operations to avoid allocation overhead.
  start_event_ = event_pool_->acquire();
  end_event_ = event_pool_->acquire();
}

TorchWorkMSCCLPP::~TorchWorkMSCCLPP() {
  // Return events to the pool instead of destroying them.
  // Matches TorchCommNCCL's returnEvent() pattern.
  event_pool_->release(start_event_);
  event_pool_->release(end_event_);
}

// Called by TorchCommMSCCLPP *before* launching the MSCCL++ executor call.
// Stamps op_stream_ with a start marker so checkStatus() can detect when
// the GPU has begun executing (and start the timeout clock from that point).
void TorchWorkMSCCLPP::recordStart() {
  GPU_CHECK(
      gpu_api_,
      gpu_api_->eventRecord(start_event_, op_stream_),
      "Failed to record MSCCL++ start event");
}

// Called by TorchCommMSCCLPP *after* launching the MSCCL++ executor call.
// Stamps op_stream_ with an end marker. wait() blocks the caller's stream
// on this event, and checkStatus() uses it to detect completion.
void TorchWorkMSCCLPP::recordEnd() {
  GPU_CHECK(
      gpu_api_,
      gpu_api_->eventRecord(end_event_, op_stream_),
      "Failed to record MSCCL++ end event");
}

// Polls GPU events to advance the work status.
TorchWork::WorkStatus TorchWorkMSCCLPP::checkStatus() {
  // Short-circuit if already terminal
  if (status() == WorkStatus::COMPLETED || status() == WorkStatus::ERROR ||
      status() == WorkStatus::TIMEDOUT) {
    return status();
  }

  // Step 1: query start event to establish when the GPU began executing
  if (!start_completed_time_.has_value()) {
    gpuError_t start_status = gpu_api_->eventQuery(start_event_);
    if (start_status == gpuSuccess) {
      start_completed_time_ = std::chrono::steady_clock::now();
      setStatus(WorkStatus::INPROGRESS);
    } else if (start_status != gpuErrorNotReady) {
      LOG(ERROR) << "[TC] GPU error during start event query: "
                 << gpu_api_->getErrorString(start_status) << " ("
                 << start_status << ")";
      setStatus(WorkStatus::ERROR);
    }
  }
  if (status() == WorkStatus::NOT_STARTED || status() == WorkStatus::ERROR) {
    return status();
  }

  // Step 2: start event done — now query end event
  gpuError_t end_status = gpu_api_->eventQuery(end_event_);
  if (end_status == gpuSuccess) {
    setStatus(WorkStatus::COMPLETED);
  } else if (end_status == gpuErrorNotReady) {
    // Still running — check timeout against start_completed_time_
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_completed_time_.value());
    if (elapsed > timeout_ms_) {
      LOG(ERROR) << "[TC] Operation timed out after " << elapsed.count()
                 << " ms (limit: " << timeout_ms_.count() << " ms)";
      setStatus(WorkStatus::TIMEDOUT);
    }
  } else {
    LOG(ERROR) << "[TC] GPU error during end event query: "
               << gpu_api_->getErrorString(end_status) << " ("
               << end_status << ")";
    setStatus(WorkStatus::ERROR);
  }

  return status();
}

// Wait until the collective completes by inserting a stream dependency.
//
// MSCCLPP memory-channel kernels poll peer GPU memory via NVLink.  As long
// as any rank's kernel is still running, it must be able to access all peers'
// NVLink-registered memory.  finalize() handles CPU-blocking synchronization
// and the bootstrap barrier to ensure safe teardown.
void TorchWorkMSCCLPP::wait() {
  WorkStatus current = checkStatus();
  if (current == WorkStatus::COMPLETED || current == WorkStatus::ERROR ||
      current == WorkStatus::TIMEDOUT) {
    return;
  }

  // GPU-side wait: make the caller's current stream wait on end_event_.
  // This matches TorchWorkNCCL::wait() — no CPU blocking, just stream ordering.
  gpuStream_t current_stream = gpu_api_->getCurrentCUDAStream(device_index_);
  GPU_CHECK(
      gpu_api_,
      gpu_api_->streamWaitEvent(current_stream, end_event_, 0),
      "Failed to make stream wait for MSCCL++ end event");
  setStatus(WorkStatus::COMPLETED);
}

} // namespace torch::comms

#endif // HAS_MSCCLPP
