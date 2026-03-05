// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef HAS_MSCCLPP

#include <comms/torchcomms/mscclpp/TorchWorkMSCCLPP.hpp>

#include <glog/logging.h>

namespace torch::comms {

// --- MscclppGpuEventPool ---

MscclppGpuEventPool::MscclppGpuEventPool(
    std::shared_ptr<CudaApi> cuda_api,
    size_t max_size)
    : cuda_api_(std::move(cuda_api)), max_size_(max_size) {}

MscclppGpuEventPool::~MscclppGpuEventPool() {
  // Acquire the lock so we don't destroy events while another thread holds one.
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto event : available_) {
    cuda_api_->eventDestroy(event);
  }
  available_.clear();
}

cudaEvent_t MscclppGpuEventPool::acquire() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!available_.empty()) {
    cudaEvent_t event = available_.back();
    available_.pop_back();
    return event;
  }
  // Pool is empty — allocate a new event.
  // cudaEventDisableTiming: no timing hardware, ~2x cheaper than timed events.
  // We only use events for stream synchronization, not elapsed-time queries.
  cudaEvent_t event;
  CUDA_CHECK(
      cuda_api_,
      cuda_api_->eventCreateWithFlags(&event, cudaEventDisableTiming),
      "Failed to create CUDA event for MscclppGpuEventPool");
  return event;
}

void MscclppGpuEventPool::release(cudaEvent_t event) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (available_.size() < max_size_) {
    available_.push_back(event);
  } else {
    // Pool is at capacity; destroy the event rather than grow unboundedly.
    cuda_api_->eventDestroy(event);
  }
}

// --- TorchWorkMSCCLPP ---

TorchWorkMSCCLPP::TorchWorkMSCCLPP(
    cudaStream_t op_stream,
    int device_index,
    std::chrono::milliseconds timeout_ms,
    std::shared_ptr<MscclppGpuEventPool> event_pool,
    std::shared_ptr<CudaApi> cuda_api)
    : op_stream_(op_stream),
      device_index_(device_index),
      timeout_ms_(timeout_ms),
      event_pool_(std::move(event_pool)),
      cuda_api_(std::move(cuda_api)) {
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
  CUDA_CHECK(
      cuda_api_,
      cuda_api_->eventRecord(start_event_, op_stream_),
      "Failed to record MSCCL++ start event");
}

// Called by TorchCommMSCCLPP *after* launching the MSCCL++ executor call.
// Stamps op_stream_ with an end marker. wait() blocks the caller's stream
// on this event, and checkStatus() uses it to detect completion.
void TorchWorkMSCCLPP::recordEnd() {
  CUDA_CHECK(
      cuda_api_,
      cuda_api_->eventRecord(end_event_, op_stream_),
      "Failed to record MSCCL++ end event");
}

// Polls CUDA events to advance the work status.
// Mirrors TorchWorkNCCL::checkStatus() exactly:
//   1. If start_completed_time_ not set, query start_event_.
//      On cudaSuccess: store current time, mark INPROGRESS.
//   2. Query end_event_.
//      On cudaSuccess: mark COMPLETED.
//      On cudaErrorNotReady: check elapsed vs timeout_ms_.
//      On other error: mark ERROR.
TorchWork::WorkStatus TorchWorkMSCCLPP::checkStatus() {
  // Short-circuit if already terminal
  if (status() == WorkStatus::COMPLETED || status() == WorkStatus::ERROR ||
      status() == WorkStatus::TIMEDOUT) {
    return status();
  }

  // Step 1: query start event to establish when the GPU began executing
  if (!start_completed_time_.has_value()) {
    cudaError_t start_status = cuda_api_->eventQuery(start_event_);
    if (start_status == cudaSuccess) {
      start_completed_time_ = std::chrono::steady_clock::now();
      setStatus(WorkStatus::INPROGRESS);
    } else if (start_status != cudaErrorNotReady) {
      LOG(ERROR) << "[TorchWorkMSCCLPP] CUDA error during start event query: "
                 << cuda_api_->getErrorString(start_status);
      setStatus(WorkStatus::ERROR);
    }
  }
  if (status() == WorkStatus::NOT_STARTED || status() == WorkStatus::ERROR) {
    return status();
  }

  // Step 2: start event done — now query end event
  cudaError_t end_status = cuda_api_->eventQuery(end_event_);
  if (end_status == cudaSuccess) {
    setStatus(WorkStatus::COMPLETED);
  } else if (end_status == cudaErrorNotReady) {
    // Still running — check timeout against start_completed_time_
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_completed_time_.value());
    if (elapsed > timeout_ms_) {
      LOG(ERROR) << "[TorchWorkMSCCLPP] Operation timed out after "
                 << elapsed.count() << " ms (limit: " << timeout_ms_.count()
                 << " ms)";
      setStatus(WorkStatus::TIMEDOUT);
    }
  } else {
    LOG(ERROR) << "[TorchWorkMSCCLPP] CUDA error during end event query: "
               << cuda_api_->getErrorString(end_status);
    setStatus(WorkStatus::ERROR);
  }

  return status();
}

// CPU-block until the collective GPU kernel has fully completed on this rank.
//
// MSCCLPP memory-channel kernels poll peer GPU memory via NVLink.  As long
// as any rank's kernel is still running, it must be able to access all peers'
// NVLink-registered memory.  finalize() destroys the communicator and
// unregisters that memory, so it is ONLY safe to call finalize() once every
// rank has confirmed its own GPU kernel is done.
//
// cudaEventSynchronize() blocks the CPU thread until end_event_ fires, i.e.,
// until the GPU has executed past recordEnd() on op_stream_.  After wait()
// returns, this rank's collective kernel is guaranteed complete.
void TorchWorkMSCCLPP::wait() {
  WorkStatus current = checkStatus();
  if (current == WorkStatus::COMPLETED || current == WorkStatus::ERROR ||
      current == WorkStatus::TIMEDOUT) {
    return;
  }

  // CPU-blocking wait: the calling thread sleeps until the GPU reaches
  // end_event_, which is recorded after the executor call returns.
  cudaError_t err = cudaEventSynchronize(end_event_);
  if (err == cudaSuccess) {
    setStatus(WorkStatus::COMPLETED);
  } else {
    LOG(ERROR) << "[TorchWorkMSCCLPP] cudaEventSynchronize failed: "
               << cudaGetErrorString(err);
    setStatus(WorkStatus::ERROR);
  }
}

} // namespace torch::comms

#endif // HAS_MSCCLPP
