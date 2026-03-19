// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comms/torchcomms/TorchCommFactory.hpp>
#include <comms/torchcomms/TorchWork.hpp>
#include <comms/torchcomms/mscclpp/TorchCommMSCCLPP.hpp>

// Include TorchCommLogging.hpp (which pulls in glog) AFTER all ATen/torch
// headers so glog's LOG macro takes precedence over the c10 stub
// (c10/util/logging_is_not_google_glog.h).
#include <comms/torchcomms/utils/Logging.hpp>

#ifdef HAS_MSCCLPP
#include <ATen/cuda/CUDAContext.h>
#include <comms/torchcomms/mscclpp/DtypeMap.hpp>
#include <comms/torchcomms/mscclpp/MscclppUtils.hpp>
#include <comms/torchcomms/mscclpp/TorchCommMSCCLPPBootstrap.hpp>
#include <filesystem>

namespace torch::comms::mscclpp_utils {
// Select the GPU stream for an operation.
// Async ops use the dedicated internal stream so they return immediately;
// synchronous ops use the caller's current torch CUDA stream so the
// executor launch is inline with any preceding work on that stream.
inline mscclpp_gpu::gpuStream_t getOperationStream(
    bool async_op,
    mscclpp_gpu::gpuStream_t internal_stream,
    int device_index) {
  if (async_op) {
    return internal_stream;
  }
  return at::cuda::getCurrentCUDAStream(device_index).stream();
}
} // namespace torch::comms::mscclpp_utils

#endif // HAS_MSCCLPP

namespace torch::comms {

TorchCommMSCCLPP::TorchCommMSCCLPP() = default;

TorchCommMSCCLPP::~TorchCommMSCCLPP() {
  if (initialized_) {
    TC_LOG(WARNING, this)
        << "TorchCommMSCCLPP was not finalized before destruction. "
        << "This may indicate a resource leak. Please call finalize() "
        << "explicitly.";
  }
}

void TorchCommMSCCLPP::checkInitialized() const {
  if (!initialized_) {
    throw std::runtime_error(
        "[TorchCommMSCCLPP] Communicator not initialized. Call init() first.");
  }
}

void TorchCommMSCCLPP::init(
    at::Device device,
    const std::string& name,
    const CommOptions& options) {
  if (initialized_) {
    throw std::runtime_error(
        "[TorchCommMSCCLPP] Already initialized. Call finalize() first.");
  }

  device_ = device;
  name_ = name;
  options_ = options;

#ifdef HAS_MSCCLPP
  // 1. GPU API (injectable; default to real CUDA/HIP calls)
  if (!gpu_api_) {
    gpu_api_ = std::make_shared<mscclpp_gpu::DefaultGpuApi>();
  }

  // 2. MSCCL++ API (injectable for tests)
  if (!mscclpp_api_) {
    mscclpp_api_ = std::make_shared<DefaultMscclppApi>();
  }

  // 3. Bootstrap: discovers rank/size and creates the Communicator
  auto bootstrap = std::make_unique<TorchCommMSCCLPPBootstrap>(
      options.store, device, mscclpp_api_, options.timeout);
  rank_ = bootstrap->getRank();
  size_ = bootstrap->getSize();
  comm_ = bootstrap->createCommunicator(name, options);

  // 4. Select GPU device
  gpu_api_->setDevice(device_.index());

  // 5. Create a dedicated internal stream for executor launches
  if (options.high_priority_stream) {
    int least_priority = 0, greatest_priority = 0;
    gpu_api_->getStreamPriorityRange(&least_priority, &greatest_priority);
    gpu_api_->streamCreateWithPriority(
        &internal_stream_,
        mscclpp_gpu::gpuStreamNonBlocking,
        greatest_priority);
  } else {
    gpu_api_->streamCreateWithPriority(
        &internal_stream_, mscclpp_gpu::gpuStreamNonBlocking, 0);
  }

  // 6. Create Executor
  executor_ = mscclpp_api_->createExecutor(comm_);

  // 7. Create GPU event pool (backed by the same gpu_api_)
  event_pool_ =
      std::make_shared<MscclppGpuEventPool>(gpu_api_, /*max_size=*/256);

  // 8. Load execution plans
  //    Priority: "torchcomm::mscclpp::plan_dir" hint >
  //    TORCHCOMM_MSCCLPP_PLAN_DIR env
  std::string plan_dir;
  auto hint_it = options.hints.find("torchcomm::mscclpp::plan_dir");
  if (hint_it != options.hints.end()) {
    plan_dir = hint_it->second;
  } else {
    const char* env_plan_dir = std::getenv("TORCHCOMM_MSCCLPP_PLAN_DIR");
    if (env_plan_dir) {
      plan_dir = env_plan_dir;
    }
  }
  if (!plan_dir.empty()) {
    loadPlans(plan_dir);
  }
#endif // HAS_MSCCLPP

  initialized_ = true;

  TC_LOG(INFO, this) << "Initialized: device=" << device_
#ifdef HAS_MSCCLPP
                     << " plans_loaded=" << plans_.size()
#endif
      ;
}

void TorchCommMSCCLPP::finalize() {
  if (!initialized_) {
    return;
  }

#ifdef HAS_MSCCLPP
  // Drain our own streams while the communicator (and NVLink memory) is alive.
  //
  // After work.wait() (which CPU-blocks), this rank's collective kernel is
  // done.  However, ring-algorithm collectives may finish on different ranks
  // at slightly different times — rank 0 can complete its last ring step while
  // rank 3's kernel is still handling its last chunk.
  //
  // Teardown sequence:
  //   1. Sync our own streams (fast — work is already done per wait()).
  //   2. bootstrap()->barrier(): CPU rendezvous that ensures ALL ranks have
  //      drained their GPU work before ANY rank destroys its communicator.
  //      Once all ranks return from the barrier, no NVLink polling kernel is
  //      running anywhere, so comm_.reset() is safe.
  //   3. CPU-side teardown: executor, plans, event pool, stream, comm.
  if (internal_stream_) {
    gpu_api_->streamSynchronize(internal_stream_);
  }
  gpu_api_->streamSynchronize(
      at::cuda::getCurrentCUDAStream(device_.index()).stream());

  // All ranks rendezvous here before any comm is destroyed.
  comm_->bootstrap()->barrier();

  executor_.reset();

  // Clear plan cache and event pool.
  plans_.clear();
  event_pool_.reset();

  // Destroy internal stream.
  if (internal_stream_) {
    gpu_api_->streamDestroy(internal_stream_);
    internal_stream_ = nullptr;
  }

  // Release communicator last (unregisters NVLink memory).
  // Safe: all ranks passed the bootstrap barrier above.
  comm_.reset();
#endif // HAS_MSCCLPP

  initialized_ = false;

  TC_LOG(INFO, this) << "Finalized.";
}

#ifdef HAS_MSCCLPP

void TorchCommMSCCLPP::loadPlans(const std::string& plan_dir) {
  namespace fs = std::filesystem;
  if (!fs::exists(plan_dir)) {
    TC_LOG(WARNING, this) << "Plan directory not found: " << plan_dir;
    return;
  }
  for (const auto& entry : fs::directory_iterator(plan_dir)) {
    if (entry.path().extension() == ".json") {
      const std::string plan_name = entry.path().stem().string();
      plans_[plan_name] =
          mscclpp_api_->loadExecutionPlan(entry.path().string(), rank_);
      TC_LOG(INFO, this) << "Loaded plan: " << plan_name;
    }
  }
}

const mscclpp::ExecutionPlan& TorchCommMSCCLPP::selectPlan(
    const std::string& collective,
    size_t message_bytes,
    const std::unordered_map<std::string, std::string>& hints) const {
  // Explicit plan override via hint
  auto hint_it = hints.find("torchcomm::mscclpp::plan");
  if (hint_it != hints.end()) {
    auto plan_it = plans_.find(hint_it->second);
    if (plan_it != plans_.end()) {
      return *plan_it->second;
    }
    throw std::runtime_error(
        "[TorchCommMSCCLPP] Requested plan not found: " + hint_it->second);
  }

  // Auto-select by naming convention:
  //   ≤1MB  → <collective>_sm_packet  (low-latency SM kernel)
  //   >1MB  → <collective>_sm          (high-throughput SM kernel)
  const std::string key = (message_bytes <= (1u << 20))
      ? (collective + "_sm_packet")
      : (collective + "_sm");

  auto plan_it = plans_.find(key);
  if (plan_it != plans_.end()) {
    return *plan_it->second;
  }

  // Final fallback: bare collective name
  plan_it = plans_.find(collective);
  if (plan_it != plans_.end()) {
    return *plan_it->second;
  }

  throw std::runtime_error(
      "[TorchCommMSCCLPP] No plan found for collective '" + collective +
      "' with message size " + std::to_string(message_bytes) +
      ". Provide plans via TORCHCOMM_MSCCLPP_PLAN_DIR or torchcomm::mscclpp::plan_dir hint.");
}

#endif // HAS_MSCCLPP

int TorchCommMSCCLPP::getRank() const {
  return rank_;
}

int TorchCommMSCCLPP::getSize() const {
  return size_;
}

std::string_view TorchCommMSCCLPP::getBackendName() const {
  return kBackendName;
}

std::string_view TorchCommMSCCLPP::getCommName() const {
  return name_;
}

const CommOptions& TorchCommMSCCLPP::getOptions() const {
  return options_;
}

const at::Device& TorchCommMSCCLPP::getDevice() const {
  return device_;
}

#ifdef HAS_MSCCLPP

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::executeCollective(
    const std::string& collective,
    void* sendbuf,
    void* recvbuf,
    size_t sendBytes,
    size_t recvBytes,
    at::ScalarType dtype,
    bool async_op,
    std::chrono::milliseconds timeout,
    const std::unordered_map<std::string, std::string>& hints) {
  const auto& plan = selectPlan(collective, sendBytes, hints);

  auto stream = mscclpp_utils::getOperationStream(
      async_op, internal_stream_, device_.index());

  auto work = c10::make_intrusive<TorchWorkMSCCLPP>(
      stream, device_.index(), timeout, event_pool_, gpu_api_);
  work->recordStart();

  mscclpp_api_->executePlan(
      *executor_,
      plan,
      rank_,
      sendbuf,
      recvbuf,
      sendBytes,
      recvBytes,
      torchDtypeToMscclpp(dtype),
      stream);

  work->recordEnd();
  return work;
}

#endif // HAS_MSCCLPP

// --- Collective implementations ---
//   Each collective will: validateReduceOp (if applicable), select a plan,
//   create TorchWorkMSCCLPP, execute via mscclpp::Executor, return work handle.

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::send(
    const at::Tensor& /*tensor*/,
    int /*dst*/,
    bool /*async_op*/,
    const SendOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] send() is not supported. "
      "MSCCL++ does not provide point-to-point send (ncclSend is unavailable "
      "in the MSCCL++ NCCL compat layer; ProxyChannel setup is required for "
      "custom P2P). "
      "Use a separate NCCL (NVIDIA) or RCCL (AMD) communicator for "
      "point-to-point.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::recv(
    at::Tensor& /*tensor*/,
    int /*src*/,
    bool /*async_op*/,
    const RecvOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] recv() is not supported. "
      "MSCCL++ does not provide point-to-point recv (ncclRecv is unavailable "
      "in the MSCCL++ NCCL compat layer; ProxyChannel setup is required for "
      "custom P2P). "
      "Use a separate NCCL (NVIDIA) or RCCL (AMD) communicator for "
      "point-to-point.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::batch_op_issue(
    const std::vector<BatchSendRecv::P2POp>& /*ops*/,
    bool /*async_op*/,
    const BatchP2POptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] batch_op_issue() is not supported (requires "
      "send/recv which MSCCL++ does not provide). "
      "Use a separate NCCL (NVIDIA) or RCCL (AMD) communicator for "
      "batched point-to-point.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::broadcast(
    at::Tensor& /*tensor*/,
    int /*root*/,
    bool /*async_op*/,
    const BroadcastOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] broadcast() is not supported. "
      "MSCCL++ does not provide a broadcast executor plan or built-in "
      "algorithm (the NCCL compat layer logs 'No FallBack implementation "
      "for broadcast'). "
      "Use a separate NCCL (NVIDIA) or RCCL (AMD) communicator for "
      "broadcast.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_reduce(
    at::Tensor& tensor,
    const ReduceOp& op,
    bool async_op,
    const AllReduceOptions& options) {
  checkInitialized();

#ifdef HAS_MSCCLPP
  mscclpp_utils::validateReduceOp(op, "all_reduce");
  tensor = mscclpp_utils::ensureContiguous(tensor);

  return executeCollective(
      "allreduce",
      tensor.data_ptr(),
      tensor.data_ptr(),
      tensor.nbytes(),
      tensor.nbytes(),
      tensor.scalar_type(),
      async_op,
      options.timeout,
      options.hints);
#else
  throw std::runtime_error(
      "[TorchCommMSCCLPP] all_reduce() requires MSCCL++ (built without HAS_MSCCLPP).");
#endif
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::reduce(
    const at::Tensor& /*tensor*/,
    int /*root*/,
    const ReduceOp& /*op*/,
    bool /*async_op*/,
    const ReduceOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] reduce() is not supported. "
      "MSCCL++ does not provide a native reduce collective or execution plan. "
      "Use a separate NCCL (NVIDIA) or RCCL (AMD) communicator for reduce.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_gather(
    const std::vector<at::Tensor>& /*tensor_list*/,
    const at::Tensor& /*tensor*/,
    bool /*async_op*/,
    const AllGatherOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] all_gather() (tensor-list variant) is not yet "
      "implemented. Use all_gather_single() instead, which is supported. "
      "Alternatively, use a separate NCCL (NVIDIA) or RCCL (AMD) "
      "communicator.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_gather_v(
    const std::vector<at::Tensor>& /*tensor_list*/,
    const at::Tensor& /*tensor*/,
    bool /*async_op*/,
    const AllGatherOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] all_gather_v() is not supported. "
      "MSCCL++ does not provide variable-length allgather. "
      "Use a separate NCCL (NVIDIA) or RCCL (AMD) communicator.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_gather_single(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    const AllGatherSingleOptions& options) {
  checkInitialized();

#ifdef HAS_MSCCLPP
  auto input_contig = mscclpp_utils::ensureContiguous(input);
  output = mscclpp_utils::ensureContiguous(output);

  const size_t chunk_bytes = static_cast<size_t>(input_contig.nbytes());

  // Pre-stage this rank's input at output[rank * N] before the plan runs.
  auto stream = mscclpp_utils::getOperationStream(
      async_op, internal_stream_, device_.index());
  gpu_api_->memcpyAsync(
      static_cast<char*>(output.data_ptr()) +
          static_cast<size_t>(rank_) * chunk_bytes,
      input_contig.data_ptr(),
      chunk_bytes,
      mscclpp_gpu::gpuMemcpyDeviceToDevice,
      stream);

  return executeCollective(
      "allgather",
      output.data_ptr(),
      output.data_ptr(),
      chunk_bytes,
      static_cast<size_t>(output.nbytes()),
      input_contig.scalar_type(),
      async_op,
      options.timeout,
      options.hints);
#else
  throw std::runtime_error(
      "[TorchCommMSCCLPP] all_gather_single() requires MSCCL++ (built without HAS_MSCCLPP).");
#endif
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::reduce_scatter(
    at::Tensor& /*output*/,
    const std::vector<at::Tensor>& /*input_list*/,
    const ReduceOp& /*op*/,
    bool /*async_op*/,
    const ReduceScatterOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] reduce_scatter() (tensor-list variant) is not yet "
      "implemented. Use reduce_scatter_single() instead, which is supported. "
      "Alternatively, use a separate NCCL (NVIDIA) or RCCL (AMD) "
      "communicator.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::reduce_scatter_v(
    at::Tensor& /*output*/,
    const std::vector<at::Tensor>& /*input_list*/,
    const ReduceOp& /*op*/,
    bool /*async_op*/,
    const ReduceScatterOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] reduce_scatter_v() is not supported. "
      "MSCCL++ does not provide variable-length reduce-scatter. "
      "Use a separate NCCL (NVIDIA) or RCCL (AMD) communicator.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::reduce_scatter_single(
    at::Tensor& output,
    const at::Tensor& input,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterSingleOptions& options) {
  checkInitialized();

#ifdef HAS_MSCCLPP
  mscclpp_utils::validateReduceOp(op, "reduce_scatter_single");
  auto input_contig = mscclpp_utils::ensureContiguous(input);
  output = mscclpp_utils::ensureContiguous(output);

  return executeCollective(
      "reducescatter",
      input_contig.data_ptr(),
      output.data_ptr(),
      static_cast<size_t>(input_contig.nbytes()),
      static_cast<size_t>(output.nbytes()),
      input_contig.scalar_type(),
      async_op,
      options.timeout,
      options.hints);
#else
  throw std::runtime_error(
      "[TorchCommMSCCLPP] reduce_scatter_single() requires MSCCL++ "
      "(built without HAS_MSCCLPP).");
#endif
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_to_all_single(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    const AllToAllSingleOptions& options) {
  checkInitialized();

#ifdef HAS_MSCCLPP
  auto input_contig = mscclpp_utils::ensureContiguous(input);
  output = mscclpp_utils::ensureContiguous(output);

  return executeCollective(
      "alltoall",
      input_contig.data_ptr(),
      output.data_ptr(),
      static_cast<size_t>(input_contig.nbytes()),
      static_cast<size_t>(output.nbytes()),
      input_contig.scalar_type(),
      async_op,
      options.timeout,
      options.hints);
#else
  throw std::runtime_error(
      "[TorchCommMSCCLPP] all_to_all_single() requires MSCCL++ "
      "(built without HAS_MSCCLPP).");
#endif
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_to_all_v_single(
    at::Tensor& /*output*/,
    const at::Tensor& /*input*/,
    const std::vector<uint64_t>& /*output_split_sizes*/,
    const std::vector<uint64_t>& /*input_split_sizes*/,
    bool /*async_op*/,
    const AllToAllvSingleOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] all_to_all_v_single() is not supported. "
      "MSCCL++ does not provide variable-length all-to-all. "
      "Use a separate NCCL (NVIDIA) or RCCL (AMD) communicator.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_to_all(
    const std::vector<at::Tensor>& /*output_tensor_list*/,
    const std::vector<at::Tensor>& /*input_tensor_list*/,
    bool /*async_op*/,
    const AllToAllOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] all_to_all() (tensor-list variant) is not yet "
      "implemented. Use all_to_all_single() instead, which is supported. "
      "Alternatively, use a separate NCCL (NVIDIA) or RCCL (AMD) "
      "communicator.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::barrier(
    bool /*async_op*/,
    const BarrierOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] barrier() is not supported. "
      "MSCCL++ does not provide a native barrier collective. "
      "Use a separate NCCL (NVIDIA) or RCCL (AMD) communicator for barrier.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::scatter(
    at::Tensor& /*output_tensor*/,
    const std::vector<at::Tensor>& /*input_tensor_list*/,
    int /*root*/,
    bool /*async_op*/,
    const ScatterOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] scatter() is not supported. "
      "MSCCL++ does not provide scatter algorithms or executor plans. "
      "Use a separate NCCL (NVIDIA) or RCCL (AMD) communicator.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::gather(
    const std::vector<at::Tensor>& /*output_tensor_list*/,
    const at::Tensor& /*input_tensor*/,
    int /*root*/,
    bool /*async_op*/,
    const GatherOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] gather() is not supported. "
      "MSCCL++ does not provide gather algorithms or executor plans. "
      "Use a separate NCCL (NVIDIA) or RCCL (AMD) communicator.");
}

std::shared_ptr<TorchCommBackend> TorchCommMSCCLPP::split(
    const std::vector<int>& /*ranks*/,
    const std::string& /*name*/,
    const CommOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] split() is not supported. "
      "MSCCL++ does not provide a sub-communicator API. "
      "This blocks TP+PP topologies that rely on split() for sub-groups. "
      "Use a separate NCCL (NVIDIA) or RCCL (AMD) communicator that "
      "supports split().");
}

// --- Factory registration ---

namespace {
class MSCCLPPRegistration {
 public:
  MSCCLPPRegistration() {
    TorchCommFactory::get().register_backend(
        "mscclpp", []() { return std::make_shared<TorchCommMSCCLPP>(); });
  }
};
static const MSCCLPPRegistration registration{};
} // namespace

} // namespace torch::comms
