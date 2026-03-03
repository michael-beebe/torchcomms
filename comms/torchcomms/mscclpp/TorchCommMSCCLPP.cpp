// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comms/torchcomms/mscclpp/TorchCommMSCCLPP.hpp>
#include <comms/torchcomms/TorchCommFactory.hpp>
#include <comms/torchcomms/TorchWork.hpp>

namespace torch::comms {

TorchCommMSCCLPP::TorchCommMSCCLPP() = default;

TorchCommMSCCLPP::~TorchCommMSCCLPP() {
  if (initialized_) {
    LOG(WARNING) << "[TorchCommMSCCLPP] Destructor called without finalize()";
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
  // TODO: Replace stub with real MSCCL++ initialization:
  //   - Create MscclppApi (or use injected one)
  //   - Bootstrap via Store adapter to discover rank/size
  //   - Create mscclpp::Communicator
  //   - Set GPU device via gpu_api_->setDevice()
  //   - Create internal GPU stream (high-priority if requested)
  //   - Create mscclpp::Executor
  //   - Load execution plans from plan directory
  device_ = device;
  name_ = name;
  options_ = options;
  initialized_ = true;
}

void TorchCommMSCCLPP::finalize() {
  // TODO: Tear down MSCCL++ state:
  //   - Destroy executor
  //   - Destroy internal GPU stream
  //   - Reset communicator
  //   - Clear plan cache
  initialized_ = false;
}

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

// --- Stub implementations: all throw ---
// TODO: Replace each stub with real MSCCL++ executor calls.
//   Each collective will: validateReduceOp (if applicable), select a plan,
//   create TorchWorkMSCCLPP, execute via mscclpp::Executor, return work handle.

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::send(
    const at::Tensor& /*tensor*/,
    int /*dst*/,
    bool /*async_op*/,
    const SendOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] send() not yet implemented.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::recv(
    at::Tensor& /*tensor*/,
    int /*src*/,
    bool /*async_op*/,
    const RecvOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] recv() not yet implemented.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::batch_op_issue(
    const std::vector<BatchSendRecv::P2POp>& /*ops*/,
    bool /*async_op*/,
    const BatchP2POptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] batch_op_issue() not yet implemented.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::broadcast(
    at::Tensor& /*tensor*/,
    int /*root*/,
    bool /*async_op*/,
    const BroadcastOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] broadcast() not yet implemented.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_reduce(
    at::Tensor& /*tensor*/,
    const ReduceOp& /*op*/,
    bool /*async_op*/,
    const AllReduceOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] all_reduce() not yet implemented.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::reduce(
    const at::Tensor& /*tensor*/,
    int /*root*/,
    const ReduceOp& /*op*/,
    bool /*async_op*/,
    const ReduceOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] reduce() not yet implemented.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_gather(
    const std::vector<at::Tensor>& /*tensor_list*/,
    const at::Tensor& /*tensor*/,
    bool /*async_op*/,
    const AllGatherOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] all_gather() not yet implemented.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_gather_v(
    const std::vector<at::Tensor>& /*tensor_list*/,
    const at::Tensor& /*tensor*/,
    bool /*async_op*/,
    const AllGatherOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] all_gather_v() not yet implemented.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_gather_single(
    at::Tensor& /*output*/,
    const at::Tensor& /*input*/,
    bool /*async_op*/,
    const AllGatherSingleOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] all_gather_single() not yet implemented.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::reduce_scatter(
    at::Tensor& /*output*/,
    const std::vector<at::Tensor>& /*input_list*/,
    const ReduceOp& /*op*/,
    bool /*async_op*/,
    const ReduceScatterOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] reduce_scatter() not yet implemented.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::reduce_scatter_v(
    at::Tensor& /*output*/,
    const std::vector<at::Tensor>& /*input_list*/,
    const ReduceOp& /*op*/,
    bool /*async_op*/,
    const ReduceScatterOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] reduce_scatter_v() not yet implemented.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::reduce_scatter_single(
    at::Tensor& /*output*/,
    const at::Tensor& /*input*/,
    const ReduceOp& /*op*/,
    bool /*async_op*/,
    const ReduceScatterSingleOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] reduce_scatter_single() not yet implemented.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_to_all_single(
    at::Tensor& /*output*/,
    const at::Tensor& /*input*/,
    bool /*async_op*/,
    const AllToAllSingleOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] all_to_all_single() not yet implemented.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_to_all_v_single(
    at::Tensor& /*output*/,
    const at::Tensor& /*input*/,
    const std::vector<uint64_t>& /*output_split_sizes*/,
    const std::vector<uint64_t>& /*input_split_sizes*/,
    bool /*async_op*/,
    const AllToAllvSingleOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] all_to_all_v_single() not yet implemented.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_to_all(
    const std::vector<at::Tensor>& /*output_tensor_list*/,
    const std::vector<at::Tensor>& /*input_tensor_list*/,
    bool /*async_op*/,
    const AllToAllOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] all_to_all() not yet implemented.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::barrier(
    bool /*async_op*/,
    const BarrierOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] barrier() not yet implemented.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::scatter(
    at::Tensor& /*output_tensor*/,
    const std::vector<at::Tensor>& /*input_tensor_list*/,
    int /*root*/,
    bool /*async_op*/,
    const ScatterOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] scatter() not yet implemented.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::gather(
    const std::vector<at::Tensor>& /*output_tensor_list*/,
    const at::Tensor& /*input_tensor*/,
    int /*root*/,
    bool /*async_op*/,
    const GatherOptions& /*options*/) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] gather() not yet implemented.");
}

std::shared_ptr<TorchCommBackend> TorchCommMSCCLPP::split(
    const std::vector<int>& /*ranks*/,
    const std::string& /*name*/,
    const CommOptions& /*options*/) {
  // TODO: Create sub-communicator via mscclpp bootstrap split
  throw std::runtime_error(
      "[TorchCommMSCCLPP] split() not yet implemented.");
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
