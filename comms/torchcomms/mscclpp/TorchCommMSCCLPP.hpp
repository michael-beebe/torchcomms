// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <ATen/ATen.h>
#include <comms/torchcomms/TorchCommBackend.hpp>

#ifdef HAS_MSCCLPP
#include <comms/torchcomms/mscclpp/GpuTypes.hpp>
#include <comms/torchcomms/mscclpp/MscclppApi.hpp>
#include <comms/torchcomms/mscclpp/TorchWorkMSCCLPP.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/executor.hpp>
// Forward-declare the test fixture so the friend declaration below resolves.
class MscclppPlanTest;
#endif

namespace torch::comms {

class TorchCommMSCCLPP : public TorchCommBackend,
                         public std::enable_shared_from_this<TorchCommMSCCLPP> {
 public:
  static constexpr std::string_view kBackendName = "mscclpp";

  TorchCommMSCCLPP();
  ~TorchCommMSCCLPP() override;

  // Delete copy and move
  TorchCommMSCCLPP(const TorchCommMSCCLPP&) = delete;
  TorchCommMSCCLPP(TorchCommMSCCLPP&&) = delete;
  TorchCommMSCCLPP& operator=(const TorchCommMSCCLPP&) = delete;
  TorchCommMSCCLPP& operator=(TorchCommMSCCLPP&&) = delete;

  // Lifecycle
  void init(
      at::Device device,
      const std::string& name,
      const CommOptions& options = {}) override;
  void finalize() override;

  // Metadata
  int getRank() const override;
  int getSize() const override;
  std::string_view getBackendName() const override;
  std::string_view getCommName() const override;
  const CommOptions& getOptions() const override;
  const at::Device& getDevice() const override;

  // Point-to-point
  c10::intrusive_ptr<TorchWork> send(
      const at::Tensor& tensor,
      int dst,
      bool async_op,
      const SendOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> recv(
      at::Tensor& tensor,
      int src,
      bool async_op,
      const RecvOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> batch_op_issue(
      const std::vector<BatchSendRecv::P2POp>& ops,
      bool async_op,
      const BatchP2POptions& options = {}) override;

  // Collectives
  c10::intrusive_ptr<TorchWork> broadcast(
      at::Tensor& tensor,
      int root,
      bool async_op,
      const BroadcastOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_reduce(
      at::Tensor& tensor,
      const ReduceOp& op,
      bool async_op,
      const AllReduceOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> reduce(
      const at::Tensor& tensor,
      int root,
      const ReduceOp& op,
      bool async_op,
      const ReduceOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_gather(
      const std::vector<at::Tensor>& tensor_list,
      const at::Tensor& tensor,
      bool async_op,
      const AllGatherOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_gather_v(
      const std::vector<at::Tensor>& tensor_list,
      const at::Tensor& tensor,
      bool async_op,
      const AllGatherOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_gather_single(
      at::Tensor& output,
      const at::Tensor& input,
      bool async_op,
      const AllGatherSingleOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> reduce_scatter(
      at::Tensor& output,
      const std::vector<at::Tensor>& input_list,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> reduce_scatter_v(
      at::Tensor& output,
      const std::vector<at::Tensor>& input_list,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> reduce_scatter_single(
      at::Tensor& output,
      const at::Tensor& input,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterSingleOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_to_all_single(
      at::Tensor& output,
      const at::Tensor& input,
      bool async_op,
      const AllToAllSingleOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_to_all_v_single(
      at::Tensor& output,
      const at::Tensor& input,
      const std::vector<uint64_t>& output_split_sizes,
      const std::vector<uint64_t>& input_split_sizes,
      bool async_op,
      const AllToAllvSingleOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_to_all(
      const std::vector<at::Tensor>& output_tensor_list,
      const std::vector<at::Tensor>& input_tensor_list,
      bool async_op,
      const AllToAllOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> barrier(
      bool async_op,
      const BarrierOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> scatter(
      at::Tensor& output_tensor,
      const std::vector<at::Tensor>& input_tensor_list,
      int root,
      bool async_op,
      const ScatterOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> gather(
      const std::vector<at::Tensor>& output_tensor_list,
      const at::Tensor& input_tensor,
      int root,
      bool async_op,
      const GatherOptions& options = {}) override;

  // Communicator management
  std::shared_ptr<TorchCommBackend> split(
      const std::vector<int>& ranks,
      const std::string& name,
      const CommOptions& options = {}) override;

 private:
  void checkInitialized() const;

  bool initialized_ = false;
  at::Device device_{at::kCUDA};
  std::string name_;
  CommOptions options_;
  int rank_ = 0;
  int size_ = 1;

#ifdef HAS_MSCCLPP
  // GPU API (device/stream management) — injected for testing, defaults to
  // DefaultGpuApi in init().
  std::shared_ptr<mscclpp_gpu::GpuApi> gpu_api_;

  // MSCCL++ API (communicator/executor) — injected for testing.
  std::shared_ptr<MscclppApi> mscclpp_api_;

  std::shared_ptr<mscclpp::Communicator> comm_;
  std::unique_ptr<mscclpp::Executor> executor_;

  // Dedicated async stream for executor launches.
  mscclpp_gpu::gpuStream_t internal_stream_ = nullptr;

  // GPU event pool shared across all work handles from this communicator.
  std::shared_ptr<MscclppGpuEventPool> event_pool_;

  // Algorithm plan cache: plan_name → loaded ExecutionPlan.
  std::unordered_map<std::string, std::unique_ptr<mscclpp::ExecutionPlan>>
      plans_;

  // Load all *.json plans from plan_dir into plans_.
  void loadPlans(const std::string& plan_dir);

  // Select the best plan for a collective + message size.
  // Checks the "torchcomm::mscclpp::plan" hint first, then applies the
  // naming convention: <collective>_sm_packet (≤1MB) / <collective>_sm (>1MB).
  const mscclpp::ExecutionPlan& selectPlan(
      const std::string& collective,
      size_t message_bytes,
      const std::unordered_map<std::string, std::string>& hints) const;

  // Common implementation for all executor-based collectives.
  // Selects a plan, creates a work handle, launches the executor, and
  // returns the work. Each public collective method validates inputs
  // and calls this with the appropriate buffers/sizes.
  c10::intrusive_ptr<TorchWork> executeCollective(
      const std::string& collective,
      void* sendbuf,
      void* recvbuf,
      size_t sendBytes,
      size_t recvBytes,
      at::ScalarType dtype,
      bool async_op,
      std::chrono::milliseconds timeout,
      const std::unordered_map<std::string, std::string>& hints);

  // Grant test fixture access to private plan methods and state.
  friend class ::MscclppPlanTest;
#endif // HAS_MSCCLPP
};

} // namespace torch::comms
