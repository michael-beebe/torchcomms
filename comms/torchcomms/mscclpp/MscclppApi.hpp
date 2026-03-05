// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#ifdef HAS_MSCCLPP

#include <memory>
#include <mutex>
#include <string>

#include <mscclpp/core.hpp>
#include <mscclpp/executor.hpp>

#include <comms/torchcomms/mscclpp/GpuTypes.hpp>

namespace torch::comms {

/**
 * Abstract interface for MSCCL++ API operations.
 *
 * Follows the NcclApi pattern: all methods are virtual for dependency
 * injection in tests. DefaultMscclppApi wraps the real MSCCL++ C++ API
 * with mutex protection (serializes all calls — sufficient for setup/teardown
 * and executor launches; revisit if lock contention becomes measurable).
 *
 * Only covers the executor-based collective path. Low-level connection and
 * memory registration APIs (needed for SM-channel collectives) are added
 * in a later commit.
 */
class MscclppApi {
 public:
  virtual ~MscclppApi() = default;

  // --- Bootstrap ---

  /// Generate a new unique ID on rank 0.
  /// All ranks must initialize their TcpBootstrap with the same UniqueId.
  virtual mscclpp::UniqueId createUniqueId() = 0;

  /// Create a TcpBootstrap for the given rank/size.
  virtual std::shared_ptr<mscclpp::TcpBootstrap> createTcpBootstrap(
      int rank,
      int size) = 0;

  /// Initialize the bootstrap using a UniqueId from the store exchange.
  virtual void bootstrapInitialize(
      mscclpp::TcpBootstrap& bootstrap,
      mscclpp::UniqueId unique_id,
      int64_t timeout_sec = 30) = 0;

  // --- Communicator ---

  /// Create a Communicator backed by the given bootstrap.
  virtual std::shared_ptr<mscclpp::Communicator> createCommunicator(
      std::shared_ptr<mscclpp::Bootstrap> bootstrap) = 0;

  // --- Executor (host-initiated collective plans) ---

  /// Create an Executor bound to this communicator.
  virtual std::unique_ptr<mscclpp::Executor> createExecutor(
      std::shared_ptr<mscclpp::Communicator> comm) = 0;

  /// Load an execution plan from a JSON file for the given rank.
  virtual std::unique_ptr<mscclpp::ExecutionPlan> loadExecutionPlan(
      const std::string& plan_path,
      int rank) = 0;

  /// Execute a pre-loaded plan on the given stream.
  ///
  /// send/recvSize are both set to `bytes` (in-place collectives where
  /// sendBuffSize == recvBuffSize, e.g., allreduce, reduce, barrier).
  /// dataType must be mapped from the tensor dtype via torchDtypeToMscclpp().
  /// Uses cudaStream_t directly — matches mscclpp::Executor::execute().
  virtual void executePlan(
      mscclpp::Executor& executor,
      const mscclpp::ExecutionPlan& plan,
      int rank,
      void* sendbuf,
      void* recvbuf,
      size_t bytes,
      mscclpp::DataType dataType,
      cudaStream_t stream) = 0;

  /// Execute a pre-loaded plan with separate send and receive buffer sizes.
  ///
  /// Use this overload for collectives where sendBuffSize != recvBuffSize
  /// (e.g., allgather: sendBytes = per-rank chunk, recvBytes = full output).
  virtual void executePlan(
      mscclpp::Executor& executor,
      const mscclpp::ExecutionPlan& plan,
      int rank,
      void* sendbuf,
      void* recvbuf,
      size_t sendBytes,
      size_t recvBytes,
      mscclpp::DataType dataType,
      cudaStream_t stream) = 0;
};

/**
 * Default implementation wrapping real MSCCL++ calls.
 * All methods acquire api_mutex_ before delegating to mscclpp::.
 */
class DefaultMscclppApi : public MscclppApi {
 public:
  DefaultMscclppApi() = default;

  mscclpp::UniqueId createUniqueId() override;

  std::shared_ptr<mscclpp::TcpBootstrap> createTcpBootstrap(int rank, int size)
      override;

  void bootstrapInitialize(
      mscclpp::TcpBootstrap& bootstrap,
      mscclpp::UniqueId unique_id,
      int64_t timeout_sec = 30) override;

  std::shared_ptr<mscclpp::Communicator> createCommunicator(
      std::shared_ptr<mscclpp::Bootstrap> bootstrap) override;

  std::unique_ptr<mscclpp::Executor> createExecutor(
      std::shared_ptr<mscclpp::Communicator> comm) override;

  std::unique_ptr<mscclpp::ExecutionPlan> loadExecutionPlan(
      const std::string& plan_path,
      int rank) override;

  void executePlan(
      mscclpp::Executor& executor,
      const mscclpp::ExecutionPlan& plan,
      int rank,
      void* sendbuf,
      void* recvbuf,
      size_t bytes,
      mscclpp::DataType dataType,
      cudaStream_t stream) override;

  void executePlan(
      mscclpp::Executor& executor,
      const mscclpp::ExecutionPlan& plan,
      int rank,
      void* sendbuf,
      void* recvbuf,
      size_t sendBytes,
      size_t recvBytes,
      mscclpp::DataType dataType,
      cudaStream_t stream) override;

 private:
  std::mutex api_mutex_;
};

} // namespace torch::comms

#endif // HAS_MSCCLPP
