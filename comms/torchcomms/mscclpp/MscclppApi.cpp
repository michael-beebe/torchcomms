// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef HAS_MSCCLPP

#include <comms/torchcomms/mscclpp/MscclppApi.hpp>

namespace torch::comms {

std::shared_ptr<mscclpp::TcpBootstrap> DefaultMscclppApi::createTcpBootstrap(
    int rank,
    int size) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return std::make_shared<mscclpp::TcpBootstrap>(rank, size);
}

void DefaultMscclppApi::bootstrapInitialize(
    mscclpp::TcpBootstrap& bootstrap,
    const std::string& ip_port_pair,
    int64_t timeout_sec) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  bootstrap.initialize(ip_port_pair, timeout_sec);
}

std::shared_ptr<mscclpp::Communicator> DefaultMscclppApi::createCommunicator(
    std::shared_ptr<mscclpp::Bootstrap> bootstrap) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return std::make_shared<mscclpp::Communicator>(std::move(bootstrap));
}

std::unique_ptr<mscclpp::Executor> DefaultMscclppApi::createExecutor(
    std::shared_ptr<mscclpp::Communicator> comm) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return std::make_unique<mscclpp::Executor>(std::move(comm));
}

std::unique_ptr<mscclpp::ExecutionPlan> DefaultMscclppApi::loadExecutionPlan(
    const std::string& plan_path,
    int rank) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return std::make_unique<mscclpp::ExecutionPlan>(plan_path, rank);
}

void DefaultMscclppApi::executePlan(
    mscclpp::Executor& executor,
    const mscclpp::ExecutionPlan& plan,
    int rank,
    void* sendbuf,
    void* recvbuf,
    size_t bytes,
    mscclpp::DataType dataType,
    cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  executor.execute(rank, sendbuf, recvbuf, bytes, bytes, dataType, plan, stream);
}

} // namespace torch::comms

#endif // HAS_MSCCLPP
