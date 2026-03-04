// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#ifdef HAS_MSCCLPP

#include <chrono>
#include <memory>
#include <string>

#include <c10/core/Device.h>
#include <torch/csrc/distributed/c10d/Store.hpp>

#include <comms/torchcomms/TorchCommOptions.hpp>
#include <comms/torchcomms/mscclpp/MscclppApi.hpp>

namespace torch::comms {

/**
 * Handles MSCCL++ bootstrap initialization using c10d::Store.
 *
 * Follows the TorchCommNCCLBootstrap pattern exactly:
 *   - Rank 0 generates a UniqueId via MscclppApi::createUniqueId()
 *   - Rank 0 writes raw bytes to the store
 *   - All other ranks wait on the store key and read the UniqueId
 *   - All ranks call TcpBootstrap::initialize(uniqueId) with the same ID
 *
 * This is the MSCCL++ equivalent of NCCL's ncclGetUniqueId / ncclCommInitRank
 * bootstrap handshake.
 */
class TorchCommMSCCLPPBootstrap {
 public:
  TorchCommMSCCLPPBootstrap(
      c10::intrusive_ptr<c10d::Store> store,
      c10::Device device,
      std::shared_ptr<MscclppApi> api,
      std::chrono::milliseconds timeout);

  ~TorchCommMSCCLPPBootstrap() noexcept;

  // Delete copy/move
  TorchCommMSCCLPPBootstrap(const TorchCommMSCCLPPBootstrap&) = delete;
  TorchCommMSCCLPPBootstrap& operator=(const TorchCommMSCCLPPBootstrap&) =
      delete;

  /// Create and initialize the MSCCL++ communicator.
  std::shared_ptr<mscclpp::Communicator> createCommunicator(
      const std::string& name,
      const CommOptions& options = {});

  int getRank() const {
    return rank_;
  }
  int getSize() const {
    return size_;
  }

 private:
  /// Exchange UniqueId via c10d::Store (mirrors exchangeUniqueIdStore in
  /// TorchCommNCCLBootstrap).
  mscclpp::UniqueId exchangeUniqueId(const std::string& name);

  c10::intrusive_ptr<c10d::Store> store_;
  c10::Device device_;
  std::shared_ptr<MscclppApi> api_;
  std::chrono::milliseconds timeout_;
  int rank_;
  int size_;

  static int counter_;
};

} // namespace torch::comms

#endif // HAS_MSCCLPP
