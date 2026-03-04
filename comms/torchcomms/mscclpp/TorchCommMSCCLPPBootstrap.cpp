// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef HAS_MSCCLPP

#include <comms/torchcomms/mscclpp/TorchCommMSCCLPPBootstrap.hpp>

#include <fmt/core.h>

#include <comms/torchcomms/StoreManager.hpp>
#include <comms/torchcomms/TorchCommLogging.hpp>
#include <comms/torchcomms/TorchCommUtils.hpp>

namespace torch::comms {

int TorchCommMSCCLPPBootstrap::counter_ = 0;

TorchCommMSCCLPPBootstrap::TorchCommMSCCLPPBootstrap(
    c10::intrusive_ptr<c10d::Store> store,
    c10::Device device,
    std::shared_ptr<MscclppApi> api,
    std::chrono::milliseconds timeout)
    : store_(std::move(store)),
      device_(device),
      api_(std::move(api)),
      timeout_(timeout) {
  auto [rank, size] = query_ranksize();
  rank_ = rank;
  size_ = size;
}

TorchCommMSCCLPPBootstrap::~TorchCommMSCCLPPBootstrap() noexcept = default;

mscclpp::UniqueId TorchCommMSCCLPPBootstrap::exchangeUniqueId(
    const std::string& name) {
  // Single-process: no coordination needed — generate the unique ID locally
  // and return immediately without touching the store.
  if (size_ == 1) {
    return api_->createUniqueId();
  }

  // Multi-process without a caller-supplied store: fall back to StoreManager
  // (same pattern as TorchCommNCCLBootstrap::exchangeUniqueIdTCPStore).
  if (!store_) {
    store_ = StoreManager::get().getStore("mscclpp", name, timeout_);
  }

  // Key format mirrors TorchCommNCCLBootstrap::getNCCLStoreKey().
  std::string key = fmt::format("mscclpp_uniqueid_{}{}", name, counter_++);

  mscclpp::UniqueId unique_id;

  if (rank_ == 0) {
    // Rank 0 generates the UniqueId and broadcasts it via the store.
    unique_id = api_->createUniqueId();
    std::vector<uint8_t> vec(unique_id.begin(), unique_id.end());
    store_->set(key, vec);
  } else {
    // Other ranks wait for rank 0, then read the UniqueId.
    store_->wait({key}, timeout_);
    auto vec = store_->get(key);
    if (vec.size() != sizeof(mscclpp::UniqueId)) {
      throw std::runtime_error(
          fmt::format(
              "[TorchCommMSCCLPPBootstrap] Invalid UniqueId size: expected {}, got {}",
              sizeof(mscclpp::UniqueId),
              vec.size()));
    }
    std::copy(vec.begin(), vec.end(), unique_id.begin());
  }

  return unique_id;
}

std::shared_ptr<mscclpp::Communicator>
TorchCommMSCCLPPBootstrap::createCommunicator(
    const std::string& name,
    const CommOptions& /*options*/) {
  // 1. Exchange UniqueId via store (rank 0 generates, all ranks receive)
  mscclpp::UniqueId unique_id = exchangeUniqueId(name);

  // 2. Create TcpBootstrap and initialize all ranks with the same UniqueId.
  //    For size==1 we skip initialize() — there are no peers to rendezvous
  //    with, and the TcpBootstrap constructor produces an unconnected object
  //    that is still usable as a handle for Communicator creation.
  auto bootstrap = api_->createTcpBootstrap(rank_, size_);
  if (size_ > 1) {
    int64_t timeout_sec = std::max(
        int64_t{1},
        std::chrono::duration_cast<std::chrono::seconds>(timeout_).count());
    api_->bootstrapInitialize(*bootstrap, unique_id, timeout_sec);
  }

  // 3. Create communicator
  auto comm = api_->createCommunicator(bootstrap);

  TC_LOG(INFO) << "[TorchCommMSCCLPP] Communicator created: name=" << name
               << " rank=" << rank_ << "/" << size_ << " device=" << device_;

  return comm;
}

} // namespace torch::comms

#endif // HAS_MSCCLPP
