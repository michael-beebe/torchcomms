// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include <stdexcept>
#include <string>

// mscclpp::DataType is only available when MSCCL++ headers are present.
// DtypeMap.hpp is a no-op when building the MSCCL++ stub (no MSCCLPP_HOME set).
#ifdef HAS_MSCCLPP

#include <mscclpp/gpu_data_types.hpp>

namespace torch::comms {

/// Map PyTorch scalar types to MSCCL++ data types.
/// Throws std::runtime_error if the dtype is not supported by the MSCCL++
/// executor. Callers should invoke this once per collective and cache the
/// result rather than calling it on every iteration.
inline mscclpp::DataType torchDtypeToMscclpp(at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat:
      return mscclpp::DataType::FLOAT32;
    case at::kHalf:
      return mscclpp::DataType::FLOAT16;
    case at::kBFloat16:
      return mscclpp::DataType::BFLOAT16;
    case at::kInt:
      return mscclpp::DataType::INT32;
    case at::kUInt32:
      return mscclpp::DataType::UINT32;
    default:
      throw std::runtime_error(
          "[TorchCommMSCCLPP] Unsupported tensor dtype for MSCCL++ executor: " +
          std::string(at::toString(dtype)) +
          ". Supported: float32, float16, bfloat16, int32, uint32.");
  }
}

} // namespace torch::comms

#endif // HAS_MSCCLPP
