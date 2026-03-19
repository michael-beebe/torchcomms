// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <stdexcept>
#include <string>

#include <ATen/ATen.h>
#include <comms/torchcomms/TorchCommTypes.hpp>

// Note: stream selection (getOperationStream) and GPU API calls require a
// GpuApi* pointer. Those helpers belong in TorchCommMSCCLPP.cpp
// where the gpu_api_ member is available, not in this header.

namespace torch::comms::mscclpp_utils {

/// Validate that a ReduceOp is supported by MSCCL++ (SUM only for now).
///
/// MSCCL++ execution plans encode the reduction op at algorithm-compile time.
/// Only SUM plans are currently provided. For other ops, throws
/// std::runtime_error with a message naming the calling collective.
///
/// Use ReduceOp::RedOpType directly (not the static const instances) to avoid
/// static initialization order issues in internal helpers.
inline void validateReduceOp(
    const ReduceOp& op,
    const std::string& collective_name) {
  if (op != ReduceOp::RedOpType::SUM) {
    throw std::runtime_error(
        "[TorchCommMSCCLPP] " + collective_name +
        " only supports SUM. "
        "MSCCL++ execution plans encode the reduction op at compile time. "
        "Got op type: " +
        std::to_string(static_cast<int>(op.type())));
  }
}

// MSCCL++ v0.8.0 executor hardcodes add_vectors<T>() in all reduce
// handlers — only SUM actually works. Other ops silently compute SUM.

/// Ensure a tensor is contiguous, returning a contiguous copy if needed.
/// Most MSCCL++ collective paths require contiguous device memory.
inline at::Tensor ensureContiguous(const at::Tensor& tensor) {
  return tensor.contiguous();
}

} // namespace torch::comms::mscclpp_utils
