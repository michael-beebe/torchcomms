// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

// Platform-neutral GPU type aliases for the MSCCL++ backend.
//
// Reuses the existing CudaApi / HipApi abstract classes (DI pattern)
// already used by the NCCL and RCCL backends. No new wrapper functions.
//
// USE_ROCM is set by CMake when building for ROCm
// (detected via PyTorch's torch.version.hip).

#ifdef USE_ROCM

#include <comms/torchcomms/rccl/HipApi.hpp>
#include <hip/hip_runtime.h>

namespace torch::comms::mscclpp_detail {

using GpuApi = HipApi;
using DefaultGpuApi = DefaultHipApi;
using gpuStream_t = hipStream_t;
using gpuEvent_t = hipEvent_t;
using gpuError_t = hipError_t;
inline constexpr gpuError_t gpuSuccess = hipSuccess;
inline constexpr gpuError_t gpuErrorNotReady = hipErrorNotReady;

} // namespace torch::comms::mscclpp_detail

#else // CUDA

#include <comms/torchcomms/device/cuda/CudaApi.hpp>
#include <cuda_runtime.h>

namespace torch::comms::mscclpp_detail {

using GpuApi = CudaApi;
using DefaultGpuApi = DefaultCudaApi;
using gpuStream_t = cudaStream_t;
using gpuEvent_t = cudaEvent_t;
using gpuError_t = cudaError_t;
inline constexpr gpuError_t gpuSuccess = cudaSuccess;
inline constexpr gpuError_t gpuErrorNotReady = cudaErrorNotReady;

} // namespace torch::comms::mscclpp_detail

#endif // USE_ROCM
