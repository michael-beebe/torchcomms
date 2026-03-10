// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <sstream>
#include <stdexcept>

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

namespace torch::comms::mscclpp_gpu {

using GpuApi = HipApi;
using DefaultGpuApi = DefaultHipApi;
using gpuStream_t = hipStream_t;
using gpuEvent_t = hipEvent_t;
using gpuError_t = hipError_t;
inline constexpr gpuError_t gpuSuccess = hipSuccess;
inline constexpr gpuError_t gpuErrorNotReady = hipErrorNotReady;
inline constexpr unsigned int gpuStreamNonBlocking = hipStreamNonBlocking;
inline constexpr unsigned int gpuEventDisableTiming = hipEventDisableTiming;
inline constexpr auto gpuMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;

} // namespace torch::comms::mscclpp_gpu

#else // CUDA

#include <comms/torchcomms/device/cuda/CudaApi.hpp>
#include <cuda_runtime.h>

namespace torch::comms::mscclpp_gpu {

using GpuApi = CudaApi;
using DefaultGpuApi = DefaultCudaApi;
using gpuStream_t = cudaStream_t;
using gpuEvent_t = cudaEvent_t;
using gpuError_t = cudaError_t;
inline constexpr gpuError_t gpuSuccess = cudaSuccess;
inline constexpr gpuError_t gpuErrorNotReady = cudaErrorNotReady;
inline constexpr unsigned int gpuStreamNonBlocking = cudaStreamNonBlocking;
inline constexpr unsigned int gpuEventDisableTiming = cudaEventDisableTiming;
inline constexpr auto gpuMemcpyDeviceToDevice = cudaMemcpyDeviceToDevice;

} // namespace torch::comms::mscclpp_gpu

// Platform-neutral error-checking macro for GPU API calls.
// Uses mscclpp_gpu::gpuError_t / gpuSuccess so it works on both CUDA and ROCm.
// Matches the CUDA_CHECK / HIP_CHECK pattern from CudaApi.hpp / HipApi.hpp.
#define GPU_CHECK(gpu_api, call, err_str)                                \
  do {                                                                   \
    torch::comms::mscclpp_gpu::gpuError_t status = call;                 \
    if (status != torch::comms::mscclpp_gpu::gpuSuccess) {               \
      std::stringstream ss;                                              \
      ss << err_str << ": " << gpu_api->getErrorString(status) << " at " \
         << __FILE__ << ":" << __LINE__;                                 \
      throw std::runtime_error(ss.str());                                \
    }                                                                    \
  } while (0)

#endif // USE_ROCM
