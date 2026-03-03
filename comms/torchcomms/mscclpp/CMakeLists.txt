# Copyright (c) Meta Platforms, Inc. and affiliates.
# Extension: torchcomms._comms_mscclpp
#
# Builds on both CUDA (NVIDIA) and ROCm (AMD). The IS_ROCM option is
# propagated from setup.py via PyTorch's torch.version.hip detection.

file(GLOB TORCHCOMMS_MSCCLPP_SOURCES "comms/torchcomms/mscclpp/*.cpp")

# --- Platform detection ---
# IS_ROCM is set by setup.py. On ROCm builds, find HIP instead of CUDA.
if(IS_ROCM)
    message(STATUS "MSCCL++ backend: building for ROCm/HIP")
    # ROCm ships its own CMake config
    list(APPEND CMAKE_PREFIX_PATH "/opt/rocm" "$ENV{ROCM_HOME}")
    find_package(hip REQUIRED)
else()
    message(STATUS "MSCCL++ backend: building for CUDA")
    find_package(CUDA)
    # Also pull in CudaApi.cpp for CUDA builds
    file(GLOB TORCHCOMMS_CUDA_API_SOURCE "comms/torchcomms/device/cuda/CudaApi.cpp")
    list(APPEND TORCHCOMMS_MSCCLPP_SOURCES ${TORCHCOMMS_CUDA_API_SOURCE})
endif()

# --- Locate MSCCL++ library ---
# Priority: MSCCLPP_HOME env var > system find_package
if(DEFINED ENV{MSCCLPP_HOME})
    set(MSCCLPP_HOME "$ENV{MSCCLPP_HOME}")
    set(MSCCLPP_INCLUDE "${MSCCLPP_HOME}/include")
    set(MSCCLPP_LIB_DIR "${MSCCLPP_HOME}/lib")
    message(STATUS "Using MSCCL++ from MSCCLPP_HOME: ${MSCCLPP_HOME}")
else()
    # Try find_package; downstream users can set CMAKE_PREFIX_PATH
    find_package(mscclpp QUIET)
    if(mscclpp_FOUND)
        message(STATUS "Found MSCCL++ via find_package")
        get_target_property(MSCCLPP_INCLUDE mscclpp::mscclpp INTERFACE_INCLUDE_DIRECTORIES)
        set(MSCCLPP_LIB_DIR "")  # linked via target
    else()
        message(FATAL_ERROR
          "MSCCL++ not found. Set MSCCLPP_HOME or install mscclpp and add to CMAKE_PREFIX_PATH.\n"
          "  git clone https://github.com/microsoft/mscclpp && cd mscclpp\n"
          "  mkdir build && cd build && cmake .. -DCMAKE_INSTALL_PREFIX=<prefix>\n"
          "  make -j && make install\n"
          "  export MSCCLPP_HOME=<prefix>")
    endif()
endif()

set(FMT_INCLUDE "${ROOT}/third-party/fmt/include")

add_library(torchcomms_comms_mscclpp MODULE
    ${TORCHCOMMS_MSCCLPP_SOURCES}
)
set_target_properties(torchcomms_comms_mscclpp PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_comms_mscclpp"
    SUFFIX ".${Python3_SOABI}.so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/comms/torchcomms"
)
target_include_directories(torchcomms_comms_mscclpp PRIVATE
    ${ROOT}
    ${MSCCLPP_INCLUDE}
    ${FMT_INCLUDE}
    ${CONDA_INCLUDE}
    ${Python3_INCLUDE_DIRS}
)
target_compile_features(torchcomms_comms_mscclpp PRIVATE cxx_std_20)

# Propagate USE_ROCM to source code so GpuApi.hpp picks the right branch
if(IS_ROCM)
    target_compile_definitions(torchcomms_comms_mscclpp PRIVATE USE_ROCM)
endif()

target_link_directories(torchcomms_comms_mscclpp PRIVATE
    ${CONDA_LIB}
)
target_link_libraries(torchcomms_comms_mscclpp PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIB}
    torchcomms
)

# Link MSCCL++ — either via target or explicit path
if(mscclpp_FOUND)
    target_link_libraries(torchcomms_comms_mscclpp PRIVATE mscclpp::mscclpp)
else()
    target_link_directories(torchcomms_comms_mscclpp PRIVATE ${MSCCLPP_LIB_DIR})
    target_link_libraries(torchcomms_comms_mscclpp PRIVATE mscclpp)
endif()

if(USE_SYSTEM_LIBS)
    target_link_libraries(torchcomms_comms_mscclpp PRIVATE "-lfmt")
else()
    target_link_libraries(torchcomms_comms_mscclpp PRIVATE "-l:libfmt.a")
endif()

# Link GPU runtime
if(IS_ROCM)
    target_link_libraries(torchcomms_comms_mscclpp PRIVATE hip::device)
elseif(CUDA_FOUND)
    target_link_libraries(torchcomms_comms_mscclpp PRIVATE CUDA::cudart)
endif()

install(TARGETS torchcomms_comms_mscclpp LIBRARY DESTINATION .)
