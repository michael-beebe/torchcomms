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

# --- Locate MSCCL++ library (optional for stub builds) ---
# Priority: MSCCLPP_HOME env var > system find_package > stub-only (no mscclpp)
set(MSCCLPP_FOUND FALSE)
if(DEFINED ENV{MSCCLPP_HOME})
    set(MSCCLPP_HOME "$ENV{MSCCLPP_HOME}")
    set(MSCCLPP_LIB_DIR "${MSCCLPP_HOME}/lib")
    # Locate headers: prefer ${MSCCLPP_HOME}/include, but fall back to
    # ${MSCCLPP_HOME}/../include for in-tree builds where the user points
    # MSCCLPP_HOME at the build directory (libs in build/lib, headers in
    # the source tree's include/).
    if(EXISTS "${MSCCLPP_HOME}/include/mscclpp/core.hpp")
        set(MSCCLPP_INCLUDE "${MSCCLPP_HOME}/include")
    elseif(EXISTS "${MSCCLPP_HOME}/../include/mscclpp/core.hpp")
        set(MSCCLPP_INCLUDE "${MSCCLPP_HOME}/../include")
    else()
        message(FATAL_ERROR "MSCCL++ headers not found under ${MSCCLPP_HOME}/include or ${MSCCLPP_HOME}/../include")
    endif()
    # Generated headers (e.g. version.hpp) live in ${MSCCLPP_HOME}/include.
    set(MSCCLPP_BUILD_INCLUDE "${MSCCLPP_HOME}/include")
    set(MSCCLPP_FOUND TRUE)
    message(STATUS "Using MSCCL++ from MSCCLPP_HOME: ${MSCCLPP_HOME}")
    message(STATUS "Using MSCCL++ headers from: ${MSCCLPP_INCLUDE}")
else()
    # Try find_package; downstream users can set CMAKE_PREFIX_PATH
    find_package(mscclpp QUIET)
    if(mscclpp_FOUND)
        message(STATUS "Found MSCCL++ via find_package")
        get_target_property(MSCCLPP_INCLUDE mscclpp::mscclpp INTERFACE_INCLUDE_DIRECTORIES)
        set(MSCCLPP_LIB_DIR "")  # linked via target
        set(MSCCLPP_FOUND TRUE)
    else()
        message(STATUS
          "MSCCL++ not found — building stub-only (no collectives).\n"
          "  To enable full MSCCL++ support, set MSCCLPP_HOME or install mscclpp:\n"
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
    ${FMT_INCLUDE}
    ${CONDA_INCLUDE}
    ${Python3_INCLUDE_DIRS}
)
if(MSCCLPP_FOUND)
    target_include_directories(torchcomms_comms_mscclpp PRIVATE ${MSCCLPP_INCLUDE})
    if(DEFINED MSCCLPP_BUILD_INCLUDE AND NOT MSCCLPP_BUILD_INCLUDE STREQUAL MSCCLPP_INCLUDE)
        target_include_directories(torchcomms_comms_mscclpp PRIVATE ${MSCCLPP_BUILD_INCLUDE})
    endif()
    target_compile_definitions(torchcomms_comms_mscclpp PRIVATE HAS_MSCCLPP)
endif()
target_compile_features(torchcomms_comms_mscclpp PRIVATE cxx_std_20)

# Propagate USE_ROCM to source code so GpuTypes.hpp picks the right branch
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

# Link MSCCL++ — either via target or explicit path (only if found)
if(MSCCLPP_FOUND)
    if(mscclpp_FOUND)
        target_link_libraries(torchcomms_comms_mscclpp PRIVATE mscclpp::mscclpp)
    else()
        target_link_directories(torchcomms_comms_mscclpp PRIVATE ${MSCCLPP_LIB_DIR})
        target_link_libraries(torchcomms_comms_mscclpp PRIVATE mscclpp)
    endif()
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

# Tests — only built when -DBUILD_TESTS=ON and MSCCL++ is available.
# Tests live in tests/ alongside the source they exercise rather than in the
# generic comms/torchcomms/tests/unit hierarchy.
if(BUILD_TESTS AND MSCCLPP_FOUND)
    enable_testing()
    # Reuse the same GoogleTest that the generic test suite fetches.
    # FetchContent is idempotent: if tests/unit/cpp already fetched it, this
    # is a no-op; if mscclpp tests are built standalone, this fetches it.
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # MscclppEventPoolTest — GPU event pool and TorchWorkMSCCLPP lifecycle.
    # TorchWorkMSCCLPP.cpp and CudaApi.cpp are compiled directly into the
    # binary because torchcomms_comms_mscclpp is a MODULE (not linkable).
    add_executable(MscclppEventPoolTest
        ${ROOT}/comms/torchcomms/mscclpp/tests/MscclppEventPoolTest.cpp
        ${ROOT}/comms/torchcomms/mscclpp/TorchWorkMSCCLPP.cpp
        ${ROOT}/comms/torchcomms/device/cuda/CudaApi.cpp
    )
    target_include_directories(MscclppEventPoolTest PRIVATE
        ${ROOT}
        ${MSCCLPP_INCLUDE}
        ${CONDA_INCLUDE}
        ${CUDA_INCLUDE_DIRS}
    )
    target_compile_features(MscclppEventPoolTest PRIVATE cxx_std_20)
    target_compile_definitions(MscclppEventPoolTest PRIVATE HAS_MSCCLPP)
    target_link_directories(MscclppEventPoolTest PRIVATE ${CONDA_LIB})
    target_link_libraries(MscclppEventPoolTest PRIVATE
        torchcomms
        ${TORCH_LIBRARIES}
        CUDA::cudart
        GTest::gtest
        GTest::gmock
        "-lglog"
    )
    add_test(NAME MscclppEventPoolTest COMMAND MscclppEventPoolTest)

    # MscclppPlanTest — plan selection and loading unit tests.
    # Exercises selectPlan() / loadPlans() via a friend fixture without needing
    # a real MSCCL++ communicator or CUDA device.
    add_executable(MscclppPlanTest
        ${ROOT}/comms/torchcomms/mscclpp/tests/MscclppPlanTest.cpp
        ${ROOT}/comms/torchcomms/mscclpp/TorchCommMSCCLPP.cpp
        ${ROOT}/comms/torchcomms/mscclpp/TorchCommMSCCLPPBootstrap.cpp
        ${ROOT}/comms/torchcomms/mscclpp/MscclppApi.cpp
        ${ROOT}/comms/torchcomms/mscclpp/TorchWorkMSCCLPP.cpp
        ${ROOT}/comms/torchcomms/device/cuda/CudaApi.cpp
    )
    target_include_directories(MscclppPlanTest PRIVATE
        ${ROOT}
        ${MSCCLPP_INCLUDE}
        ${CONDA_INCLUDE}
        ${CUDA_INCLUDE_DIRS}
    )
    if(DEFINED MSCCLPP_BUILD_INCLUDE AND NOT MSCCLPP_BUILD_INCLUDE STREQUAL MSCCLPP_INCLUDE)
        target_include_directories(MscclppPlanTest PRIVATE ${MSCCLPP_BUILD_INCLUDE})
    endif()
    target_compile_features(MscclppPlanTest PRIVATE cxx_std_20)
    target_compile_definitions(MscclppPlanTest PRIVATE HAS_MSCCLPP)
    target_link_directories(MscclppPlanTest PRIVATE ${CONDA_LIB})
    if(MSCCLPP_FOUND AND NOT mscclpp_FOUND)
        target_link_directories(MscclppPlanTest PRIVATE ${MSCCLPP_LIB_DIR})
        target_link_libraries(MscclppPlanTest PRIVATE mscclpp)
    else()
        target_link_libraries(MscclppPlanTest PRIVATE mscclpp::mscclpp)
    endif()
    target_link_libraries(MscclppPlanTest PRIVATE
        torchcomms
        ${TORCH_LIBRARIES}
        CUDA::cudart
        GTest::gtest
        "-lglog"
    )
    if(USE_SYSTEM_LIBS)
        target_link_libraries(MscclppPlanTest PRIVATE "-lfmt")
    else()
        target_link_libraries(MscclppPlanTest PRIVATE "-l:libfmt.a")
    endif()
    add_test(NAME MscclppPlanTest COMMAND MscclppPlanTest)
endif()
