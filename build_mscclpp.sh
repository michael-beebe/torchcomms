#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Build MSCCL++ from source for use with torchcomms.
# Supports NVIDIA (CUDA) and AMD (ROCm) platforms.
#
# Usage:
#   ./build_mscclpp.sh                    # Default: build for CUDA
#   ./build_mscclpp.sh --rocm             # Build for ROCm
#   ./build_mscclpp.sh --clean            # Clean rebuild
#   ./build_mscclpp.sh --tag v0.5.2       # Pin to a specific release
#
# After building:
#   export MSCCLPP_HOME=$PWD/build/mscclpp/install
#   USE_MSCCLPP=ON pip install --no-build-isolation -v .

set -euo pipefail

# --- Defaults ---
MSCCLPP_REPO="https://github.com/microsoft/mscclpp.git"
MSCCLPP_TAG="${MSCCLPP_TAG:-v0.5.2}"  # Pin a release; override with --tag
BUILDDIR="${BUILDDIR:-${PWD}/build/mscclpp}"
INSTALL_PREFIX="${MSCCLPP_HOME:-${BUILDDIR}/install}"
CLEAN_BUILD=0
USE_ROCM=0
JOBS="$(nproc)"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
  case $1 in
    --rocm)       USE_ROCM=1;      shift ;;
    --clean)      CLEAN_BUILD=1;   shift ;;
    --tag)        MSCCLPP_TAG="$2"; shift 2 ;;
    --jobs|-j)    JOBS="$2";        shift 2 ;;
    --help|-h)
      echo "Usage: $0 [--rocm] [--clean] [--tag <git-tag>] [--jobs N]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

echo "=== MSCCL++ build ==="
echo "  Tag:            ${MSCCLPP_TAG}"
echo "  Build dir:      ${BUILDDIR}"
echo "  Install prefix: ${INSTALL_PREFIX}"
echo "  ROCm:           ${USE_ROCM}"
echo "  Jobs:           ${JOBS}"
echo ""

# --- Clean ---
if [[ "${CLEAN_BUILD}" == "1" ]]; then
  echo "Cleaning previous build..."
  rm -rf "${BUILDDIR}"
fi

mkdir -p "${BUILDDIR}"

# --- Clone ---
SRC_DIR="${BUILDDIR}/src"
if [[ ! -d "${SRC_DIR}" ]]; then
  echo "Cloning MSCCL++ (${MSCCLPP_TAG})..."
  git clone --depth 1 -b "${MSCCLPP_TAG}" "${MSCCLPP_REPO}" "${SRC_DIR}"
else
  echo "Source already present at ${SRC_DIR}, skipping clone."
  echo "  (Use --clean to force a fresh clone)"
fi

# --- Build ---
BUILD_OUT="${BUILDDIR}/build"
mkdir -p "${BUILD_OUT}"
cd "${BUILD_OUT}"

CMAKE_ARGS=(
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_TESTS=OFF
  -DBUILD_PYTHON_BINDINGS=OFF
)

if [[ "${USE_ROCM}" == "1" ]]; then
  # ROCm build
  ROCM_HOME="${ROCM_HOME:-/opt/rocm}"
  echo "Building for ROCm (ROCM_HOME=${ROCM_HOME})..."
  CMAKE_ARGS+=(
    -DMSCCLPP_USE_ROCM=ON
    -DCMAKE_PREFIX_PATH="${ROCM_HOME}"
  )
else
  # CUDA build
  CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  echo "Building for CUDA (CUDA_HOME=${CUDA_HOME})..."
  CMAKE_ARGS+=(
    -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}"
  )
fi

cmake "${CMAKE_ARGS[@]}" "${SRC_DIR}"
make -j"${JOBS}"
make install

echo ""
echo "=== MSCCL++ installed to: ${INSTALL_PREFIX} ==="
echo ""
echo "To use with torchcomms:"
echo "  export MSCCLPP_HOME=${INSTALL_PREFIX}"
echo "  USE_MSCCLPP=ON pip install --no-build-isolation -v ."
