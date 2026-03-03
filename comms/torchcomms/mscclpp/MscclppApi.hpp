// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

// TODO: Placeholder
//
// MscclppApi provides an injectable abstraction over the MSCCL++ C++ API,
// following the NcclApi pattern with mutex-protected calls.
// DefaultMscclppApi wraps real mscclpp:: calls; a mock implementation
// enables unit testing without MSCCL++ present at test time.
