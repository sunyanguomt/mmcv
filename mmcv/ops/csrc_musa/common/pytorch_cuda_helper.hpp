#ifndef PYTORCH_CUDA_HELPER
#define PYTORCH_CUDA_HELPER

#include <ATen/ATen.h>
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

#include <ATen/musa/MUSA_PORT_ApplyUtils.muh>
#include <THC/THCAtomics.muh>

#include "common_cuda_helper.hpp"

using at::Half;
using at::Tensor;
using phalf = at::Half;

#define __PHALF(x) (x)

#endif  // PYTORCH_CUDA_HELPER
