#ifndef PYTORCH_CUDA_HELPER
#define PYTORCH_CUDA_HELPER

#include <ATen/ATen.h>
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"


#if defined(TORCH_MUSA_2) && TORCH_MUSA_2
#pragma message("[MUSA HELPER] Using <ATen/musa/MUSAApplyUtils.muh> for torch_musa > 2.0.0")
#include <ATen/musa/MUSAApplyUtils.muh>]
#else
#pragma message("[MUSA HELPER] Using <ATen/musa/MUSA_PORT_ApplyUtils.muh> for torch_musa <= 2.0.0")
#include <ATen/musa/MUSA_PORT_ApplyUtils.muh>
#endif
#include <THC/THCAtomics.muh>

#include "common_cuda_helper.hpp"

using at::Half;
using at::Tensor;
using phalf = at::Half;

#define __PHALF(x) (x)

#endif  // PYTORCH_CUDA_HELPER
