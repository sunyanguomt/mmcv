// Copyright (c) OpenMMLab. All rights reserved
#include "bbox_overlaps_cuda_kernel.muh"
#include "pytorch_cuda_helper.hpp"

void BBoxOverlapsCUDAKernelLauncher(const Tensor bboxes1, const Tensor bboxes2,
                                    Tensor ious, const int mode,
                                    const bool aligned, const int offset) {
  int output_size = ious.numel();
  int num_bbox1 = bboxes1.size(0);
  int num_bbox2 = bboxes2.size(0);

  at::musa::MUSAGuard device_guard(bboxes1.device());
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  AT_DISPATCH_FLOATING_TYPES(
      bboxes1.scalar_type(), "bbox_overlaps_cuda_kernel", ([&] {
        bbox_overlaps_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                bboxes1.data_ptr<scalar_t>(), bboxes2.data_ptr<scalar_t>(),
                ious.data_ptr<scalar_t>(), num_bbox1, num_bbox2, mode, aligned,
                offset);
      }));
  AT_MUSA_CHECK(musaGetLastError());
}
