# Copyright (c) OpenMMLab. All rights reserved.
"""This file holding some environment constant for sharing by other files."""

import os.path as osp
import subprocess
import sys
from collections import defaultdict

import cv2
import torch

import mmcv
from .parrots_wrapper import get_build_config


def collect_env():
    """Collect the information of the running environments.

    Returns:
        dict: The environment information. The following fields are contained.

            - sys.platform: The variable of ``sys.platform``.
            - Python: Python version.
            - MUSA available: Bool, indicating if MUSA is available.
            - GPU devices: Device type of each GPU.
            - MUSA_HOME (optional): The env var ``MUSA_HOME``.
            - NVCC (optional): NVCC version.
            - GCC: GCC version, "n/a" if GCC is not installed.
            - PyTorch: PyTorch version.
            - PyTorch compiling details: The output of \
                ``torch.__config__.show()``.
            - TorchVision (optional): TorchVision version.
            - OpenCV: OpenCV version.
            - MMCV: MMCV version.
            - MMCV Compiler: The GCC version for compiling MMCV ops.
            - MMCV MUSA Compiler: The MUSA version for compiling MMCV ops.
    """
    env_info = {}
    env_info['sys.platform'] = sys.platform
    env_info['Python'] = sys.version.replace('\n', '')

    musa_available = torch.musa.is_available()
    env_info['MUSA available'] = musa_available

    if musa_available:
        devices = defaultdict(list)
        for k in range(torch.musa.device_count()):
            devices[torch.musa.get_device_name(k)].append(str(k))
        for name, device_ids in devices.items():
            env_info['GPU ' + ','.join(device_ids)] = name

        from mmcv.utils.parrots_wrapper import _get_musa_home
        MUSA_HOME = _get_musa_home()
        env_info['MUSA_HOME'] = MUSA_HOME

        if MUSA_HOME is not None and osp.isdir(MUSA_HOME):
            try:
                mcc = osp.join(MUSA_HOME, 'bin/mcc')
                mcc = subprocess.check_output(
                    f'"{mcc}" -V | tail -n1', shell=True)
                mcc = mcc.decode('utf-8').strip()
            except subprocess.SubprocessError:
                mcc = 'Not Available'
            env_info['NVCC'] = mcc

    try:
        gcc = subprocess.check_output('gcc --version | head -n1', shell=True)
        gcc = gcc.decode('utf-8').strip()
        env_info['GCC'] = gcc
    except subprocess.CalledProcessError:  # gcc is unavailable
        env_info['GCC'] = 'n/a'

    env_info['PyTorch'] = torch.__version__
    env_info['PyTorch compiling details'] = get_build_config()

    try:
        import torchvision
        env_info['TorchVision'] = torchvision.__version__
    except ModuleNotFoundError:
        pass

    env_info['OpenCV'] = cv2.__version__

    env_info['MMCV'] = mmcv.__version__

    try:
        from mmcv.ops import get_compiler_version, get_compiling_musa_version
    except ModuleNotFoundError:
        env_info['MMCV Compiler'] = 'n/a'
        env_info['MMCV MUSA Compiler'] = 'n/a'
    else:
        env_info['MMCV Compiler'] = get_compiler_version()
        env_info['MMCV MUSA Compiler'] = get_compiling_musa_version()

    return env_info
