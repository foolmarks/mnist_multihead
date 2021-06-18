#!/bin/bash
# Copyright 2020 Xilinx Inc.

conda activate vitis-ai-tensorflow2


# fetch patches
wget --no-clobber https://www.xilinx.com/bin/public/openDownload?filename=unilog-1.3.2-h7b12538_35.tar.bz2 -O unilog-1.3.2-h7b12538_35.tar.bz2
wget --no-clobber https://www.xilinx.com/bin/public/openDownload?filename=target_factory-1.3.2-hf484d3e_35.tar.bz2 -O target_factory-1.3.2-hf484d3e_35.tar.bz2
wget --no-clobber https://www.xilinx.com/bin/public/openDownload?filename=xir-1.3.2-py37h7b12538_47.tar.bz2 -O xir-1.3.2-py37h7b12538_47.tar.bz2
wget --no-clobber https://www.xilinx.com/bin/public/openDownload?filename=xcompiler-1.3.2-py37h7b12538_53.tar.bz2 -O xcompiler-1.3.2-py37h7b12538_53.tar.bz2
wget --no-clobber https://www.xilinx.com/bin/public/openDownload?filename=xnnc-1.3.2-py37_48.tar.bz2 -O xnnc-1.3.2-py37_48.tar.bz2


# install patches
# MUST MAINTAIN THIS INSTALL ORDER!
sudo env PATH=/opt/vitis_ai/conda/bin:$PATH CONDA_PREFIX=/opt/vitis_ai/conda/envs/vitis-ai-tensorflow2 conda install unilog-1.3.2-h7b12538_35.tar.bz2
sudo env PATH=/opt/vitis_ai/conda/bin:$PATH CONDA_PREFIX=/opt/vitis_ai/conda/envs/vitis-ai-tensorflow2 conda install target_factory-1.3.2-hf484d3e_35.tar.bz2
sudo env PATH=/opt/vitis_ai/conda/bin:$PATH CONDA_PREFIX=/opt/vitis_ai/conda/envs/vitis-ai-tensorflow2 conda install xir-1.3.2-py37h7b12538_47.tar.bz2
sudo env PATH=/opt/vitis_ai/conda/bin:$PATH CONDA_PREFIX=/opt/vitis_ai/conda/envs/vitis-ai-tensorflow2 conda install xcompiler-1.3.2-py37h7b12538_53.tar.bz2
sudo env PATH=/opt/vitis_ai/conda/bin:$PATH CONDA_PREFIX=/opt/vitis_ai/conda/envs/vitis-ai-tensorflow2 conda install xnnc-1.3.2-py37_48.tar.bz2

