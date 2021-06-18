#!/bin/sh

# Copyright 2020 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Mark Harvey, Xilinx Inc

if [ $1 = zcu102 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json
      target=ZCU102
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ZYNQ ZCU102.."
      echo "-----------------------------------------"
elif [ $1 = zcu104 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json
      target=ZCU104
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ZYNQ ZCU104.."
      echo "-----------------------------------------"
elif [ $1 = u50 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCAHX8H/U50/arch.json
      target=U50
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ALVEO U50.."
      echo "-----------------------------------------"
elif [ $1 = vck190 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK90/arch.json
      target=VCK190
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR VERSAL VK190.."
      echo "-----------------------------------------"
else
      echo  "Target not found. Valid choices are: zcu102, zcu104 u50, vck190 ..exiting"
      exit 1
fi

compile() {
      vai_c_tensorflow2 \
            --model           quant_model/q_model.h5 \
            --arch            $ARCH \
            --output_dir      compiled_model_$target \
            --net_name        customcnn
}


compile 2>&1 | tee logs/compile.log


echo "-----------------------------------------"
echo "MODEL COMPILED"
echo "-----------------------------------------"

