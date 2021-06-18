'''
 Copyright 2020 Xilinx Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

'''
Utility functions
'''

'''
Author: Mark Harvey
'''

import os
import numpy as np


# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from tensorflow.keras.datasets import mnist



def mnist_download():
  '''
  MNIST dataset download and pre-processing
  Pixels are scaled to range 0.0 to 1.0
  Returns:
    - train & test data as numpy arrays
    - train & test labels as numpy arrays
  '''
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  # scale to (0,1)
  x_train = (x_train/255.0).astype(np.float32)
  x_test = (x_test/255.0).astype(np.float32)

  # add channel dimension
  x_train = x_train.reshape(x_train.shape[0],28,28,1)
  x_test = x_test.reshape(x_test.shape[0],28,28,1)

  # create sets of labels for each of the 10 classifier heads
  y_test0=(y_test==0).astype(np.uint8)
  y_test1=(y_test==1).astype(np.uint8)
  y_test2=(y_test==2).astype(np.uint8)
  y_test3=(y_test==3).astype(np.uint8)
  y_test4=(y_test==4).astype(np.uint8)
  y_test5=(y_test==5).astype(np.uint8)
  y_test6=(y_test==6).astype(np.uint8)
  y_test7=(y_test==7).astype(np.uint8)
  y_test8=(y_test==8).astype(np.uint8)
  y_test9=(y_test==9).astype(np.uint8)

  y_train0=(y_train==0).astype(np.uint8)
  y_train1=(y_train==1).astype(np.uint8)
  y_train2=(y_train==2).astype(np.uint8)
  y_train3=(y_train==3).astype(np.uint8)
  y_train4=(y_train==4).astype(np.uint8)
  y_train5=(y_train==5).astype(np.uint8)
  y_train6=(y_train==6).astype(np.uint8)
  y_train7=(y_train==7).astype(np.uint8)
  y_train8=(y_train==8).astype(np.uint8)
  y_train9=(y_train==9).astype(np.uint8)

  return x_train,  x_test, y_train, y_test, \
         y_train0, y_test0, \
         y_train1, y_test1, y_train2, y_test2, \
         y_train3, y_test3, y_train4, y_test4, \
         y_train5, y_test5, y_train6, y_test6, \
         y_train7, y_test7, y_train8, y_test8, \
         y_train9, y_test9




